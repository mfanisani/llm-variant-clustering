from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from .clustering import ClusteringResult, weighted_kmeans
from .evaluation import evaluate_partition
from .features import FeatureSet


@dataclass(frozen=True)
class DynamicKConfig:
    min_k: int = 2
    max_k: int = 12
    min_seed_stability_ari: float = 0.80
    min_case_cluster_entropy: float = 0.75
    min_cluster_case_fraction: float = 0.02
    min_cluster_variants: int = 2
    score_profiles: tuple[tuple[str, float, float, float], ...] = (
        ("balanced", 0.40, 0.40, 0.20),
        ("stability_heavy", 0.30, 0.50, 0.20),
        ("separation_heavy", 0.50, 0.30, 0.20),
        ("balance_heavy", 0.30, 0.30, 0.40),
    )

    def validate(self) -> None:
        if self.min_k < 2 or self.max_k < self.min_k:
            raise ValueError("Invalid dynamic K range")
        for value in (
            self.min_seed_stability_ari,
            self.min_case_cluster_entropy,
            self.min_cluster_case_fraction,
        ):
            if not 0 <= value <= 1:
                raise ValueError("Dynamic K thresholds must be between zero and one")
        if self.min_cluster_variants < 1:
            raise ValueError("min_cluster_variants must be positive")
        names = [profile[0] for profile in self.score_profiles]
        if len(names) != len(set(names)) or "balanced" not in names:
            raise ValueError("Score profiles require unique names and balanced profile")
        for _, separation, stability, balance in self.score_profiles:
            if not np.isclose(separation + stability + balance, 1.0):
                raise ValueError("Score profile weights must sum to one")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["score_profiles"] = [list(profile) for profile in self.score_profiles]
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DynamicKConfig":
        converted = dict(value)
        converted["score_profiles"] = tuple(
            tuple(profile) for profile in converted["score_profiles"]
        )
        config = cls(**converted)
        config.validate()
        return config


@dataclass(frozen=True)
class DynamicKSelection:
    selected_k: int
    selected_result: ClusteringResult
    candidates: pd.DataFrame
    profile_votes: dict[str, int]


class NoEligibleKError(ValueError):
    def __init__(self, candidates: pd.DataFrame) -> None:
        super().__init__(
            "No K satisfies the frozen stability, balance, and minimum-size gates"
        )
        self.candidates = candidates


def select_dynamic_k(
    features: FeatureSet,
    weights: np.ndarray,
    config: DynamicKConfig,
    *,
    distances: np.ndarray | None = None,
) -> DynamicKSelection:
    config.validate()
    weights = np.asarray(weights, dtype=np.float64)
    max_k = min(config.max_k, features.matrix.shape[0] - 1)
    if config.min_k > max_k:
        raise ValueError("Not enough variants for the configured K range")

    results: dict[int, ClusteringResult] = {}
    rows = []
    if distances is None:
        distances = pairwise_distances(features.matrix, metric=features.metric)
    elif distances.shape != (features.matrix.shape[0], features.matrix.shape[0]):
        raise ValueError("Precomputed distances do not match feature rows")
    minimum_cases = max(
        1,
        int(np.ceil(weights.sum() * config.min_cluster_case_fraction)),
    )
    for k in range(config.min_k, max_k + 1):
        result = weighted_kmeans(
            features.matrix,
            weights,
            n_clusters=k,
        )
        metrics = evaluate_partition(
            features.matrix,
            result.labels,
            weights,
            metric=features.metric,
            distances=distances,
        )
        eligible = (
            result.mean_seed_stability_ari >= config.min_seed_stability_ari
            and metrics["case_cluster_entropy"]
            >= config.min_case_cluster_entropy
            and metrics["min_cluster_cases"] >= minimum_cases
            and metrics["min_cluster_variants"] >= config.min_cluster_variants
        )
        rows.append(
            {
                "k": k,
                "eligible": eligible,
                "selected_seed": result.selected_seed,
                "weighted_inertia": result.weighted_inertia,
                "mean_seed_stability_ari": result.mean_seed_stability_ari,
                **metrics,
            }
        )
        results[k] = result
    candidates = pd.DataFrame(rows)
    eligible = candidates[candidates["eligible"]].copy()
    if eligible.empty:
        raise NoEligibleKError(candidates)

    votes: dict[str, int] = {}
    for name, separation_weight, stability_weight, balance_weight in (
        config.score_profiles
    ):
        eligible[f"score_{name}"] = (
            separation_weight
            * ((eligible["case_weighted_silhouette"] + 1.0) / 2.0)
            + stability_weight * eligible["mean_seed_stability_ari"]
            + balance_weight * eligible["case_cluster_entropy"]
        )
        winner = eligible.sort_values(
            [f"score_{name}", "k"],
            ascending=[False, True],
        ).iloc[0]
        votes[name] = int(winner["k"])

    vote_counts = pd.Series(votes).value_counts()
    most_votes = int(vote_counts.max())
    finalists = set(vote_counts[vote_counts.eq(most_votes)].index.astype(int))
    selected_row = (
        eligible[eligible["k"].isin(finalists)]
        .sort_values(["score_balanced", "k"], ascending=[False, True])
        .iloc[0]
    )
    selected_k = int(selected_row["k"])
    for column in eligible.columns:
        if column.startswith("score_"):
            candidates.loc[
                candidates["k"].isin(eligible["k"]),
                column,
            ] = eligible.set_index("k")[column].reindex(
                candidates.loc[candidates["k"].isin(eligible["k"]), "k"]
            ).to_numpy()
    candidates["selected"] = candidates["k"].eq(selected_k)
    return DynamicKSelection(
        selected_k=selected_k,
        selected_result=results[selected_k],
        candidates=candidates,
        profile_votes=votes,
    )
