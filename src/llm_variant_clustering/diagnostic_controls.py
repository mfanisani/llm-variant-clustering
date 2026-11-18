from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .clustering import weighted_kmeans
from .embeddings import embedding_feature_set
from .evaluation import evaluate_partition
from .features import parse_sequences
from .semantic_compiler import ordered_semantic_pyramid

BASELINE_NAME = "baseline_p7_ordered"
ACTIVITY_PERMUTATION_NAME = "activity_identity_permutation"
WITHIN_VARIANT_SHUFFLE_NAME = "within_variant_shuffle"
RANDOM_EMBEDDING_NAME = "random_gaussian_activity_embeddings"


def load_activity_embedding_map(
    index_path: Path,
    embeddings_path: Path,
) -> dict[str, np.ndarray]:
    """Load activity vectors using the explicit row order in their index."""
    index = pd.read_parquet(index_path)
    if "activity_label" not in index:
        raise ValueError("Activity embedding index is missing activity_label")
    labels = index["activity_label"]
    if labels.isna().any() or labels.astype(str).duplicated().any():
        raise ValueError("Activity embedding labels must be non-null and unique")

    matrix = np.load(embeddings_path, allow_pickle=False)
    if matrix.ndim != 2 or matrix.shape[0] != len(index):
        raise ValueError(
            "Activity embedding index and compact embedding rows do not align"
        )
    if matrix.shape[1] < 1 or not np.isfinite(matrix).all():
        raise ValueError("Activity embeddings must be finite two-dimensional data")
    return {
        str(label): np.asarray(matrix[position], dtype=np.float32).copy()
        for position, label in enumerate(labels)
    }


def permute_activity_identity(
    activity_embeddings: Mapping[str, np.ndarray],
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    """Randomly reassign existing vectors to labels without any fixed points."""
    permuted, _ = _permuted_activity_identity(activity_embeddings, seed=seed)
    return permuted


def shuffle_within_variants(
    sequences: Sequence[Sequence[str]],
    variant_ids: Sequence[str],
    *,
    seed: int,
) -> list[list[str]]:
    """Independently shuffle each variant using a stable variant-specific seed."""
    if len(sequences) != len(variant_ids):
        raise ValueError("Sequences and variant IDs must have equal length")
    if len(set(map(str, variant_ids))) != len(variant_ids):
        raise ValueError("Variant IDs must be unique")

    shuffled: list[list[str]] = []
    for sequence, variant_id in zip(sequences, variant_ids, strict=True):
        original = list(sequence)
        if not original:
            raise ValueError("Cannot shuffle an empty activity sequence")
        local_seed = _variant_seed(seed, str(variant_id))
        random = np.random.default_rng(local_seed)
        candidate = original.copy()
        if len(original) > 1 and len(set(original)) > 1:
            while candidate == original:
                candidate = [original[index] for index in random.permutation(len(original))]
        shuffled.append(candidate)
    return shuffled


def random_gaussian_activity_embeddings(
    activity_labels: Sequence[str],
    *,
    dimension: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Generate deterministic, isotropic, unit-normalized Gaussian vectors."""
    labels = [str(label) for label in activity_labels]
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Activity labels must be nonempty and unique")
    if dimension < 1:
        raise ValueError("Embedding dimension must be positive")

    matrix = np.random.default_rng(seed).normal(size=(len(labels), dimension))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = (matrix / norms).astype(np.float32)
    return {
        label: matrix[position].copy()
        for position, label in enumerate(labels)
    }


def build_diagnostic_embeddings(
    sequences: Sequence[Sequence[str]],
    variant_ids: Sequence[str],
    activity_embeddings: Mapping[str, np.ndarray],
    *,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Compile the three negative-control representations from frozen inputs."""
    labels, matrix = _validated_activity_matrix(activity_embeddings)
    used = {activity for sequence in sequences for activity in sequence}
    missing = sorted(used.difference(activity_embeddings))
    if missing:
        raise ValueError(f"Missing activity embeddings: {missing}")

    permuted, source_by_label = _permuted_activity_identity(
        activity_embeddings,
        seed=seed,
    )
    shuffled_sequences = shuffle_within_variants(
        sequences,
        variant_ids,
        seed=seed,
    )
    random_embeddings = random_gaussian_activity_embeddings(
        labels,
        dimension=matrix.shape[1],
        seed=seed,
    )
    controls = {
        ACTIVITY_PERMUTATION_NAME: ordered_semantic_pyramid(sequences, permuted),
        WITHIN_VARIANT_SHUFFLE_NAME: ordered_semantic_pyramid(
            shuffled_sequences,
            activity_embeddings,
        ),
        RANDOM_EMBEDDING_NAME: ordered_semantic_pyramid(
            sequences,
            random_embeddings,
        ),
    }
    metadata: dict[str, Any] = {
        ACTIVITY_PERMUTATION_NAME: {
            "seed": seed,
            "vector_source_by_activity_label": source_by_label,
            "preserves": ["sequence", "activity_equality", "frequency", "vector_set"],
        },
        WITHIN_VARIANT_SHUFFLE_NAME: {
            "seed": seed,
            "seed_derivation": "sha256(global_seed, variant_id)",
            "preserves": ["per_variant_activity_multiset", "frequency"],
        },
        RANDOM_EMBEDDING_NAME: {
            "seed": seed,
            "distribution": "independent standard Gaussian, row L2-normalized",
            "activity_embedding_dimension": matrix.shape[1],
            "preserves": ["sequence", "activity_equality", "frequency"],
        },
    }
    return controls, metadata


def run_diagnostic_controls(
    variants: pd.DataFrame,
    activity_embeddings: Mapping[str, np.ndarray],
    frozen_p7_embeddings: np.ndarray,
    *,
    fixed_k: int,
    output_dir: Path,
    seed: int = 1729,
    manifest_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run fixed-K baseline and controls and persist auditable result artifacts."""
    sequences = parse_sequences(variants)
    variant_ids = tuple(variants["variant_id"].astype(str))
    weights = variants["frequency"].to_numpy(dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Frequencies must be finite and positive")
    if not 2 <= fixed_k < len(variants):
        raise ValueError("fixed_k must be between 2 and the number of variants - 1")

    frozen = np.asarray(frozen_p7_embeddings, dtype=np.float32)
    reconstructed = ordered_semantic_pyramid(sequences, activity_embeddings)
    if frozen.shape != reconstructed.shape or not np.allclose(
        frozen,
        reconstructed,
        atol=3e-5,
        rtol=2e-5,
    ):
        raise ValueError(
            "Frozen P7 embeddings do not align with the variant and activity indices"
        )

    controls, control_metadata = build_diagnostic_embeddings(
        sequences,
        variant_ids,
        activity_embeddings,
        seed=seed,
    )
    representations = {BASELINE_NAME: frozen, **controls}
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    assignments: dict[str, np.ndarray] = {}
    for name, embeddings in representations.items():
        features = embedding_feature_set(name, variants, embeddings)
        result = weighted_kmeans(
            features.matrix,
            weights,
            n_clusters=fixed_k,
        )
        metrics = evaluate_partition(
            features.matrix,
            result.labels,
            weights,
            metric=features.metric,
        )
        assignments[name] = result.labels
        metric_rows.append(
            {
                "representation": name,
                "fixed_k": fixed_k,
                "feature_count": features.matrix.shape[1],
                "selected_seed": result.selected_seed,
                "weighted_inertia": result.weighted_inertia,
                "mean_seed_stability_ari": result.mean_seed_stability_ari,
                **metrics,
            }
        )
        pd.DataFrame(
            {
                "variant_id": variant_ids,
                "frequency": variants["frequency"].to_numpy(copy=True),
                "cluster_label": result.labels,
                "cluster_id": [
                    f"{name}_k{fixed_k}_c{int(label):02d}"
                    for label in result.labels
                ],
            }
        ).to_parquet(
            output_dir / f"assignments_{name}_k{fixed_k}.parquet",
            index=False,
        )

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(output_dir / "fixed_k_metrics.csv", index=False)
    agreement_frame = pd.DataFrame(
        [
            {
                "baseline": BASELINE_NAME,
                "control": name,
                "ari": adjusted_rand_score(assignments[BASELINE_NAME], labels),
                "nmi": normalized_mutual_info_score(
                    assignments[BASELINE_NAME],
                    labels,
                ),
            }
            for name, labels in controls_assignment_items(assignments)
        ]
    )
    agreement_frame.to_csv(
        output_dir / "partition_agreement_against_baseline.csv",
        index=False,
    )

    summary_rows = _empirical_summary(metrics_frame, agreement_frame)
    summary_rows.to_csv(output_dir / "summary.csv", index=False)
    summary = {
        "interpretation": (
            "Empirical control-minus-baseline differences are descriptive; "
            "no fixed pass/fail threshold is imposed."
        ),
        "fixed_k": fixed_k,
        "seed": seed,
        "controls": summary_rows.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "analysis": "frozen_p7_diagnostic_controls",
        "fixed_k": fixed_k,
        "seed": seed,
        "variant_count": len(variants),
        "case_count": float(weights.sum()),
        "activity_count": len(activity_embeddings),
        "frequency_usage": (
            "preserved unchanged; used as sample_weight in every clustering objective "
            "and in case-weighted evaluation metrics"
        ),
        "baseline_validation": (
            "frozen P7 rows matched a fresh ordered_semantic_pyramid compilation"
        ),
        "controls": control_metadata,
        "context": dict(manifest_context or {}),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "metrics": metrics_frame,
        "agreement": agreement_frame,
        "summary": summary_rows,
        "manifest": manifest,
    }


def controls_assignment_items(
    assignments: Mapping[str, np.ndarray],
) -> list[tuple[str, np.ndarray]]:
    return [
        (name, labels)
        for name, labels in assignments.items()
        if name != BASELINE_NAME
    ]


def _empirical_summary(
    metrics: pd.DataFrame,
    agreement: pd.DataFrame,
) -> pd.DataFrame:
    baseline = metrics.loc[metrics["representation"] == BASELINE_NAME].iloc[0]
    reported_metrics = (
        "variant_silhouette",
        "case_weighted_silhouette",
        "case_cluster_entropy",
        "weighted_inertia",
        "mean_seed_stability_ari",
    )
    rows = []
    for agreement_row in agreement.itertuples(index=False):
        control = metrics.loc[
            metrics["representation"] == agreement_row.control
        ].iloc[0]
        row: dict[str, Any] = {
            "control": agreement_row.control,
            "ari_against_baseline": agreement_row.ari,
            "nmi_against_baseline": agreement_row.nmi,
        }
        for metric in reported_metrics:
            row[f"{metric}_baseline"] = baseline[metric]
            row[f"{metric}_control"] = control[metric]
            row[f"{metric}_delta"] = control[metric] - baseline[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def _validated_activity_matrix(
    activity_embeddings: Mapping[str, np.ndarray],
) -> tuple[list[str], np.ndarray]:
    if not activity_embeddings:
        raise ValueError("Activity embeddings must not be empty")
    labels = [str(label) for label in activity_embeddings]
    if len(set(labels)) != len(labels):
        raise ValueError("Activity labels must be unique when converted to strings")
    vectors = [np.asarray(activity_embeddings[label]) for label in activity_embeddings]
    shapes = {vector.shape for vector in vectors}
    if len(shapes) != 1 or len(next(iter(shapes))) != 1:
        raise ValueError("Activity embeddings must have one consistent dimension")
    matrix = np.vstack(vectors).astype(np.float32)
    if matrix.shape[1] < 1 or not np.isfinite(matrix).all():
        raise ValueError("Activity embeddings must be finite and nonempty")
    return labels, matrix


def _permuted_activity_identity(
    activity_embeddings: Mapping[str, np.ndarray],
    *,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    labels, matrix = _validated_activity_matrix(activity_embeddings)
    if len(labels) < 2:
        raise ValueError("At least two activities are required for identity permutation")

    random = np.random.default_rng(seed)
    positions = np.arange(len(labels))
    while True:
        permutation = random.permutation(len(labels))
        if np.all(permutation != positions):
            break
    permuted = {
        label: matrix[source].copy()
        for label, source in zip(labels, permutation, strict=True)
    }
    source_by_label = {
        label: labels[source]
        for label, source in zip(labels, permutation, strict=True)
    }
    return permuted, source_by_label


def _variant_seed(seed: int, variant_id: str) -> int:
    digest = hashlib.sha256(
        json.dumps(
            [int(seed), variant_id],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def sequence_multisets(
    sequences: Sequence[Sequence[str]],
) -> list[Counter[str]]:
    """Return sequence multisets, primarily for auditing shuffle preservation."""
    return [Counter(sequence) for sequence in sequences]
