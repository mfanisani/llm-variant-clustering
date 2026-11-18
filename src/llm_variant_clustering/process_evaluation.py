from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import pandas as pd
import pm4py


def evaluate_discovered_model(
    events: pd.DataFrame,
    *,
    noise_threshold: float = 0.2,
    case_key: str = "case:concept:name",
    activity_key: str = "concept:name",
    timestamp_key: str = "time:timestamp",
) -> dict[str, Any]:
    if events.empty:
        raise ValueError("Cannot discover a process model from an empty log")
    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(
        events,
        noise_threshold=noise_threshold,
        activity_key=activity_key,
        timestamp_key=timestamp_key,
        case_id_key=case_key,
    )
    return {
        "cases": int(events[case_key].nunique()),
        "events": int(len(events)),
        **_evaluate_existing_model(
            events,
            net,
            initial_marking,
            final_marking,
            case_key=case_key,
            activity_key=activity_key,
            timestamp_key=timestamp_key,
        ),
    }


def evaluate_holdout_model(
    train_events: pd.DataFrame,
    test_events: pd.DataFrame,
    *,
    noise_threshold: float = 0.2,
    case_key: str = "case:concept:name",
    activity_key: str = "concept:name",
    timestamp_key: str = "time:timestamp",
) -> dict[str, Any]:
    if train_events.empty:
        raise ValueError("Cannot discover a process model from an empty training log")
    if test_events.empty:
        raise ValueError("Cannot evaluate a process model on an empty test log")
    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(
        train_events,
        noise_threshold=noise_threshold,
        activity_key=activity_key,
        timestamp_key=timestamp_key,
        case_id_key=case_key,
    )
    return {
        "train_cases": int(train_events[case_key].nunique()),
        "train_events": int(len(train_events)),
        "test_cases": int(test_events[case_key].nunique()),
        "test_events": int(len(test_events)),
        **_evaluate_existing_model(
            test_events,
            net,
            initial_marking,
            final_marking,
            case_key=case_key,
            activity_key=activity_key,
            timestamp_key=timestamp_key,
        ),
    }


def _evaluate_existing_model(
    events: pd.DataFrame,
    net: Any,
    initial_marking: Any,
    final_marking: Any,
    *,
    case_key: str,
    activity_key: str,
    timestamp_key: str,
) -> dict[str, Any]:
    fitness_result = pm4py.fitness_token_based_replay(
        events,
        net,
        initial_marking,
        final_marking,
        activity_key=activity_key,
        timestamp_key=timestamp_key,
        case_id_key=case_key,
    )
    precision = float(
        pm4py.precision_token_based_replay(
            events,
            net,
            initial_marking,
            final_marking,
            activity_key=activity_key,
            timestamp_key=timestamp_key,
            case_id_key=case_key,
        )
    )
    fitness = float(fitness_result["log_fitness"])
    harmonic = (
        2 * fitness * precision / (fitness + precision)
        if fitness + precision > 0
        else 0.0
    )
    return {
        "fitness": fitness,
        "precision": precision,
        "fitness_precision_f1": harmonic,
        "simplicity_arc_degree": float(
            pm4py.simplicity_petri_net(
                net,
                initial_marking,
                final_marking,
                variant="arc_degree",
            )
        ),
        "places": len(net.places),
        "transitions": len(net.transitions),
        "silent_transitions": sum(
            transition.label is None for transition in net.transitions
        ),
        "arcs": len(net.arcs),
    }


def evaluate_process_partition(
    events: pd.DataFrame,
    case_variant_map: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    method: str,
    noise_threshold: float = 0.2,
    noise_threshold_by_cluster: Mapping[str, float] | None = None,
    case_key: str = "case:concept:name",
) -> pd.DataFrame:
    required_mapping = {"case_id", "variant_id"}
    required_assignments = {"variant_id", "cluster_id"}
    if missing := required_mapping.difference(case_variant_map.columns):
        raise ValueError(f"Case map is missing columns: {sorted(missing)}")
    if missing := required_assignments.difference(assignments.columns):
        raise ValueError(f"Assignments are missing columns: {sorted(missing)}")
    if assignments["variant_id"].duplicated().any():
        raise ValueError("Assignments must contain one row per variant")

    case_clusters = case_variant_map.merge(
        assignments[["variant_id", "cluster_id"]],
        on="variant_id",
        how="left",
        validate="many_to_one",
    )
    if case_clusters["cluster_id"].isna().any():
        raise ValueError("Some cases refer to variants without assignments")
    if case_clusters["case_id"].duplicated().any():
        raise ValueError("Case-to-variant map must contain one row per case")

    cluster_by_case = dict(
        zip(
            case_clusters["case_id"].astype(str),
            case_clusters["cluster_id"],
            strict=True,
        )
    )
    clustered_events = events.copy()
    clustered_events["_cluster_id"] = (
        clustered_events[case_key].astype(str).map(cluster_by_case)
    )
    if clustered_events["_cluster_id"].isna().any():
        raise ValueError("Some events refer to cases absent from the case map")

    rows = []
    for cluster_id, cluster_events in clustered_events.groupby(
        "_cluster_id", sort=True
    ):
        cluster_threshold = (
            noise_threshold_by_cluster[str(cluster_id)]
            if noise_threshold_by_cluster is not None
            else noise_threshold
        )
        metrics = evaluate_discovered_model(
            cluster_events.drop(columns="_cluster_id"),
            noise_threshold=cluster_threshold,
            case_key=case_key,
        )
        rows.append(
            {
                "method": method,
                "scope": "cluster",
                "cluster_id": cluster_id,
                "noise_threshold": cluster_threshold,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def select_variant_holdout_split(
    case_variant_map: pd.DataFrame,
    assignments_by_method: Mapping[str, pd.DataFrame],
    *,
    test_fraction: float = 0.2,
    max_seed: int = 10_000,
    case_fraction_tolerance: float = 0.05,
) -> tuple[pd.DataFrame, int]:
    if not 0 < test_fraction < 1:
        raise ValueError("Test fraction must be strictly between zero and one")
    if max_seed < 1:
        raise ValueError("Maximum seed count must be positive")
    if not 0 <= case_fraction_tolerance < 1:
        raise ValueError("Case-fraction tolerance must be in [0, 1)")
    required_mapping = {"case_id", "variant_id"}
    if missing := required_mapping.difference(case_variant_map.columns):
        raise ValueError(f"Case map is missing columns: {sorted(missing)}")
    if case_variant_map["case_id"].duplicated().any():
        raise ValueError("Case-to-variant map must contain one row per case")
    if not assignments_by_method:
        raise ValueError("At least one assignment table is required")

    normalized_map = case_variant_map[["case_id", "variant_id"]].copy()
    normalized_map["variant_id"] = normalized_map["variant_id"].astype(str)
    variant_cases = normalized_map.groupby("variant_id", sort=True).size()
    expected_variants = frozenset(variant_cases.index)
    normalized_assignments: dict[str, pd.DataFrame] = {}
    for method, assignments in assignments_by_method.items():
        required_assignments = {"variant_id", "cluster_id"}
        if missing := required_assignments.difference(assignments.columns):
            raise ValueError(
                f"Assignments for {method!r} are missing columns: {sorted(missing)}"
            )
        frame = assignments[["variant_id", "cluster_id"]].copy()
        frame["variant_id"] = frame["variant_id"].astype(str)
        if frame["variant_id"].duplicated().any():
            raise ValueError(
                f"Assignments for {method!r} must contain one row per variant"
            )
        if frozenset(frame["variant_id"]) != expected_variants:
            raise ValueError(
                f"Assignments for {method!r} do not match the case-map variants"
            )
        singleton_clusters = (
            frame.groupby("cluster_id", sort=True)["variant_id"]
            .size()
            .loc[lambda sizes: sizes < 2]
        )
        if not singleton_clusters.empty:
            cluster_ids = ", ".join(map(str, singleton_clusters.index))
            raise ValueError(
                "Holdout split is structurally infeasible because every cluster "
                "needs at least one training and one test variant; "
                f"{method!r} has fewer than two variants in cluster(s): "
                f"{cluster_ids}"
            )
        normalized_assignments[method] = frame

    minimum_case_fraction = max(0.0, test_fraction - case_fraction_tolerance)
    maximum_case_fraction = min(1.0, test_fraction + case_fraction_tolerance)
    total_cases = int(variant_cases.sum())
    for seed in range(max_seed):
        is_test = {
            variant_id: _holdout_score(variant_id, seed) < test_fraction
            for variant_id in variant_cases.index
        }
        test_cases = sum(
            int(variant_cases[variant_id])
            for variant_id, selected in is_test.items()
            if selected
        )
        observed_fraction = test_cases / total_cases
        if not minimum_case_fraction <= observed_fraction <= maximum_case_fraction:
            continue

        split = pd.DataFrame(
            {
                "variant_id": list(variant_cases.index),
                "split": [
                    "test" if is_test[variant_id] else "train"
                    for variant_id in variant_cases.index
                ],
                "cases": [
                    int(variant_cases[variant_id])
                    for variant_id in variant_cases.index
                ],
            }
        )
        if all(
            _clusters_cover_both_splits(assignments, split)
            for assignments in normalized_assignments.values()
        ):
            return split, seed

    raise ValueError(
        "No deterministic variant split satisfies the requested case fraction "
        "and train/test coverage for every method cluster"
    )


def evaluate_process_partition_holdout(
    events: pd.DataFrame,
    case_variant_map: pd.DataFrame,
    assignments: pd.DataFrame,
    variant_split: pd.DataFrame,
    *,
    method: str,
    noise_threshold: float = 0.2,
    case_key: str = "case:concept:name",
) -> pd.DataFrame:
    required_mapping = {"case_id", "variant_id"}
    required_assignments = {"variant_id", "cluster_id"}
    required_split = {"variant_id", "split"}
    if missing := required_mapping.difference(case_variant_map.columns):
        raise ValueError(f"Case map is missing columns: {sorted(missing)}")
    if missing := required_assignments.difference(assignments.columns):
        raise ValueError(f"Assignments are missing columns: {sorted(missing)}")
    if missing := required_split.difference(variant_split.columns):
        raise ValueError(f"Variant split is missing columns: {sorted(missing)}")

    normalized_map = case_variant_map[["case_id", "variant_id"]].copy()
    normalized_map["case_id"] = normalized_map["case_id"].astype(str)
    normalized_map["variant_id"] = normalized_map["variant_id"].astype(str)
    normalized_assignments = assignments[["variant_id", "cluster_id"]].copy()
    normalized_assignments["variant_id"] = normalized_assignments[
        "variant_id"
    ].astype(str)
    normalized_split = variant_split[["variant_id", "split"]].copy()
    normalized_split["variant_id"] = normalized_split["variant_id"].astype(str)
    if normalized_assignments["variant_id"].duplicated().any():
        raise ValueError("Assignments must contain one row per variant")
    if normalized_split["variant_id"].duplicated().any():
        raise ValueError("Variant split must contain one row per variant")
    if not normalized_split["split"].isin({"train", "test"}).all():
        raise ValueError("Variant split values must be 'train' or 'test'")

    case_clusters = (
        normalized_map.merge(
            normalized_assignments,
            on="variant_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            normalized_split,
            on="variant_id",
            how="left",
            validate="many_to_one",
        )
    )
    if case_clusters[["cluster_id", "split"]].isna().any().any():
        raise ValueError("Some cases lack a cluster assignment or split")
    if case_clusters["case_id"].duplicated().any():
        raise ValueError("Case-to-variant map must contain one row per case")

    cluster_by_case = dict(
        zip(
            case_clusters["case_id"],
            case_clusters["cluster_id"],
            strict=True,
        )
    )
    split_by_case = dict(
        zip(case_clusters["case_id"], case_clusters["split"], strict=True)
    )
    clustered_events = events.copy()
    event_cases = clustered_events[case_key].astype(str)
    clustered_events["_cluster_id"] = event_cases.map(cluster_by_case)
    clustered_events["_split"] = event_cases.map(split_by_case)
    if clustered_events[["_cluster_id", "_split"]].isna().any().any():
        raise ValueError("Some events refer to cases absent from the case map")

    variant_counts = (
        case_clusters[["variant_id", "cluster_id", "split"]]
        .drop_duplicates()
        .groupby(["cluster_id", "split"], sort=True)
        .size()
    )
    rows = []
    for cluster_id, cluster_events in clustered_events.groupby(
        "_cluster_id", sort=True
    ):
        train_events = cluster_events[cluster_events["_split"].eq("train")].drop(
            columns=["_cluster_id", "_split"]
        )
        test_events = cluster_events[cluster_events["_split"].eq("test")].drop(
            columns=["_cluster_id", "_split"]
        )
        metrics = evaluate_holdout_model(
            train_events,
            test_events,
            noise_threshold=noise_threshold,
            case_key=case_key,
        )
        rows.append(
            {
                "method": method,
                "scope": "cluster",
                "cluster_id": cluster_id,
                "noise_threshold": noise_threshold,
                "train_variants": int(variant_counts.loc[(cluster_id, "train")]),
                "test_variants": int(variant_counts.loc[(cluster_id, "test")]),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _holdout_score(variant_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{variant_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _clusters_cover_both_splits(
    assignments: pd.DataFrame,
    split: pd.DataFrame,
) -> bool:
    coverage = assignments.merge(
        split[["variant_id", "split"]],
        on="variant_id",
        how="left",
        validate="one_to_one",
    )
    observed = coverage.groupby("cluster_id", sort=False)["split"].agg(set)
    return observed.map(lambda values: values == {"train", "test"}).all()
