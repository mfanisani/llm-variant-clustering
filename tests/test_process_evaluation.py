from __future__ import annotations

import pandas as pd
import pytest

from llm_variant_clustering.process_evaluation import (
    evaluate_process_partition_holdout,
    evaluate_process_partition,
    select_variant_holdout_split,
)


def test_process_partition_evaluates_each_cluster() -> None:
    events = pd.DataFrame(
        {
            "case:concept:name": ["c1", "c1", "c2", "c2", "c3", "c3"],
            "concept:name": ["A", "B", "A", "B", "A", "C"],
            "time:timestamp": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T01:00:00Z",
                    "2024-01-02T00:00:00Z",
                    "2024-01-02T01:00:00Z",
                    "2024-01-03T00:00:00Z",
                    "2024-01-03T01:00:00Z",
                ]
            ),
        }
    )
    case_map = pd.DataFrame(
        {
            "case_id": ["c1", "c2", "c3"],
            "variant_id": ["v_ab", "v_ab", "v_ac"],
        }
    )
    assignments = pd.DataFrame(
        {
            "variant_id": ["v_ab", "v_ac"],
            "cluster_id": ["cluster_ab", "cluster_ac"],
        }
    )

    metrics = evaluate_process_partition(
        events,
        case_map,
        assignments,
        method="synthetic",
        noise_threshold=0.0,
        noise_threshold_by_cluster={
            "cluster_ab": 0.1,
            "cluster_ac": 0.3,
        },
    )

    assert set(metrics["cluster_id"]) == {"cluster_ab", "cluster_ac"}
    assert metrics["cases"].sum() == 3
    assert metrics["fitness"].between(0, 1).all()
    assert metrics["precision"].between(0, 1).all()
    assert dict(zip(metrics["cluster_id"], metrics["noise_threshold"])) == {
        "cluster_ab": 0.1,
        "cluster_ac": 0.3,
    }


def test_variant_holdout_split_is_deterministic_and_covers_clusters() -> None:
    case_map = pd.DataFrame(
        {
            "case_id": [f"c{index}" for index in range(8)],
            "variant_id": [f"v{index}" for index in range(8)],
        }
    )
    assignments = {
        "method_a": pd.DataFrame(
            {
                "variant_id": [f"v{index}" for index in range(8)],
                "cluster_id": ["a0"] * 4 + ["a1"] * 4,
            }
        ),
        "method_b": pd.DataFrame(
            {
                "variant_id": [f"v{index}" for index in range(8)],
                "cluster_id": ["b0", "b1"] * 4,
            }
        ),
    }

    first, first_seed = select_variant_holdout_split(
        case_map,
        assignments,
        test_fraction=0.5,
        case_fraction_tolerance=0.25,
    )
    second, second_seed = select_variant_holdout_split(
        case_map,
        assignments,
        test_fraction=0.5,
        case_fraction_tolerance=0.25,
    )

    pd.testing.assert_frame_equal(first, second)
    assert first_seed == second_seed
    assert set(first["split"]) == {"train", "test"}
    for method_assignments in assignments.values():
        coverage = method_assignments.merge(first, on="variant_id")
        assert (
            coverage.groupby("cluster_id")["split"]
            .agg(set)
            .eq({"train", "test"})
            .all()
        )


def test_process_partition_holdout_trains_and_tests_each_cluster() -> None:
    events = pd.DataFrame(
        {
            "case:concept:name": [
                "c1",
                "c1",
                "c2",
                "c2",
                "c3",
                "c3",
                "c4",
                "c4",
            ],
            "concept:name": ["A", "B", "A", "C", "D", "E", "D", "F"],
            "time:timestamp": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T01:00:00Z",
                    "2024-01-02T00:00:00Z",
                    "2024-01-02T01:00:00Z",
                    "2024-01-03T00:00:00Z",
                    "2024-01-03T01:00:00Z",
                    "2024-01-04T00:00:00Z",
                    "2024-01-04T01:00:00Z",
                ]
            ),
        }
    )
    case_map = pd.DataFrame(
        {
            "case_id": ["c1", "c2", "c3", "c4"],
            "variant_id": ["v_ab", "v_ac", "v_de", "v_df"],
        }
    )
    assignments = pd.DataFrame(
        {
            "variant_id": ["v_ab", "v_ac", "v_de", "v_df"],
            "cluster_id": ["cluster_a", "cluster_a", "cluster_d", "cluster_d"],
        }
    )
    variant_split = pd.DataFrame(
        {
            "variant_id": ["v_ab", "v_ac", "v_de", "v_df"],
            "split": ["train", "test", "train", "test"],
        }
    )

    metrics = evaluate_process_partition_holdout(
        events,
        case_map,
        assignments,
        variant_split,
        method="synthetic",
        noise_threshold=0.0,
    )

    assert set(metrics["cluster_id"]) == {"cluster_a", "cluster_d"}
    assert metrics["train_cases"].sum() == 2
    assert metrics["test_cases"].sum() == 2
    assert metrics["train_variants"].eq(1).all()
    assert metrics["test_variants"].eq(1).all()
    assert metrics["fitness"].between(0, 1).all()
    assert metrics["precision"].between(0, 1).all()


def test_variant_holdout_rejects_singleton_cluster() -> None:
    case_map = pd.DataFrame(
        {
            "case_id": ["c1", "c2", "c3"],
            "variant_id": ["v1", "v2", "v3"],
        }
    )
    assignments = {
        "method": pd.DataFrame(
            {
                "variant_id": ["v1", "v2", "v3"],
                "cluster_id": ["large", "large", "singleton"],
            }
        )
    }

    with pytest.raises(ValueError, match="structurally infeasible.*singleton"):
        select_variant_holdout_split(case_map, assignments)
