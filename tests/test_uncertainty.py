import numpy as np
import pytest
from sklearn.metrics import pairwise_distances

from llm_variant_clustering.uncertainty import (
    bootstrap_cluster_metric,
    bootstrap_seed_agreement,
    bootstrap_silhouette,
    cluster_scope_mask,
    paired_silhouette_difference,
)


def _separated_case(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    first = generator.normal(loc=0.0, scale=0.05, size=(30, 4))
    second = generator.normal(loc=5.0, scale=0.05, size=(30, 4))
    points = np.vstack([first, second])
    labels = np.array([0] * 30 + [1] * 30)
    weights = np.full(60, 3.0)
    return pairwise_distances(points), labels, weights


def test_bootstrap_silhouette_brackets_point_estimate() -> None:
    distances, labels, weights = _separated_case()

    interval = bootstrap_silhouette(distances, labels, weights, resamples=60)

    assert interval.lower <= interval.point_estimate <= interval.upper
    assert interval.point_estimate > 0.9
    assert interval.confidence == pytest.approx(0.95)


def test_bootstrap_silhouette_is_deterministic() -> None:
    distances, labels, weights = _separated_case()

    first = bootstrap_silhouette(distances, labels, weights, resamples=40, seed=7)
    second = bootstrap_silhouette(distances, labels, weights, resamples=40, seed=7)

    assert first == second


def test_paired_difference_detects_a_worse_partition() -> None:
    distances, labels, weights = _separated_case()
    shuffled = np.array([index % 2 for index in range(labels.size)])

    difference = paired_silhouette_difference(
        distances,
        labels,
        distances,
        shuffled,
        weights,
        resamples=60,
    )

    assert difference.interval.point_estimate > 0
    assert difference.excludes_zero
    assert difference.proportion_favouring_first == pytest.approx(1.0)


def test_paired_difference_of_identical_partitions_includes_zero() -> None:
    distances, labels, weights = _separated_case()

    difference = paired_silhouette_difference(
        distances,
        labels,
        distances,
        labels,
        weights,
        resamples=60,
    )

    assert difference.interval.point_estimate == pytest.approx(0.0, abs=1e-12)
    assert not difference.excludes_zero


def test_bootstrap_cluster_metric_respects_weights() -> None:
    values = np.array([0.2, 0.8, 0.5, 0.6])
    weights = np.array([1.0, 9.0, 1.0, 1.0])

    interval = bootstrap_cluster_metric(values, weights, resamples=200)

    assert interval.point_estimate == pytest.approx(
        np.average(values, weights=weights)
    )
    assert interval.lower <= interval.point_estimate <= interval.upper


def test_cluster_scope_mask_excludes_summary_rows() -> None:
    scopes = ["cluster", "cluster", "case_weighted_aggregate", "whole_log", "cluster"]

    mask = cluster_scope_mask(scopes)

    assert mask.tolist() == [True, True, False, False, True]


def test_aggregate_rows_would_bias_the_cluster_bootstrap() -> None:
    values = np.array([0.2, 0.9, 0.5])
    cases = np.array([10.0, 30.0, 60.0])
    aggregate_value = float(np.average(values, weights=cases))
    scopes = ["cluster", "cluster", "cluster", "case_weighted_aggregate"]
    all_values = np.append(values, aggregate_value)
    all_cases = np.append(cases, cases.sum())

    mask = cluster_scope_mask(scopes)
    corrected = bootstrap_cluster_metric(
        all_values[mask], all_cases[mask], resamples=500
    )
    contaminated = bootstrap_cluster_metric(all_values, all_cases, resamples=500)

    assert corrected.point_estimate == pytest.approx(aggregate_value)
    assert contaminated.upper - contaminated.lower < corrected.upper - corrected.lower


def test_adjusted_interval_reuses_samples_and_widens() -> None:
    distances, labels, weights = _separated_case()
    shuffled = np.array([index % 2 for index in range(labels.size)])
    difference = paired_silhouette_difference(
        distances, labels, distances, shuffled, weights, resamples=200
    )

    nominal = difference.adjusted_interval(0.95)
    adjusted = difference.adjusted_interval(1 - 0.05 / 6)

    assert nominal == difference.interval
    assert adjusted.lower <= nominal.lower
    assert adjusted.upper >= nominal.upper
    assert adjusted.point_estimate == difference.interval.point_estimate


def test_bootstrap_rejects_invalid_configuration() -> None:
    distances, labels, weights = _separated_case()

    with pytest.raises(ValueError, match="confidence"):
        bootstrap_silhouette(distances, labels, weights, confidence=1.0)
    with pytest.raises(ValueError, match="resamples"):
        bootstrap_silhouette(distances, labels, weights, resamples=1)
    with pytest.raises(ValueError, match="same variants"):
        paired_silhouette_difference(
            distances,
            labels,
            distances,
            labels[:-1],
            weights,
        )


def _agreement_matrix(values: np.ndarray) -> np.ndarray:
    size = values.size
    matrix = np.ones((size, size))
    for left in range(size):
        for right in range(left + 1, size):
            matrix[left, right] = matrix[right, left] = (values[left] + values[right]) / 2
    return matrix


def test_seed_agreement_point_estimate_is_mean_of_pairs() -> None:
    matrix = _agreement_matrix(np.linspace(0.5, 0.9, 6))
    interval = bootstrap_seed_agreement(matrix, resamples=200)
    pairs = matrix[np.triu_indices(6, k=1)]
    assert interval.point_estimate == pytest.approx(pairs.mean())
    assert interval.lower <= interval.point_estimate <= interval.upper


def test_seed_agreement_ignores_self_pairs_from_duplicate_draws() -> None:
    matrix = np.full((5, 5), 0.4)
    np.fill_diagonal(matrix, 1.0)
    interval = bootstrap_seed_agreement(matrix, resamples=300)
    assert interval.lower == pytest.approx(0.4)
    assert interval.upper == pytest.approx(0.4)


def test_seed_level_interval_is_wider_than_pair_level() -> None:
    generator = np.random.default_rng(3)
    matrix = _agreement_matrix(generator.uniform(0.3, 1.0, 12))
    seed_level = bootstrap_seed_agreement(matrix, resamples=2000)
    pair_level = bootstrap_cluster_metric(
        matrix[np.triu_indices(12, k=1)], resamples=2000
    )
    assert (seed_level.upper - seed_level.lower) > (pair_level.upper - pair_level.lower)


def test_seed_agreement_paired_difference() -> None:
    better = _agreement_matrix(np.full(8, 0.9))
    worse = _agreement_matrix(np.full(8, 0.6))
    result = bootstrap_seed_agreement(better, second=worse, resamples=200)
    assert result.interval.point_estimate == pytest.approx(0.3)
    assert result.excludes_zero
