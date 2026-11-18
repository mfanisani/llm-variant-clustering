"""Regression tests for the singleton convention of the weighted silhouette.

A case that is alone in its cluster has silhouette 0 (Rousseeuw 1987;
scikit-learn). A one-variant cluster whose variant has frequency > 1 is not a
singleton at case level, because its duplicated cases lie at distance 0.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import pairwise_distances, silhouette_samples, silhouette_score

from llm_variant_clustering.evaluation import frequency_weighted_silhouette


def _expand(distances: np.ndarray, labels: np.ndarray, weights: np.ndarray):
    rows = np.repeat(np.arange(labels.size), weights.astype(int))
    return distances[np.ix_(rows, rows)], labels[rows]


def test_three_case_reviewer_example_matches_sklearn() -> None:
    distances = np.array(
        [
            [0.0, 0.2, 0.9],
            [0.2, 0.0, 0.8],
            [0.9, 0.8, 0.0],
        ]
    )
    labels = np.array([0, 0, 1])
    weights = np.ones(3)

    expected = silhouette_score(distances, labels, metric="precomputed")
    observed = frequency_weighted_silhouette(distances, labels, weights)

    assert observed == pytest.approx(expected, abs=1e-12)
    # The singleton contributes 0, not 1.
    assert observed == pytest.approx((1 - 0.2 / 0.9 + 1 - 0.2 / 0.8) / 3)


def test_integer_weights_match_sklearn_on_expanded_rows() -> None:
    rng = np.random.default_rng(3)
    points = rng.normal(size=(12, 4))
    distances = pairwise_distances(points, metric="cosine")
    labels = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4])
    weights = np.array([3, 1, 2, 1, 1, 4, 1, 2, 1, 1, 1, 1], dtype=float)
    # Cluster 4 is a case-level singleton (one variant, frequency 1).

    expanded_distances, expanded_labels = _expand(distances, labels, weights)
    expected = silhouette_score(
        expanded_distances, expanded_labels, metric="precomputed"
    )
    observed = frequency_weighted_silhouette(distances, labels, weights)

    assert observed == pytest.approx(expected, abs=1e-12)


def test_one_variant_cluster_with_weight_35_behaves_as_expanded_rows() -> None:
    rng = np.random.default_rng(11)
    points = rng.normal(size=(6, 3))
    distances = pairwise_distances(points, metric="euclidean")
    labels = np.array([0, 0, 0, 1, 1, 2])
    weights = np.array([2, 1, 3, 1, 4, 35], dtype=float)

    expanded_distances, expanded_labels = _expand(distances, labels, weights)
    expected = silhouette_score(
        expanded_distances, expanded_labels, metric="precomputed"
    )
    observed = frequency_weighted_silhouette(distances, labels, weights)

    assert observed == pytest.approx(expected, abs=1e-12)
    # The 35 duplicated cases are at distance 0 from each other, so each has
    # silhouette 1 rather than the singleton value 0.
    per_case = silhouette_samples(
        expanded_distances, expanded_labels, metric="precomputed"
    )
    assert np.allclose(per_case[expanded_labels == 2], 1.0)
