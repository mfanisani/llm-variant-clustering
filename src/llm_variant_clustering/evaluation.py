from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
    silhouette_score,
)


def evaluate_partition(
    matrix: csr_matrix,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    metric: str,
    distances: np.ndarray | None = None,
) -> dict[str, Any]:
    labels = np.asarray(labels)
    weights = np.asarray(weights, dtype=np.float64)
    if matrix.shape[0] != len(labels) or len(labels) != len(weights):
        raise ValueError("Matrix, labels, and weights must have equal row counts")
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("Partition must contain at least two clusters")

    if distances is None:
        distances = pairwise_distances(matrix, metric=metric)
    elif distances.shape != (matrix.shape[0], matrix.shape[0]):
        raise ValueError("Precomputed distances do not match feature rows")
    weighted_silhouette = frequency_weighted_silhouette(
        distances, labels, weights
    )
    case_sizes = np.array(
        [weights[labels == cluster].sum() for cluster in unique_labels]
    )
    variant_sizes = np.array(
        [(labels == cluster).sum() for cluster in unique_labels]
    )
    proportions = case_sizes / case_sizes.sum()
    normalized_entropy = float(
        -np.sum(proportions * np.log(proportions))
        / math.log(len(unique_labels))
    )

    metrics: dict[str, Any] = {
        "clusters": int(len(unique_labels)),
        "variant_silhouette": float(
            silhouette_score(matrix, labels, metric=metric)
        ),
        "case_weighted_silhouette": weighted_silhouette,
        "min_cluster_cases": int(case_sizes.min()),
        "max_cluster_cases": int(case_sizes.max()),
        "min_cluster_variants": int(variant_sizes.min()),
        "max_cluster_variants": int(variant_sizes.max()),
        "case_cluster_entropy": normalized_entropy,
    }
    if matrix.shape[0] * matrix.shape[1] <= 10_000_000:
        dense = matrix.toarray()
        metrics["variant_davies_bouldin"] = float(
            davies_bouldin_score(dense, labels)
        )
        metrics["variant_calinski_harabasz"] = float(
            calinski_harabasz_score(dense, labels)
        )
    else:
        metrics["variant_davies_bouldin"] = None
        metrics["variant_calinski_harabasz"] = None
    return metrics


def evaluate_distance_partition(
    distances: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(labels)
    weights = np.asarray(weights, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    unique_labels = np.unique(labels)
    if distances.shape != (len(labels), len(labels)):
        raise ValueError("Distance matrix shape does not match labels")

    case_sizes = np.array(
        [weights[labels == cluster].sum() for cluster in unique_labels]
    )
    variant_sizes = np.array(
        [(labels == cluster).sum() for cluster in unique_labels]
    )
    proportions = case_sizes / case_sizes.sum()
    return {
        "clusters": int(len(unique_labels)),
        "variant_silhouette": float(
            silhouette_score(distances, labels, metric="precomputed")
        ),
        "case_weighted_silhouette": frequency_weighted_silhouette(
            distances, labels, weights
        ),
        "min_cluster_cases": int(case_sizes.min()),
        "max_cluster_cases": int(case_sizes.max()),
        "min_cluster_variants": int(variant_sizes.min()),
        "max_cluster_variants": int(variant_sizes.max()),
        "case_cluster_entropy": float(
            -np.sum(proportions * np.log(proportions))
            / math.log(len(unique_labels))
        ),
        "variant_davies_bouldin": None,
        "variant_calinski_harabasz": None,
    }



def frequency_weighted_silhouette(
    distances: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> float:
    distances = np.asarray(distances, dtype=np.float64)
    labels = np.asarray(labels)
    weights = np.asarray(weights, dtype=np.float64)
    size = len(labels)
    if distances.shape != (size, size):
        raise ValueError("distances must be a square sample-by-sample matrix")
    if np.any(weights <= 0):
        raise ValueError("weights must be positive")

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("At least two clusters are required")

    cluster_distances = np.empty((size, len(unique_labels)), dtype=np.float64)
    cluster_weights = np.empty(len(unique_labels), dtype=np.float64)
    encoded_labels = np.empty(size, dtype=np.int64)
    for position, cluster in enumerate(unique_labels):
        mask = labels == cluster
        cluster_weights[position] = weights[mask].sum()
        cluster_distances[:, position] = (
            distances[:, mask] @ weights[mask]
        ) / cluster_weights[position]
        encoded_labels[mask] = position

    row_indices = np.arange(size)
    own_weights = cluster_weights[encoded_labels]
    within = (
        cluster_distances[row_indices, encoded_labels]
        * own_weights
        / np.maximum(own_weights - 1.0, 1.0)
    )
    within[own_weights <= 1.0] = 0.0
    cluster_distances[row_indices, encoded_labels] = np.inf
    nearest = cluster_distances.min(axis=1)
    denominator = np.maximum(within, nearest)
    silhouettes = np.divide(
        nearest - within,
        denominator,
        out=np.zeros(size, dtype=np.float64),
        where=denominator > 0,
    )
    # A point alone in its cluster at case level has silhouette 0 (Rousseeuw
    # 1987; scikit-learn). A one-variant cluster with frequency > 1 is not a
    # singleton because its duplicated cases lie at distance 0.
    silhouettes[own_weights <= 1.0] = 0.0
    return float(np.average(silhouettes, weights=weights))


def weighted_medoid_indices(
    matrix: csr_matrix,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    metric: str,
) -> dict[int, int]:
    labels = np.asarray(labels)
    weights = np.asarray(weights, dtype=np.float64)
    medoids: dict[int, int] = {}
    for cluster in np.unique(labels):
        member_indices = np.flatnonzero(labels == cluster)
        member_distances = pairwise_distances(
            matrix[member_indices], metric=metric
        )
        objectives = member_distances @ weights[member_indices]
        medoids[int(cluster)] = int(member_indices[int(np.argmin(objectives))])
    return medoids
