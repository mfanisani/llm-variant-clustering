from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


@dataclass(frozen=True)
class ClusteringResult:
    labels: np.ndarray
    selected_seed: int
    weighted_inertia: float
    mean_seed_stability_ari: float
    seed_labels: tuple[np.ndarray, ...]


def weighted_kmeans(
    matrix: csr_matrix,
    weights: np.ndarray,
    *,
    n_clusters: int,
    seeds: tuple[int, ...] = (17, 29, 43, 59, 71),
    n_init: int = 10,
) -> ClusteringResult:
    if matrix.shape[0] != len(weights):
        raise ValueError("Feature rows and weights must have equal length")
    if np.any(weights <= 0):
        raise ValueError("All sample weights must be positive")
    if not 2 <= n_clusters < matrix.shape[0]:
        raise ValueError("n_clusters must be between 2 and n_samples - 1")
    if not seeds:
        raise ValueError("At least one random seed is required")

    runs: list[tuple[int, float, np.ndarray]] = []
    for seed in seeds:
        model = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init=n_init,
            algorithm="lloyd",
        )
        labels = model.fit_predict(matrix, sample_weight=weights)
        runs.append((seed, float(model.inertia_), labels))

    selected_seed, inertia, labels = min(runs, key=lambda run: run[1])
    seed_labels = tuple(run[2] for run in runs)
    stability = mean_pairwise_ari(seed_labels)
    return ClusteringResult(
        labels=labels,
        selected_seed=selected_seed,
        weighted_inertia=inertia,
        mean_seed_stability_ari=stability,
        seed_labels=seed_labels,
    )


def mean_pairwise_ari(labelings: tuple[np.ndarray, ...]) -> float:
    if len(labelings) < 2:
        return 1.0
    scores = [
        adjusted_rand_score(labelings[left], labelings[right])
        for left in range(len(labelings))
        for right in range(left + 1, len(labelings))
    ]
    return float(np.mean(scores))


def weighted_kmedoids(
    distances: np.ndarray,
    weights: np.ndarray,
    *,
    n_clusters: int,
    seeds: tuple[int, ...] = (17, 29, 43, 59, 71),
    max_iterations: int = 100,
) -> tuple[ClusteringResult, np.ndarray]:
    distances = np.asarray(distances, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    sample_count = len(weights)
    if distances.shape != (sample_count, sample_count):
        raise ValueError("distances must be square and match weights")
    if not np.allclose(distances, distances.T, atol=1e-7):
        raise ValueError("distances must be symmetric")
    if np.any(distances < 0) or np.any(weights <= 0):
        raise ValueError("Distances must be nonnegative and weights positive")
    if not 2 <= n_clusters < sample_count:
        raise ValueError("n_clusters must be between 2 and n_samples - 1")

    runs: list[tuple[int, float, np.ndarray, np.ndarray]] = []
    for seed in seeds:
        medoids = _weighted_kmedoids_plus_plus(
            distances, weights, n_clusters, seed
        )
        for _ in range(max_iterations):
            labels = np.argmin(distances[:, medoids], axis=1)
            labels[medoids] = np.arange(n_clusters)
            updated = medoids.copy()
            for cluster in range(n_clusters):
                members = np.flatnonzero(labels == cluster)
                objectives = (
                    distances[np.ix_(members, members)] @ weights[members]
                )
                updated[cluster] = members[int(np.argmin(objectives))]
            if np.array_equal(updated, medoids):
                break
            medoids = updated

        labels = np.argmin(distances[:, medoids], axis=1)
        labels[medoids] = np.arange(n_clusters)
        inertia = float(
            np.sum(weights * distances[np.arange(sample_count), medoids[labels]])
        )
        runs.append((seed, inertia, labels, medoids))

    selected_seed, inertia, labels, medoids = min(runs, key=lambda run: run[1])
    labelings = tuple(run[2] for run in runs)
    result = ClusteringResult(
        labels=labels,
        selected_seed=selected_seed,
        weighted_inertia=inertia,
        mean_seed_stability_ari=mean_pairwise_ari(labelings),
        seed_labels=labelings,
    )
    return result, medoids


def _weighted_kmedoids_plus_plus(
    distances: np.ndarray,
    weights: np.ndarray,
    n_clusters: int,
    seed: int,
) -> np.ndarray:
    random = np.random.default_rng(seed)
    sample_count = len(weights)
    medoids = [
        int(random.choice(sample_count, p=weights / weights.sum()))
    ]
    while len(medoids) < n_clusters:
        nearest = np.min(distances[:, medoids], axis=1)
        probabilities = weights * np.square(nearest)
        probabilities[medoids] = 0.0
        if probabilities.sum() == 0:
            candidate = next(
                index for index in range(sample_count) if index not in medoids
            )
        else:
            candidate = int(
                random.choice(
                    sample_count,
                    p=probabilities / probabilities.sum(),
                )
            )
        medoids.append(candidate)
    return np.asarray(medoids, dtype=np.int64)
