"""Bootstrap uncertainty for clustering-quality and paired method comparisons.

Point estimates alone cannot justify claims about differences of a few
thousandths. These helpers resample variants (with case frequency preserved as
the analysis weight) to produce percentile confidence intervals for a single
method and for the paired difference between two methods evaluated on the same
variant set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .evaluation import frequency_weighted_silhouette

CLUSTER_SCOPE = "cluster"


@dataclass(frozen=True)
class BootstrapInterval:
    point_estimate: float
    lower: float
    upper: float
    confidence: float
    resamples: int

    def as_dict(self, prefix: str = "") -> dict[str, float | int]:
        return {
            f"{prefix}point_estimate": self.point_estimate,
            f"{prefix}ci_lower": self.lower,
            f"{prefix}ci_upper": self.upper,
            f"{prefix}confidence": self.confidence,
            f"{prefix}resamples": self.resamples,
        }


@dataclass(frozen=True)
class PairedDifference:
    interval: BootstrapInterval
    excludes_zero: bool
    proportion_favouring_first: float
    samples: np.ndarray | None = field(default=None, repr=False, compare=False)

    def adjusted_interval(self, confidence: float) -> BootstrapInterval:
        """Re-read the stored bootstrap samples at another confidence level.

        Used for family-wise (for example Bonferroni) adjustment without
        drawing a second, different set of resamples.
        """
        if self.samples is None:
            raise ValueError("Bootstrap samples were not retained")
        _validate(confidence, self.samples.size)
        return _percentile_interval(
            self.interval.point_estimate, self.samples, confidence
        )

    def as_dict(self, prefix: str = "") -> dict[str, float | int | bool]:
        record: dict[str, float | int | bool] = self.interval.as_dict(prefix)
        record[f"{prefix}excludes_zero"] = self.excludes_zero
        record[f"{prefix}proportion_favouring_first"] = (
            self.proportion_favouring_first
        )
        return record


def _validate(confidence: float, resamples: int) -> None:
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if resamples < 2:
        raise ValueError("resamples must be at least 2")


def _percentile_interval(
    point_estimate: float,
    samples: np.ndarray,
    confidence: float,
) -> BootstrapInterval:
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.percentile(samples, [100 * tail, 100 * (1 - tail)])
    return BootstrapInterval(
        point_estimate=float(point_estimate),
        lower=float(lower),
        upper=float(upper),
        confidence=float(confidence),
        resamples=int(samples.size),
    )


def _resample_indices(
    size: int,
    resamples: int,
    seed: int,
    sample_size: int | None = None,
) -> np.ndarray:
    draw = size if sample_size is None else min(sample_size, size)
    if draw < 2:
        raise ValueError("sample_size must allow at least two variants")
    generator = np.random.default_rng(seed)
    return generator.integers(0, size, size=(resamples, draw))


def _silhouette_on_subset(
    distances: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    indices: np.ndarray,
) -> float | None:
    subset_labels = labels[indices]
    if np.unique(subset_labels).size < 2:
        return None
    subset_distances = distances[np.ix_(indices, indices)]
    return frequency_weighted_silhouette(
        subset_distances,
        subset_labels,
        weights[indices],
    )


def bootstrap_silhouette(
    distances: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    resamples: int = 500,
    confidence: float = 0.95,
    seed: int = 20240601,
    sample_size: int | None = None,
) -> BootstrapInterval:
    """Percentile interval for the case-weighted silhouette of one partition.

    ``sample_size`` enables an unrescaled m-out-of-n bootstrap for logs where a
    full-size resample is computationally prohibitive. Its spread reflects
    samples of ``m`` variants rather than ``n``, and silhouette is not a simple
    mean, so the interval is neither guaranteed to be wider than the full
    bootstrap nor guaranteed to be centred on the full-data estimate. Report
    a cap-sensitivity check instead of treating it as conservative.
    """
    _validate(confidence, resamples)
    labels = np.asarray(labels)
    weights = np.asarray(weights, dtype=np.float64)
    if distances.shape[0] != labels.size or labels.size != weights.size:
        raise ValueError("distances, labels, and weights must be aligned")

    point_estimate = frequency_weighted_silhouette(distances, labels, weights)
    draws = _resample_indices(labels.size, resamples, seed, sample_size)
    samples = []
    for indices in draws:
        value = _silhouette_on_subset(distances, labels, weights, indices)
        if value is not None:
            samples.append(value)
    if len(samples) < 2:
        raise ValueError("Too few valid bootstrap resamples to form an interval")
    return _percentile_interval(point_estimate, np.asarray(samples), confidence)


def paired_silhouette_difference(
    first_distances: np.ndarray,
    first_labels: np.ndarray,
    second_distances: np.ndarray,
    second_labels: np.ndarray,
    weights: np.ndarray,
    *,
    resamples: int = 500,
    confidence: float = 0.95,
    seed: int = 20240601,
    sample_size: int | None = None,
) -> PairedDifference:
    """Paired interval for ``first`` minus ``second`` on identical variants.

    The same resampled variant indices are used for both methods, which removes
    variant-level variance shared by the two partitions and makes small but
    systematic differences detectable. ``sample_size`` selects an m-out-of-n
    bootstrap for large logs.
    """
    _validate(confidence, resamples)
    weights = np.asarray(weights, dtype=np.float64)
    first_labels = np.asarray(first_labels)
    second_labels = np.asarray(second_labels)
    if first_labels.size != second_labels.size or first_labels.size != weights.size:
        raise ValueError("Both partitions must cover the same variants")
    if (
        first_distances.shape[0] != weights.size
        or second_distances.shape[0] != weights.size
    ):
        raise ValueError("Distance matrices must match the variant count")

    point_estimate = frequency_weighted_silhouette(
        first_distances, first_labels, weights
    ) - frequency_weighted_silhouette(second_distances, second_labels, weights)

    draws = _resample_indices(weights.size, resamples, seed, sample_size)
    samples = []
    for indices in draws:
        first_value = _silhouette_on_subset(
            first_distances, first_labels, weights, indices
        )
        second_value = _silhouette_on_subset(
            second_distances, second_labels, weights, indices
        )
        if first_value is None or second_value is None:
            continue
        samples.append(first_value - second_value)
    if len(samples) < 2:
        raise ValueError("Too few valid bootstrap resamples to form an interval")

    sample_array = np.asarray(samples)
    interval = _percentile_interval(point_estimate, sample_array, confidence)
    return PairedDifference(
        interval=interval,
        excludes_zero=bool(interval.lower > 0 or interval.upper < 0),
        proportion_favouring_first=float(np.mean(sample_array > 0)),
        samples=sample_array,
    )


def cluster_scope_mask(scopes) -> np.ndarray:
    """Boolean mask selecting rows that describe one discovered cluster.

    Process-model metric tables also contain ``whole_log`` and
    ``case_weighted_aggregate`` summary rows. An aggregate row carries the
    total case count of the clusters it summarises, so treating it as another
    cluster during resampling double-counts those cases. Only rows whose scope
    is exactly ``cluster`` are valid bootstrap units.
    """
    return np.asarray(scopes, dtype=object) == CLUSTER_SCOPE


def bootstrap_cluster_metric(
    values: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 20240601,
) -> BootstrapInterval:
    """Interval for a weighted mean over per-cluster values such as process F1.

    Used when the unit of resampling is a discovered cluster rather than a
    variant, for example when averaging model-quality scores across clusters.
    """
    _validate(confidence, resamples)
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        raise ValueError("At least two values are required")
    if weights is None:
        weights = np.ones_like(values)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.size != values.size:
        raise ValueError("values and weights must be aligned")
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be non-negative with a positive sum")

    point_estimate = float(np.average(values, weights=weights))
    draws = _resample_indices(values.size, resamples, seed)
    samples = np.array(
        [
            np.average(values[indices], weights=weights[indices])
            for indices in draws
            if weights[indices].sum() > 0
        ]
    )
    if samples.size < 2:
        raise ValueError("Too few valid bootstrap resamples to form an interval")
    return _percentile_interval(point_estimate, samples, confidence)


def _mean_distinct_pairs(matrix: np.ndarray, indices: np.ndarray) -> float | None:
    sub = matrix[np.ix_(indices, indices)]
    distinct = indices[:, None] != indices[None, :]
    upper = np.triu(distinct, k=1)
    if not upper.any():
        return None
    return float(sub[upper].mean())


def bootstrap_seed_agreement(
    agreement: np.ndarray,
    *,
    second: np.ndarray | None = None,
    resamples: int = 4000,
    confidence: float = 0.95,
    seed: int = 20240601,
) -> BootstrapInterval | PairedDifference:
    """Seed-level bootstrap for a mean pairwise agreement such as ARI.

    The pairwise agreements among ``s`` reseeded clusterings share labelings,
    so the ``s(s-1)/2`` pairs are not independent. This bootstraps the seeds:
    each replicate draws ``s`` seeds with replacement and averages the
    agreement over pairs of draws that refer to different seeds (a U-statistic
    bootstrap). With ``second`` supplied, the same seed draws are applied to
    both symmetric matrices and the interval is for the difference
    ``first - second``.
    """
    _validate(confidence, resamples)
    first = np.asarray(agreement, dtype=np.float64)
    if first.ndim != 2 or first.shape[0] != first.shape[1] or first.shape[0] < 3:
        raise ValueError("agreement must be a square matrix over at least 3 seeds")
    if not np.allclose(first, first.T):
        raise ValueError("agreement matrix must be symmetric")
    other = None
    if second is not None:
        other = np.asarray(second, dtype=np.float64)
        if other.shape != first.shape or not np.allclose(other, other.T):
            raise ValueError("second must be a symmetric matrix of the same shape")
    size = first.shape[0]
    every = np.arange(size)
    point = _mean_distinct_pairs(first, every)
    if other is not None:
        point -= _mean_distinct_pairs(other, every)
    samples = []
    for indices in _resample_indices(size, resamples, seed):
        value = _mean_distinct_pairs(first, indices)
        if value is None:
            continue
        if other is not None:
            value -= _mean_distinct_pairs(other, indices)
        samples.append(value)
    sample_array = np.asarray(samples)
    if sample_array.size < 2:
        raise ValueError("Too few valid bootstrap resamples to form an interval")
    interval = _percentile_interval(point, sample_array, confidence)
    if other is None:
        return interval
    return PairedDifference(
        interval=interval,
        excludes_zero=bool(interval.lower > 0 or interval.upper < 0),
        proportion_favouring_first=float(np.mean(sample_array > 0)),
        samples=sample_array,
    )
