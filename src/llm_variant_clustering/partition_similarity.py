from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def partition_similarity(
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    frequencies: np.ndarray,
) -> dict[str, float]:
    left = np.asarray(left_labels)
    right = np.asarray(right_labels)
    weights = np.asarray(frequencies)
    if left.ndim != 1 or right.ndim != 1 or weights.ndim != 1:
        raise ValueError("Labels and frequencies must be one-dimensional")
    if len(left) != len(right) or len(left) != len(weights):
        raise ValueError("Labels and frequencies must have equal lengths")
    if len(left) == 0:
        raise ValueError("At least one variant is required")
    if not np.issubdtype(weights.dtype, np.integer) or np.any(weights <= 0):
        raise ValueError("Frequencies must be positive integers")

    expanded_left = np.repeat(left, weights)
    expanded_right = np.repeat(right, weights)
    return {
        "variant_ari": float(adjusted_rand_score(left, right)),
        "variant_nmi": float(normalized_mutual_info_score(left, right)),
        "case_weighted_ari": float(
            adjusted_rand_score(expanded_left, expanded_right)
        ),
        "case_weighted_nmi": float(
            normalized_mutual_info_score(expanded_left, expanded_right)
        ),
        "optimal_variant_agreement": _optimal_label_agreement(
            left,
            right,
            np.ones(len(left), dtype=np.int64),
        ),
        "optimal_case_agreement": _optimal_label_agreement(
            left,
            right,
            weights,
        ),
    }


def _optimal_label_agreement(
    left: np.ndarray,
    right: np.ndarray,
    weights: np.ndarray,
) -> float:
    left_values, left_codes = np.unique(left, return_inverse=True)
    right_values, right_codes = np.unique(right, return_inverse=True)
    contingency = np.zeros(
        (len(left_values), len(right_values)),
        dtype=np.float64,
    )
    np.add.at(contingency, (left_codes, right_codes), weights)
    rows, columns = linear_sum_assignment(-contingency)
    return float(contingency[rows, columns].sum() / weights.sum())
