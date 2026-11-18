import numpy as np
import pytest

from llm_variant_clustering.partition_similarity import partition_similarity


def test_partition_similarity_ignores_label_permutations() -> None:
    metrics = partition_similarity(
        np.array([0, 0, 1, 1]),
        np.array([7, 7, 3, 3]),
        np.array([1, 2, 3, 4]),
    )

    assert all(value == pytest.approx(1.0) for value in metrics.values())


def test_partition_similarity_uses_case_frequencies() -> None:
    metrics = partition_similarity(
        np.array([0, 0, 1]),
        np.array([0, 1, 1]),
        np.array([100, 1, 1]),
    )

    assert metrics["optimal_variant_agreement"] == pytest.approx(2 / 3)
    assert metrics["optimal_case_agreement"] == pytest.approx(101 / 102)
    assert metrics["variant_ari"] != pytest.approx(metrics["case_weighted_ari"])


def test_partition_similarity_rejects_noninteger_frequencies() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        partition_similarity(
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([1.0, 2.0]),
        )
