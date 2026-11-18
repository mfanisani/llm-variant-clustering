from __future__ import annotations

import numpy as np
import pytest

from llm_variant_clustering.embeddings import (
    serialize_activity_sequence,
    validate_embedding_matrix,
)


def test_sequence_serialization_preserves_order() -> None:
    assert serialize_activity_sequence(["Create", "Approve", "Close"]) == (
        "Create -> Approve -> Close"
    )
    assert serialize_activity_sequence(["Close", "Approve", "Create"]) != (
        "Create -> Approve -> Close"
    )


def test_embedding_validation_requires_normalized_finite_rows() -> None:
    valid = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    validate_embedding_matrix(valid, expected_rows=2)

    with pytest.raises(ValueError, match="L2-normalized"):
        validate_embedding_matrix(np.array([[2.0, 0.0]]), expected_rows=1)
    with pytest.raises(ValueError, match="nonfinite"):
        validate_embedding_matrix(
            np.array([[np.nan, 0.0]]), expected_rows=1
        )

