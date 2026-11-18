import json

import numpy as np
import pandas as pd
import pytest

from llm_variant_clustering.features import (
    activity2vec_svd_features,
    activity_context_embeddings,
)


def _variants() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variant_id": ["v1", "v2", "v3"],
            "activity_sequence": [
                json.dumps(["submit", "approve", "pay"]),
                json.dumps(["submit", "reject", "submit", "approve"]),
                json.dumps(["approve", "pay"]),
            ],
            "frequency": [10, 4, 2],
        }
    )


def test_activity_context_embeddings_cover_vocabulary() -> None:
    embeddings = activity_context_embeddings(_variants(), embedding_dim=3)

    assert set(embeddings) == {"submit", "approve", "pay", "reject"}
    for vector in embeddings.values():
        assert vector.ndim == 1
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


def test_activity_context_embeddings_match_variant_features() -> None:
    variants = _variants()
    embeddings = activity_context_embeddings(variants, embedding_dim=3)
    features = activity2vec_svd_features(variants, embedding_dim=3)

    expected = np.mean(
        [embeddings[activity] for activity in ["approve", "pay"]],
        axis=0,
    )
    expected = expected / np.linalg.norm(expected)
    actual = features.matrix.toarray()[2]

    assert actual == pytest.approx(expected, abs=1e-5)


def test_activity_context_embeddings_reject_invalid_dimension() -> None:
    with pytest.raises(ValueError, match="embedding_dim must be positive"):
        activity_context_embeddings(_variants(), embedding_dim=0)
