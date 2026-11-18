from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from llm_variant_clustering.dynamic_clustering import (
    DynamicKConfig,
    NoEligibleKError,
    select_dynamic_k,
)
from llm_variant_clustering.features import FeatureSet


def test_dynamic_k_selects_separated_groups() -> None:
    matrix = csr_matrix(
        np.asarray(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.98, 0.02],
                [0.0, 1.0],
                [0.01, 0.99],
                [0.02, 0.98],
            ]
        )
    )
    features = FeatureSet(
        name="synthetic",
        matrix=matrix,
        variant_ids=tuple(f"v{index}" for index in range(6)),
        feature_names=("x", "y"),
        metric="cosine",
    )
    config = DynamicKConfig(
        min_k=2,
        max_k=3,
        min_seed_stability_ari=0.0,
        min_case_cluster_entropy=0.0,
        min_cluster_case_fraction=0.0,
        min_cluster_variants=1,
    )

    selection = select_dynamic_k(features, np.ones(6), config)

    assert selection.selected_k == 2
    assert selection.candidates["selected"].sum() == 1


def test_dynamic_config_round_trip() -> None:
    config = DynamicKConfig()
    assert DynamicKConfig.from_dict(config.to_dict()) == config


def test_no_eligible_k_exposes_candidate_diagnostics() -> None:
    matrix = csr_matrix(np.eye(4))
    features = FeatureSet(
        name="synthetic",
        matrix=matrix,
        variant_ids=("v0", "v1", "v2", "v3"),
        feature_names=("a", "b", "c", "d"),
        metric="cosine",
    )
    config = DynamicKConfig(
        min_k=2,
        max_k=2,
        min_seed_stability_ari=0.0,
        min_case_cluster_entropy=0.0,
        min_cluster_case_fraction=0.0,
        min_cluster_variants=3,
    )

    with pytest.raises(NoEligibleKError) as caught:
        select_dynamic_k(features, np.ones(4), config)

    assert caught.value.candidates["k"].tolist() == [2]
    assert not caught.value.candidates["eligible"].any()
