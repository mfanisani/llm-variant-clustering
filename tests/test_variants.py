from __future__ import annotations

import json

import pandas as pd
import pytest

from llm_variant_clustering.variants import extract_variants, variant_id


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case:concept:name": ["c1", "c2", "c1", "c2", "c3"],
            "concept:name": ["A", "A", "B", "B", "C"],
            "time:timestamp": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T01:00:00Z",
                    "2024-01-01T01:00:00Z",
                    "2024-01-01T00:00:00Z",
                ]
            ),
        }
    )


def test_extracts_frequency_preserving_variants() -> None:
    variants, case_map, summary = extract_variants(_events(), source="synthetic")

    assert summary.cases == 3
    assert summary.variants == 2
    assert summary.lifecycle_policy_applied == "all_column_absent"
    assert variants["frequency"].tolist() == [2, 1]
    assert json.loads(variants.iloc[0]["activity_sequence"]) == ["A", "B"]
    assert case_map["variant_id"].value_counts().max() == 2


def test_variant_hash_is_stable_and_order_sensitive() -> None:
    assert variant_id(("A", "B")) == variant_id(("A", "B"))
    assert variant_id(("A", "B")) != variant_id(("B", "A"))


def test_complete_policy_rejects_missing_lifecycle_column() -> None:
    with pytest.raises(ValueError, match="requires column"):
        extract_variants(
            _events(), source="synthetic", lifecycle_policy="complete"
        )


def test_complete_policy_filters_noncomplete_events() -> None:
    events = _events()
    events["lifecycle:transition"] = [
        "start",
        "complete",
        "complete",
        "complete",
        "complete",
    ]

    variants, case_map, summary = extract_variants(
        events, source="synthetic", lifecycle_policy="complete"
    )

    assert summary.events_after_filter == 4
    assert summary.lifecycle_policy_applied == "complete"
    assert len(case_map) == 3
    assert set(variants["activity_sequence"]) == {'["A", "B"]', '["B"]', '["C"]'}

