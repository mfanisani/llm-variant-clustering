from __future__ import annotations

import json

import pandas as pd
import pytest

from llm_variant_clustering.sampling import stratified_variant_sample


def _variants(size: int = 30) -> pd.DataFrame:
    rows = []
    terminals = ["Complete", "Reject", "Cancel"]
    for index in range(size):
        sequence = ["Start", f"Step {index % 5}", terminals[index % 3]]
        if index % 2:
            sequence.insert(2, f"Step {index % 5}")
        rows.append(
            {
                "variant_id": f"v_{index:03d}",
                "activity_sequence": json.dumps(sequence),
                "frequency": index + 1,
                "sequence_length": len(sequence),
            }
        )
    return pd.DataFrame(rows)


def test_sampling_is_deterministic_diverse_and_disjoint() -> None:
    variants = _variants()
    first = stratified_variant_sample(variants, sample_size=10, seed="seed")
    second = stratified_variant_sample(variants, sample_size=10, seed="seed")

    pd.testing.assert_frame_equal(first, second)
    assert first["terminal_activity"].nunique() == 3
    assert first["has_repetition"].nunique() == 2

    validation = stratified_variant_sample(
        variants,
        sample_size=10,
        seed="validation",
        excluded_variant_ids=set(first["variant_id"]),
    )
    assert set(first["variant_id"]).isdisjoint(validation["variant_id"])


def test_sampling_rejects_invalid_size() -> None:
    with pytest.raises(ValueError, match="Requested"):
        stratified_variant_sample(_variants(3), sample_size=4, seed="seed")

