from __future__ import annotations

import json

import numpy as np
import pytest

from llm_variant_clustering.semantic_compiler import (
    compose_variant_semantics,
    mean_semantic_embeddings,
    ordered_semantic_pyramid,
    validate_activity_semantics,
)


def semantics(label: str, object_name: str, state: str) -> dict:
    return {
        "schema_version": "p7",
        "activity_label": label,
        "action": state,
        "business_object": object_name,
        "process_stage": "decision",
        "explicit_state": state,
        "explicit_role": "none explicit",
        "ambiguity": "none",
    }


def test_activity_validation_rejects_a_different_label(tmp_path) -> None:
    schema = {
        "type": "object",
        "required": ["activity_label"],
        "properties": {"activity_label": {"type": "string"}},
    }
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")

    with pytest.raises(ValueError, match="different label"):
        validate_activity_semantics(
            {"activity_label": "changed"},
            schema_path=schema_path,
            expected_activity_label="original",
        )


def test_activity_validation_tolerates_case_and_whitespace(tmp_path) -> None:
    """A model that lowercases the echo has still answered the right question.

    Qwen2.5-7B applies the prompt's lowercasing rule to the echo field, which
    previously rejected 10 of 21 labels on an out-of-sample log despite the
    descriptors themselves being correct.
    """
    schema = {
        "type": "object",
        "required": ["activity_label"],
        "properties": {"activity_label": {"type": "string"}},
    }
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")

    for echoed in ("status change to Approved", " Status Change To Approved "):
        validate_activity_semantics(
            {"activity_label": echoed},
            schema_path=schema_path,
            expected_activity_label="Status change to Approved",
        )


def test_composer_preserves_exact_order_and_finds_resubmission() -> None:
    rejected = "Declaration REJECTED"
    submitted = "Declaration SUBMITTED"
    mapping = {
        rejected: semantics(rejected, "declaration", "rejected"),
        submitted: semantics(submitted, "declaration", "submitted"),
    }

    result = compose_variant_semantics(
        [submitted, rejected, submitted],
        mapping,
    )

    assert result["ordered_activity_labels"] == [submitted, rejected, submitted]
    assert result["start"]["index"] == 0
    assert result["end"]["index"] == 2
    assert result["repeated_activities"][0]["indices"] == [0, 2]
    assert result["rejection_resubmissions"] == [
        {
            "business_object": "declaration",
            "rejection_index": 1,
            "resubmission_index": 2,
        }
    ]


def test_ordered_pyramid_distinguishes_reversed_sequences() -> None:
    embeddings = {
        "a": np.asarray([1.0, 0.0], dtype=np.float32),
        "b": np.asarray([0.0, 1.0], dtype=np.float32),
    }

    mean = mean_semantic_embeddings([["a", "b"], ["b", "a"]], embeddings)
    pyramid = ordered_semantic_pyramid(
        [["a", "b"], ["b", "a"]],
        embeddings,
        levels=(1, 2),
    )

    assert np.allclose(mean[0], mean[1])
    assert not np.allclose(pyramid[0], pyramid[1])
    assert np.allclose(np.linalg.norm(pyramid, axis=1), 1.0)

