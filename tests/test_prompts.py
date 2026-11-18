from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_variant_clustering.prompts import (
    canonical_signature_text,
    render_prompt,
    validate_response,
)

ROOT = Path(__file__).parents[1]


def test_all_json_schemas_are_valid() -> None:
    from jsonschema import Draft202012Validator

    for path in (ROOT / "configs" / "schemas").glob("*.schema.json"):
        Draft202012Validator.check_schema(
            json.loads(path.read_text(encoding="utf-8"))
        )


def test_render_prompt_is_deterministic() -> None:
    template = ROOT / "configs" / "prompts" / "p2_evidence_grounded_signature.txt"
    schema = ROOT / "configs" / "schemas" / "variant_signature_p2.schema.json"
    values = {
        "VARIANT_ID": "v_123",
        "ACTIVITY_SEQUENCE_JSON": '["Create", "Approve"]',
    }

    first = render_prompt(template, schema, values, prompt_version="p2")
    second = render_prompt(template, schema, values, prompt_version="p2")

    assert first == second
    assert "<VARIANT_ID>" not in first.rendered_prompt
    assert len(first.prompt_sha256) == 64
    assert len(first.schema_sha256) == 64


def test_render_prompt_rejects_missing_or_extra_values() -> None:
    template = ROOT / "configs" / "prompts" / "p1_minimal_signature.txt"
    schema = ROOT / "configs" / "schemas" / "variant_signature_p1.schema.json"

    with pytest.raises(ValueError, match="missing"):
        render_prompt(template, schema, {}, prompt_version="p1")


def test_response_validation_checks_id_and_evidence_bounds() -> None:
    schema = ROOT / "configs" / "schemas" / "variant_signature_p2.schema.json"
    claim = {
        "value": "request",
        "inference_type": "observed",
        "evidence_activity_indices": [0],
        "confidence": 0.9,
    }
    response = {
        "schema_version": "p2",
        "variant_id": "v_123",
        "primary_object": claim,
        "apparent_goal": claim,
        "ordered_stages": [
            {
                "name": "initiation",
                "inference_type": "inferred",
                "evidence_activity_indices": [0],
                "confidence": 0.8,
            }
        ],
        "explicit_outcome": {
            "value": "unknown",
            "inference_type": "unknown",
            "evidence_activity_indices": [],
            "confidence": 0.1,
        },
        "exception_or_correction": {
            "value": "unknown",
            "inference_type": "unknown",
            "evidence_activity_indices": [],
            "confidence": 0.1,
        },
        "short_description": claim,
        "ambiguities": [],
    }

    validate_response(
        response,
        schema,
        expected_variant_id="v_123",
        sequence_length=2,
    )

    response["primary_object"]["evidence_activity_indices"] = [2]
    with pytest.raises(ValueError, match="outside a sequence"):
        validate_response(response, schema, sequence_length=2)


def test_canonical_signature_excludes_audit_tokens() -> None:
    response = {
        "schema_version": "p2",
        "primary_object": {
            "value": "request",
            "inference_type": "observed",
            "evidence_activity_indices": [0],
            "confidence": 0.9,
        },
        "apparent_goal": {
            "value": "approval",
            "inference_type": "inferred",
            "evidence_activity_indices": [1],
            "confidence": 0.7,
        },
        "ordered_stages": [
            {
                "name": "submission",
                "inference_type": "inferred",
                "evidence_activity_indices": [0],
                "confidence": 1,
            }
        ],
        "explicit_outcome": {
            "value": "unknown",
            "inference_type": "unknown",
            "evidence_activity_indices": [],
            "confidence": 0.1,
        },
        "exception_or_correction": {
            "value": "unknown",
            "inference_type": "unknown",
            "evidence_activity_indices": [],
            "confidence": 0.1,
        },
        "short_description": {
            "value": "request submission followed by approval",
            "inference_type": "observed",
            "evidence_activity_indices": [0, 1],
            "confidence": 0.9,
        },
    }

    serialized = canonical_signature_text(response)

    assert "request" in serialized
    assert "confidence" not in serialized
    assert "evidence_activity_indices" not in serialized
    assert "observed" not in serialized

