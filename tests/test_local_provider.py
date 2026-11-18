from __future__ import annotations

import pytest

from llm_variant_clustering.local_provider import (
    extract_json_object,
    split_rendered_prompt,
)


def test_split_rendered_prompt() -> None:
    system, user = split_rendered_prompt("SYSTEM\nRules\n\nUSER\nVariant")
    assert system == "Rules"
    assert user == "Variant"


def test_extract_json_object_handles_plain_and_fenced_json() -> None:
    assert extract_json_object('{"value": 1}') == {"value": 1}
    assert extract_json_object('```json\n{"value": 2}\n```') == {"value": 2}


def test_extract_json_object_rejects_missing_object() -> None:
    with pytest.raises(ValueError, match="does not contain"):
        extract_json_object("not json")

