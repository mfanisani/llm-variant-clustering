from __future__ import annotations

import pytest

from llm_variant_clustering.azure_openai_provider import (
    AzureOpenAIToolProvider,
    azure_strict_tool_schema,
)
from llm_variant_clustering.llm_cache import GenerationParameters


def test_azure_provider_rejects_non_https_endpoint() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        AzureOpenAIToolProvider(
            endpoint="http://example.test",
            deployment="deployment",
            api_version="version",
            parameters=GenerationParameters(
                temperature=0.0,
                seed=None,
                max_output_tokens=100,
            ),
            schema={"type": "object"},
            schema_name="schema",
        )


def test_azure_schema_conversion_keeps_local_schema_unchanged() -> None:
    schema = {
        "$schema": "ignored",
        "type": "object",
        "properties": {
            "version": {"const": "p1"},
            "indices": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "integer", "minimum": 0},
            },
        },
    }

    converted = azure_strict_tool_schema(schema)

    assert converted["properties"]["version"] == {
        "enum": ["p1"],
        "type": "string",
    }
    assert converted["properties"]["indices"] == {
        "type": "array",
        "description": (
            "Array items must be unique. The array must contain at least 1 items."
        ),
        "items": {
            "type": "integer",
            "description": "The numeric value must be at least 0.",
        },
    }
    assert schema["properties"]["indices"]["uniqueItems"] is True
