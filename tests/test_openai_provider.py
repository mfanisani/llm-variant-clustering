from __future__ import annotations

import pytest

from llm_variant_clustering.llm_cache import GenerationParameters
from llm_variant_clustering.openai_provider import OpenAIResponsesProvider


def test_openai_provider_requires_environment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIResponsesProvider(
            model="model",
            parameters=GenerationParameters(
                temperature=0.0,
                seed=None,
                max_output_tokens=100,
            ),
            schema={"type": "object"},
            schema_name="schema",
        )


def test_openai_provider_rejects_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "not-used")
    with pytest.raises(ValueError, match="does not support a seed"):
        OpenAIResponsesProvider(
            model="model",
            parameters=GenerationParameters(
                temperature=0.0,
                seed=17,
                max_output_tokens=100,
            ),
            schema={"type": "object"},
            schema_name="schema",
        )

