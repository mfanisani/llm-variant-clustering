from __future__ import annotations

from pathlib import Path

from llm_variant_clustering.llm_cache import (
    GenerationParameters,
    ModelIdentity,
    StructuredResponseCache,
    ProviderGeneration,
    canonical_json_hash,
    execute_cached_request,
)
from llm_variant_clustering.prompts import PromptArtifact


def test_cached_request_calls_provider_once(tmp_path: Path) -> None:
    calls = 0

    def generate(_: str) -> dict:
        nonlocal calls
        calls += 1
        return {"answer": "valid"}

    def validate(response: dict) -> None:
        if response != {"answer": "valid"}:
            raise ValueError("invalid")

    cache = StructuredResponseCache(tmp_path / "cache")
    model = ModelIdentity("test", "model", "model-2026-08-12")
    parameters = GenerationParameters(temperature=0.0, seed=17, max_output_tokens=500)
    prompt = PromptArtifact("p2", "rendered", "a" * 64, "b" * 64)
    input_hash = canonical_json_hash(["A", "B"])

    first, first_hit = execute_cached_request(
        cache=cache,
        model=model,
        parameters=parameters,
        prompt=prompt,
        input_sha256=input_hash,
        generate=generate,
        validate=validate,
    )
    second, second_hit = execute_cached_request(
        cache=cache,
        model=model,
        parameters=parameters,
        prompt=prompt,
        input_sha256=input_hash,
        generate=generate,
        validate=validate,
    )

    assert first == second == {"answer": "valid"}
    assert first_hit is False
    assert second_hit is True
    assert calls == 1


def test_canonical_hash_ignores_mapping_order() -> None:
    assert canonical_json_hash({"a": 1, "b": 2}) == canonical_json_hash(
        {"b": 2, "a": 1}
    )


def test_provider_metadata_is_persisted(tmp_path: Path) -> None:
    cache = StructuredResponseCache(tmp_path / "cache")
    model = ModelIdentity("provider", "model", "version")
    parameters = GenerationParameters(temperature=0.0, seed=None, max_output_tokens=100)
    prompt = PromptArtifact("p1", "prompt", "c" * 64, "d" * 64)

    response, cache_hit = execute_cached_request(
        cache=cache,
        model=model,
        parameters=parameters,
        prompt=prompt,
        input_sha256=canonical_json_hash(["A"]),
        generate=lambda _: ProviderGeneration(
            response={"answer": "valid"},
            provider_response_id="response-1",
            provider_model="model-snapshot",
            usage={"input_tokens": 10, "output_tokens": 2},
            attempt_count=2,
        ),
        validate=lambda value: None,
    )

    assert response == {"answer": "valid"}
    assert cache_hit is False
    record_path = next((tmp_path / "cache").rglob("*.json"))
    record = __import__("json").loads(record_path.read_text(encoding="utf-8"))
    assert record["provider_metadata"]["provider_response_id"] == "response-1"
    assert record["provider_metadata"]["usage"]["input_tokens"] == 10
    assert record["provider_metadata"]["attempt_count"] == 2
