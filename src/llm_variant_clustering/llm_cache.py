from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .prompts import PromptArtifact


@dataclass(frozen=True)
class ModelIdentity:
    provider: str
    model: str
    model_version: str


@dataclass(frozen=True)
class GenerationParameters:
    temperature: float
    seed: int | None
    max_output_tokens: int


@dataclass(frozen=True)
class ProviderGeneration:
    response: Mapping[str, Any]
    provider_response_id: str | None
    provider_model: str | None
    usage: Mapping[str, Any]
    attempt_count: int = 1
    request_metadata: Mapping[str, Any] = field(default_factory=dict)


def canonical_json_hash(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def request_cache_key(
    model: ModelIdentity,
    parameters: GenerationParameters,
    prompt: PromptArtifact,
    *,
    input_sha256: str,
) -> str:
    return canonical_json_hash(
        {
            "model": asdict(model),
            "parameters": asdict(parameters),
            "prompt_version": prompt.prompt_version,
            "prompt_sha256": prompt.prompt_sha256,
            "schema_sha256": prompt.schema_sha256,
            "input_sha256": input_sha256,
        }
    )


class StructuredResponseCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, key: str) -> Path:
        if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
            raise ValueError("Cache key must be a lowercase SHA-256 digest")
        return self.root / key[:2] / f"{key}.json"

    def load(self, key: str) -> dict[str, Any] | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("cache_key") != key:
            raise ValueError(f"Cache record key mismatch: {path}")
        if record.get("status") != "validated":
            raise ValueError(f"Cache record is not validated: {path}")
        if not isinstance(record.get("response"), dict):
            raise ValueError(f"Cache response is not a JSON object: {path}")
        return record

    def store(self, key: str, record: Mapping[str, Any]) -> Path:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        complete_record = dict(record)
        complete_record["cache_key"] = key
        serialized = (
            json.dumps(
                complete_record,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{key}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.replace(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return path


def execute_cached_request(
    *,
    cache: StructuredResponseCache,
    model: ModelIdentity,
    parameters: GenerationParameters,
    prompt: PromptArtifact,
    input_sha256: str,
    generate: Callable[[str], Mapping[str, Any] | ProviderGeneration],
    validate: Callable[[Mapping[str, Any]], None],
) -> tuple[dict[str, Any], bool]:
    key = request_cache_key(
        model,
        parameters,
        prompt,
        input_sha256=input_sha256,
    )
    cached = cache.load(key)
    if cached is not None:
        response = dict(cached["response"])
        validate(response)
        return response, True

    started_at = datetime.now(UTC)
    provider_generation = generate(prompt.rendered_prompt)
    if isinstance(provider_generation, ProviderGeneration):
        generated = dict(provider_generation.response)
        provider_metadata = {
            "provider_response_id": provider_generation.provider_response_id,
            "provider_model": provider_generation.provider_model,
            "usage": dict(provider_generation.usage),
            "attempt_count": provider_generation.attempt_count,
            "request_metadata": dict(provider_generation.request_metadata),
        }
    else:
        generated = dict(provider_generation)
        provider_metadata = {
            "provider_response_id": None,
            "provider_model": None,
            "usage": {},
            "attempt_count": 1,
        }
    validate(generated)
    completed_at = datetime.now(UTC)
    record = {
        "status": "validated",
        "model": asdict(model),
        "parameters": asdict(parameters),
        "prompt_version": prompt.prompt_version,
        "prompt_sha256": prompt.prompt_sha256,
        "schema_sha256": prompt.schema_sha256,
        "input_sha256": input_sha256,
        "rendered_prompt": prompt.rendered_prompt,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_ms": int((completed_at - started_at).total_seconds() * 1000),
        "provider_metadata": provider_metadata,
        "response": generated,
    }
    cache.store(key, record)
    return generated, False
