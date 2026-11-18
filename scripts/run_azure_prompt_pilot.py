from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import pandas as pd
from openai import OpenAIError

from llm_variant_clustering.azure_openai_provider import AzureOpenAIToolProvider
from llm_variant_clustering.llm_cache import (
    GenerationParameters,
    ModelIdentity,
    StructuredResponseCache,
    canonical_json_hash,
    execute_cached_request,
    ProviderGeneration,
)
from llm_variant_clustering.prompts import render_prompt, validate_response
from run_openai_prompt_pilot import PROMPTS


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_entries(
    prompt_versions: list[str],
    bundle_dir: Path | None,
) -> dict[str, tuple[Path, Path]]:
    if bundle_dir is None:
        return {version: PROMPTS[version] for version in prompt_versions}
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen":
        raise ValueError("Prompt bundle is not frozen")
    version = manifest["prompt_version"]
    prompt = bundle_dir / manifest["prompt_file"]
    schema = bundle_dir / manifest["schema_file"]
    if file_sha256(prompt) != manifest["prompt_file_sha256"]:
        raise ValueError("Frozen prompt hash mismatch")
    if file_sha256(schema) != manifest["schema_file_sha256"]:
        raise ValueError("Frozen schema hash mismatch")
    if prompt_versions != ["p1", "p2", "p3"] and prompt_versions != [version]:
        raise ValueError("Prompt-version argument conflicts with frozen bundle")
    return {version: (prompt, schema)}


def generate_with_validation_repairs(
    rendered_prompt: str,
    *,
    provider: AzureOpenAIToolProvider,
    validator: object,
    max_validation_attempts: int = 3,
) -> ProviderGeneration:
    validation_errors = []
    total_provider_attempts = 0
    current_prompt = rendered_prompt
    generation = None
    for validation_attempt in range(1, max_validation_attempts + 1):
        generation = provider.generate(current_prompt)
        total_provider_attempts += generation.attempt_count
        try:
            validator(generation.response)  # type: ignore[operator]
        except ValueError as error:
            validation_errors.append(str(error))
            if validation_attempt == max_validation_attempts:
                break
            current_prompt = (
                rendered_prompt
                + "\n\nVALIDATION CORRECTION\n"
                + "The previous tool arguments failed local validation. Produce a "
                + "fresh tool call for the same variant and activity sequence. "
                + "Correct every listed problem without dropping required fields or "
                + "inventing activities:\n"
                + str(error)
            )
        else:
            return replace(
                generation,
                attempt_count=total_provider_attempts,
                request_metadata={
                    "validation_attempt_count": validation_attempt,
                    "repair_prompt_sha256": (
                        hashlib.sha256(current_prompt.encode("utf-8")).hexdigest()
                        if validation_attempt > 1
                        else None
                    ),
                    "prior_validation_errors": validation_errors,
                },
            )
    if generation is None:
        raise RuntimeError("Validation repair loop did not generate a response")
    return replace(
        generation,
        attempt_count=total_provider_attempts,
        request_metadata={
            "validation_attempt_count": max_validation_attempts,
            "repair_prompt_sha256": hashlib.sha256(
                current_prompt.encode("utf-8")
            ).hexdigest(),
            "prior_validation_errors": validation_errors,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cached Azure OpenAI signatures with Entra authentication."
    )
    parser.add_argument("sample", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--api-version", default="2025-01-01-preview")
    parser.add_argument(
        "--prompt-versions",
        nargs="+",
        choices=tuple(PROMPTS),
        default=["p1", "p2", "p3"],
    )
    parser.add_argument("--max-output-tokens", type=int, default=4000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prompt-bundle", type=Path)
    args = parser.parse_args()

    sample = pd.read_parquet(args.sample)
    if args.limit is not None:
        sample = sample.head(args.limit)
    parameters = GenerationParameters(
        temperature=0.0,
        seed=None,
        max_output_tokens=args.max_output_tokens,
    )
    identity = ModelIdentity(
        provider="azure_openai_entra",
        model=args.deployment,
        model_version=args.model_version,
    )
    cache = StructuredResponseCache(args.cache_dir)
    output_rows = []
    entries = prompt_entries(args.prompt_versions, args.prompt_bundle)
    for prompt_version, (template_path, schema_path) in entries.items():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        provider = AzureOpenAIToolProvider(
            endpoint=args.endpoint,
            deployment=args.deployment,
            api_version=args.api_version,
            parameters=parameters,
            schema=schema,
            schema_name=f"variant_signature_{prompt_version}",
        )
        for row in sample.itertuples(index=False):
            sequence = json.loads(row.activity_sequence)
            artifact = render_prompt(
                template_path,
                schema_path,
                {
                    "VARIANT_ID": row.variant_id,
                    "ACTIVITY_SEQUENCE_JSON": json.dumps(
                        sequence, ensure_ascii=False
                    ),
                },
                prompt_version=prompt_version,
            )
            validator = partial(
                validate_response,
                schema_path=schema_path,
                expected_variant_id=row.variant_id,
                activity_sequence=sequence,
            )
            base_row = {
                "variant_id": row.variant_id,
                "prompt_version": prompt_version,
                "prompt_sha256": artifact.prompt_sha256,
                "schema_sha256": artifact.schema_sha256,
            }
            try:
                response, cache_hit = execute_cached_request(
                    cache=cache,
                    model=identity,
                    parameters=parameters,
                    prompt=artifact,
                    input_sha256=canonical_json_hash(sequence),
                    generate=partial(
                        generate_with_validation_repairs,
                        provider=provider,
                        validator=validator,
                    ),
                    validate=validator,
                )
            except (
                OpenAIError,
                ValueError,
                RuntimeError,
                json.JSONDecodeError,
            ) as error:
                output_rows.append(
                    {
                        **base_row,
                        "status": "failed",
                        "cache_hit": False,
                        "response_json": None,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            else:
                output_rows.append(
                    {
                        **base_row,
                        "status": "validated",
                        "cache_hit": cache_hit,
                        "response_json": json.dumps(
                            response,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        "error_type": None,
                        "error": None,
                    }
                )

    results = pd.DataFrame(output_rows)
    expected_rows = len(sample) * len(entries)
    if len(results) != expected_rows or results[
        ["variant_id", "prompt_version"]
    ].duplicated().any():
        raise RuntimeError("Azure result matrix is incomplete or duplicated")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(args.output, index=False)
    failed = int(results["status"].eq("failed").sum())
    if failed:
        raise RuntimeError(
            f"{failed}/{expected_rows} Azure cells failed; failures remain in "
            f"{args.output}"
        )


if __name__ == "__main__":
    main()
