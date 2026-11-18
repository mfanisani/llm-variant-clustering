from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import pandas as pd
from openai import OpenAIError

from llm_variant_clustering.llm_cache import (
    GenerationParameters,
    ModelIdentity,
    StructuredResponseCache,
    canonical_json_hash,
    execute_cached_request,
)
from llm_variant_clustering.openai_provider import OpenAIResponsesProvider
from llm_variant_clustering.prompts import render_prompt, validate_response

ROOT = Path(__file__).parents[1]
PROMPTS = {
    "p1": (
        ROOT / "configs/prompts/p1_minimal_signature.txt",
        ROOT / "configs/schemas/variant_signature_p1.schema.json",
    ),
    "p2": (
        ROOT / "configs/prompts/p2_evidence_grounded_signature.txt",
        ROOT / "configs/schemas/variant_signature_p2.schema.json",
    ),
    "p3": (
        ROOT / "configs/prompts/p3_structural_semantic_signature.txt",
        ROOT / "configs/schemas/variant_signature_p3.schema.json",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cached OpenAI structured signatures on a pilot sample."
    )
    parser.add_argument("sample", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument(
        "--prompt-versions",
        nargs="+",
        choices=tuple(PROMPTS),
        default=["p1", "p2", "p3"],
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=4000)
    args = parser.parse_args()

    sample = pd.read_parquet(args.sample)
    parameters = GenerationParameters(
        temperature=args.temperature,
        seed=None,
        max_output_tokens=args.max_output_tokens,
    )
    model = ModelIdentity(
        provider="openai",
        model=args.model,
        model_version=args.model_version,
    )
    cache = StructuredResponseCache(args.cache_dir)

    output_rows = []
    for prompt_version in args.prompt_versions:
        template_path, schema_path = PROMPTS[prompt_version]
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        provider = OpenAIResponsesProvider(
            model=args.model,
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
                    model=model,
                    parameters=parameters,
                    prompt=artifact,
                    input_sha256=canonical_json_hash(sequence),
                    generate=provider.generate,
                    validate=validator,
                )
            except (OpenAIError, ValueError, RuntimeError, json.JSONDecodeError) as error:
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
                            response, ensure_ascii=False, separators=(",", ":")
                        ),
                        "error_type": None,
                        "error": None,
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(output_rows)
    expected_rows = len(sample) * len(args.prompt_versions)
    if len(results) != expected_rows or results[
        ["variant_id", "prompt_version"]
    ].duplicated().any():
        raise RuntimeError("Pilot result matrix is incomplete or duplicated")
    results.to_parquet(args.output, index=False)
    failed = int(results["status"].eq("failed").sum())
    if failed:
        raise RuntimeError(
            f"{failed}/{expected_rows} pilot cells failed; failures remain in "
            f"{args.output}"
        )


if __name__ == "__main__":
    main()
