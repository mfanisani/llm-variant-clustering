from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import pandas as pd
from openai import OpenAIError

from llm_variant_clustering.azure_openai_provider import AzureOpenAIToolProvider
from llm_variant_clustering.features import parse_sequences
from llm_variant_clustering.llm_cache import (
    GenerationParameters,
    ModelIdentity,
    StructuredResponseCache,
    canonical_json_hash,
    execute_cached_request,
)
from llm_variant_clustering.prompts import render_prompt
from llm_variant_clustering.semantic_compiler import validate_activity_semantics
from run_azure_prompt_pilot import generate_with_validation_repairs

ROOT = Path(__file__).parents[1]
PROMPT = ROOT / "configs/prompts/p7_activity_semantics.txt"
SCHEMA = ROOT / "configs/schemas/activity_semantics_p7.schema.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variants", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--api-version", default="2025-01-01-preview")
    parser.add_argument("--prompt", type=Path, default=PROMPT)
    parser.add_argument("--schema", type=Path, default=SCHEMA)
    args = parser.parse_args()

    variants = pd.read_parquet(args.variants)
    activities = sorted(
        {activity for sequence in parse_sequences(variants) for activity in sequence}
    )
    parameters = GenerationParameters(
        temperature=0.0,
        seed=None,
        max_output_tokens=1000,
    )
    identity = ModelIdentity(
        provider="azure_openai_entra",
        model=args.deployment,
        model_version=args.model_version,
    )
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    cache = StructuredResponseCache(args.cache_dir)
    rows = []
    for activity in activities:
        artifact = render_prompt(
            args.prompt,
            args.schema,
            {"ACTIVITY_LABEL": activity},
            prompt_version="p7_activity",
        )
        validator = partial(
            validate_activity_semantics,
            schema_path=args.schema,
            expected_activity_label=activity,
        )
        provider = AzureOpenAIToolProvider(
            endpoint=args.endpoint,
            deployment=args.deployment,
            api_version=args.api_version,
            parameters=parameters,
            schema=schema,
            schema_name="activity_semantics_p7",
        )
        try:
            response, cache_hit = execute_cached_request(
                cache=cache,
                model=identity,
                parameters=parameters,
                prompt=artifact,
                input_sha256=canonical_json_hash(activity),
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
            rows.append(
                {
                    "activity_label": activity,
                    "status": "failed",
                    "cache_hit": False,
                    "response_json": None,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        else:
            rows.append(
                {
                    "activity_label": activity,
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
    results = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(args.output, index=False)
    failures = int(results["status"].eq("failed").sum())
    if failures:
        raise RuntimeError(
            f"{failures}/{len(results)} activity normalizations failed; "
            f"failures remain in {args.output}"
        )


if __name__ == "__main__":
    main()
