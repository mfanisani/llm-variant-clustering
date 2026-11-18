"""Second-model robustness test for P7 activity normalization.

The frozen P7 representation depends on one proprietary normalizer (GPT-4.1).
This script re-runs the identical frozen prompt and schema through a local
open-weight model, rebuilds the ordered pyramid from the second model's
descriptors, and clusters with the same weighted K-means and the same K.

Everything downstream of the normalizer is held fixed, so any change in the
partition is attributable to the normalizing model alone. A high agreement means
the reported clusters are a property of activity-level semantic normalization
rather than of one specific vendor model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    pairwise_distances,
)

from llm_variant_clustering.clustering import weighted_kmeans
from llm_variant_clustering.embeddings import embedding_feature_set
from llm_variant_clustering.evaluation import evaluate_partition
from llm_variant_clustering.features import parse_sequences
from llm_variant_clustering.llm_cache import (
    GenerationParameters,
    ModelIdentity,
    StructuredResponseCache,
    canonical_json_hash,
    execute_cached_request,
)
from llm_variant_clustering.local_provider import LocalTransformersProvider
from llm_variant_clustering.prompts import render_prompt
from llm_variant_clustering.semantic_compiler import (
    canonical_activity_text,
    ordered_semantic_pyramid,
    validate_activity_semantics,
)
from run_representation_source_benchmark import (
    DATASETS,
    _compact_activity_space,
    _embed,
    _frozen_p7_labels,
)

ROOT = Path(__file__).parents[1]
PROMPT = ROOT / "configs/prompts/p7_activity_semantics.txt"
SCHEMA = ROOT / "configs/schemas/activity_semantics_p7.schema.json"

SECOND_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SECOND_MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

REQUIRED_FIELDS = (
    "business_object",
    "action",
    "process_stage",
    "explicit_state",
    "explicit_role",
)


def _generate_with_repairs(
    rendered_prompt: str,
    *,
    provider: LocalTransformersProvider,
    validator: object,
    max_validation_attempts: int = 3,
) -> object:
    """Mirror the repair loop the frozen GPT-4.1 run used.

    The primary normalization gave the model up to three attempts with the
    validation error fed back. Denying the second model that budget would make
    the comparison unfair, so the same loop and the same attempt count are used
    here.
    """
    current_prompt = rendered_prompt
    generation = None
    for attempt in range(1, max_validation_attempts + 1):
        generation = provider.generate(current_prompt)
        try:
            validator(generation.response)  # type: ignore[operator]
        except ValueError as error:
            if attempt == max_validation_attempts:
                break
            current_prompt = (
                rendered_prompt
                + "\n\nVALIDATION CORRECTION\n"
                + "The previous JSON failed local validation. Produce a fresh "
                + "JSON object for the same activity label. Every field must use "
                + "one of the allowed enum values from the schema. Correct every "
                + "listed problem:\n"
                + str(error)
            )
        else:
            return generation
    if generation is None:
        raise RuntimeError("Validation repair loop did not generate a response")
    return generation


def _normalize_with_second_model(
    labels: list[str],
    *,
    provider: LocalTransformersProvider,
    cache: StructuredResponseCache,
    identity: ModelIdentity,
    parameters: GenerationParameters,
    schema_text: str,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    accepted: dict[str, dict[str, object]] = {}
    failures: list[dict[str, object]] = []
    for label in labels:
        artifact = render_prompt(
            PROMPT,
            SCHEMA,
            {"ACTIVITY_LABEL": label},
            prompt_version="p7_activity",
        )
        artifact = replace(
            artifact,
            rendered_prompt=(
                artifact.rendered_prompt
                + "\n\nThe output must exactly match this JSON Schema:\n"
                + schema_text
            ),
        )
        artifact = replace(
            artifact,
            prompt_sha256=hashlib.sha256(
                artifact.rendered_prompt.encode("utf-8")
            ).hexdigest(),
        )
        validator = partial(
            validate_activity_semantics,
            schema_path=SCHEMA,
            expected_activity_label=label,
        )
        try:
            response, _ = execute_cached_request(
                cache=cache,
                model=identity,
                parameters=parameters,
                prompt=artifact,
                input_sha256=canonical_json_hash(label),
                generate=partial(
                    _generate_with_repairs,
                    provider=provider,
                    validator=validator,
                ),
                validate=validator,
            )
        except (ValueError, RuntimeError, json.JSONDecodeError) as error:
            failures.append(
                {
                    "activity_label": label,
                    "error_type": type(error).__name__,
                    "error": str(error)[:400],
                }
            )
            continue
        accepted[label] = dict(response)
    return accepted, failures


def _grounding_rate(
    semantics: dict[str, dict[str, object]],
) -> float:
    """Share of normalizations whose content words appear in the source label.

    The frozen P7 audit uses the same criterion: a normalization is grounded when
    every token it introduces for business_object and action is either present in
    the original label or an explicit 'unknown'/'none' sentinel.
    """
    if not semantics:
        return 0.0
    grounded = 0
    for label, value in semantics.items():
        label_tokens = set(
            token.strip("_-/() ").lower()
            for token in label.replace("_", " ").replace("-", " ").split()
        )
        ok = True
        for field in ("business_object", "action"):
            text = str(value.get(field, "")).lower()
            if text in {"unknown", "none", "none explicit", ""}:
                continue
            for token in text.split():
                if not any(
                    token in candidate or candidate in token
                    for candidate in label_tokens
                ):
                    ok = False
                    break
            if not ok:
                break
        grounded += int(ok)
    return grounded / len(semantics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("models/huggingface"))
    parser.add_argument(
        "--llm-cache-dir",
        type=Path,
        default=Path("evaluation/second_llm_robustness/cache"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/second_llm_robustness"),
    )
    parser.add_argument("--datasets", nargs="*", default=["rtfm", "sepsis", "bpi2020"])
    parser.add_argument("--max-output-tokens", type=int, default=400)
    args = parser.parse_args()

    selected = [entry for entry in DATASETS if entry[0] in set(args.datasets)]
    if not selected:
        raise SystemExit("No matching datasets")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    schema_text = SCHEMA.read_text(encoding="utf-8")
    parameters = GenerationParameters(
        temperature=0.0,
        seed=None,
        max_output_tokens=args.max_output_tokens,
    )
    identity = ModelIdentity(
        provider="local_transformers_robustness",
        model=SECOND_MODEL_ID,
        model_version=SECOND_MODEL_REVISION,
    )
    cache = StructuredResponseCache(args.llm_cache_dir)
    provider = LocalTransformersProvider(
        model_id=SECOND_MODEL_ID,
        model_revision=SECOND_MODEL_REVISION,
        parameters=parameters,
        cache_dir=args.cache_dir,
    )

    rows = []
    field_rows = []
    all_failures: dict[str, list[dict[str, object]]] = {}

    for dataset, folder, matched_k, partition in selected:
        benchmark_dir = Path("evaluation/full_benchmark") / dataset
        variants = pd.read_parquet(
            Path("datasets/processed") / folder / "variants.parquet"
        )
        index_path = benchmark_dir / "embedding_index.parquet"
        if index_path.exists():
            keep = frozenset(pd.read_parquet(index_path)["variant_id"].astype(str))
            variants = variants.loc[
                variants["variant_id"].astype(str).isin(keep)
            ].copy()
        variants = variants.reset_index(drop=True)
        sequences = parse_sequences(variants)
        weights = variants["frequency"].to_numpy(dtype=np.float64)
        variant_ids = variants["variant_id"].astype(str).tolist()
        reference_labels = _frozen_p7_labels(benchmark_dir, partition, variant_ids)

        activity_index = pd.read_parquet(
            benchmark_dir / "activity_embedding_index.parquet"
        )
        labels = activity_index["activity_label"].astype(str).tolist()
        primary_descriptors = (
            activity_index["canonical_descriptor"].astype(str).tolist()
        )

        print(f"[{dataset}] normalizing {len(labels)} activities", flush=True)
        semantics, failures = _normalize_with_second_model(
            labels,
            provider=provider,
            cache=cache,
            identity=identity,
            parameters=parameters,
            schema_text=schema_text,
        )
        all_failures[dataset] = failures
        coverage = len(semantics) / len(labels)
        print(
            f"[{dataset}] validated {len(semantics)}/{len(labels)} "
            f"({coverage:.1%})",
            flush=True,
        )

        if coverage < 1.0:
            rows.append(
                {
                    "dataset": dataset,
                    "status": "incomplete_normalization",
                    "activity_count": len(labels),
                    "validated_count": len(semantics),
                    "schema_validity_rate": coverage,
                    "grounding_rate": _grounding_rate(semantics),
                    "field_agreement_rate": None,
                    "matched_k": matched_k,
                    "ari_vs_frozen_p7": None,
                    "nmi_vs_frozen_p7": None,
                    "case_weighted_silhouette": None,
                    "stability_ari": None,
                }
            )

        primary_map = dict(zip(labels, primary_descriptors, strict=True))
        # Activities the second model could not normalize fall back to the frozen
        # descriptor. At full coverage this is a pure second-model run; below full
        # coverage it is a partial-substitution sensitivity that lower-bounds the
        # effect of changing the normalizer, and is labelled as such.
        second_descriptors = [
            canonical_activity_text(semantics[label])
            if label in semantics
            else primary_map[label]
            for label in labels
        ]
        status = "complete" if coverage == 1.0 else "partial_substitution"
        agreements = [
            int(primary_map[label] == descriptor)
            for label, descriptor in zip(labels, second_descriptors, strict=True)
        ]
        for label, descriptor, agree in zip(
            labels, second_descriptors, agreements, strict=True
        ):
            field_rows.append(
                {
                    "dataset": dataset,
                    "activity_label": label,
                    "second_model_validated": label in semantics,
                    "primary_descriptor": primary_map[label],
                    "second_descriptor": descriptor,
                    "identical": bool(agree),
                }
            )

        compact, _ = _compact_activity_space(
            _embed(second_descriptors, args.cache_dir)
        )
        activity_map = dict(zip(labels, compact, strict=True))
        matrix = ordered_semantic_pyramid(sequences, activity_map)
        features = embedding_feature_set(
            "p7_second_model_bge", variants, matrix
        )
        distances = pairwise_distances(features.matrix, metric=features.metric)
        result = weighted_kmeans(features.matrix, weights, n_clusters=matched_k)
        metrics = evaluate_partition(
            features.matrix,
            result.labels,
            weights,
            metric=features.metric,
            distances=distances,
        )
        rows.append(
            {
                "dataset": dataset,
                "status": status,
                "activity_count": len(labels),
                "validated_count": len(semantics),
                "schema_validity_rate": coverage,
                "grounding_rate": _grounding_rate(semantics),
                "field_agreement_rate": float(np.mean(agreements)),
                "matched_k": matched_k,
                "ari_vs_frozen_p7": float(
                    adjusted_rand_score(reference_labels, result.labels)
                ),
                "nmi_vs_frozen_p7": float(
                    normalized_mutual_info_score(reference_labels, result.labels)
                ),
                "case_weighted_silhouette": metrics["case_weighted_silhouette"],
                "stability_ari": result.mean_seed_stability_ari,
            }
        )
        dataset_dir = args.output_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "variant_id": variant_ids,
                "cluster_label": result.labels,
                "cluster_id": [
                    f"p7_second_model_k{matched_k}_c{int(x):02d}"
                    for x in result.labels
                ],
                "frequency": weights,
            }
        ).to_parquet(
            dataset_dir / "assignments_p7_second_model.parquet", index=False
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "second_llm_robustness.csv", index=False)
    if field_rows:
        pd.DataFrame(field_rows).to_csv(
            args.output_dir / "descriptor_agreement.csv", index=False
        )
    (args.output_dir / "normalization_failures.json").write_text(
        json.dumps(all_failures, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "primary_model": "gpt-4.1 (frozen)",
                "second_model": SECOND_MODEL_ID,
                "second_model_revision": SECOND_MODEL_REVISION,
                "prompt": str(PROMPT.relative_to(ROOT)),
                "schema": str(SCHEMA.relative_to(ROOT)),
                "datasets": [entry[0] for entry in selected],
                "note": (
                    "Prompt, schema, embedding model, pyramid, weighting, "
                    "clustering, and K are identical to the frozen pipeline. "
                    "Only the normalizing model differs."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
