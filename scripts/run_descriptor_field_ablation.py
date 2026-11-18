"""Which field of the LLM normalization actually carries the partition?

The second-model test produced an unexpected result: an open-weight model that
collapses `action` to a near-constant value and hallucinates `explicit_state`
still reproduces the frozen RTFM partition exactly. That points at the canonical
descriptor being dominated by one field rather than by the inferred semantics.

This script tests that directly. It rebuilds the canonical descriptor from
subsets of the five fields, embeds it with the same model, applies the same
ordered pyramid, and clusters at the same K. Fields are held out one at a time
(leave-one-out) and included one at a time (single field), so both the marginal
contribution and the standalone sufficiency of each field are measured.

If a single-field descriptor reproduces the frozen partition, the LLM's inferred
semantics are not what the clustering responds to.
"""

from __future__ import annotations

import argparse
import json
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
from llm_variant_clustering.semantic_compiler import ordered_semantic_pyramid
from run_representation_source_benchmark import (
    DATASETS,
    _compact_activity_space,
    _embed,
    _frozen_p7_labels,
)

FIELDS = ("object", "action", "stage", "state", "role")


def _parse_descriptor(descriptor: str) -> dict[str, str]:
    parsed = {}
    for chunk in descriptor.split(";"):
        if ":" not in chunk:
            continue
        key, value = chunk.split(":", maxsplit=1)
        parsed[key.strip()] = value.strip()
    return parsed


def _render(fields: dict[str, str], keep: tuple[str, ...]) -> str:
    return "; ".join(f"{name}: {fields.get(name, 'unknown')}" for name in keep)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("models/huggingface"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/descriptor_field_ablation"),
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["bpi2020", "bpi2017", "sepsis", "rtfm"],
    )
    args = parser.parse_args()

    selected = [entry for entry in DATASETS if entry[0] in set(args.datasets)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configurations: list[tuple[str, tuple[str, ...]]] = [("all_fields", FIELDS)]
    configurations += [
        (f"only_{field}", (field,)) for field in FIELDS
    ]
    configurations += [
        (f"drop_{field}", tuple(f for f in FIELDS if f != field))
        for field in FIELDS
    ]

    rows = []
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
        reference = _frozen_p7_labels(benchmark_dir, partition, variant_ids)

        activity_index = pd.read_parquet(
            benchmark_dir / "activity_embedding_index.parquet"
        )
        labels = activity_index["activity_label"].astype(str).tolist()
        parsed = [
            _parse_descriptor(text)
            for text in activity_index["canonical_descriptor"].astype(str)
        ]

        for name, keep_fields in configurations:
            texts = [_render(fields, keep_fields) for fields in parsed]
            distinct = len(set(texts))
            compact, _ = _compact_activity_space(_embed(texts, args.cache_dir))
            matrix = ordered_semantic_pyramid(
                sequences, dict(zip(labels, compact, strict=True))
            )
            features = embedding_feature_set(name, variants, matrix)
            distances = pairwise_distances(features.matrix, metric=features.metric)
            result = weighted_kmeans(features.matrix, weights, n_clusters=matched_k)
            # A field that takes one value across the whole vocabulary maps every
            # variant to the same vector, so no partition exists. That is a
            # substantive result about the field, not an error.
            if len(np.unique(result.labels)) < 2:
                rows.append(
                    {
                        "dataset": dataset,
                        "configuration": name,
                        "fields": "+".join(keep_fields),
                        "distinct_descriptors": distinct,
                        "activity_count": len(labels),
                        "matched_k": matched_k,
                        "case_weighted_silhouette": None,
                        "stability_ari": None,
                        "ari_vs_frozen_p7": 0.0,
                        "nmi_vs_frozen_p7": 0.0,
                        "degenerate": True,
                    }
                )
                print(f"{dataset} {name} degenerate (single cluster)", flush=True)
                continue
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
                    "configuration": name,
                    "fields": "+".join(keep_fields),
                    "distinct_descriptors": distinct,
                    "activity_count": len(labels),
                    "matched_k": matched_k,
                    "case_weighted_silhouette": metrics["case_weighted_silhouette"],
                    "stability_ari": result.mean_seed_stability_ari,
                    "ari_vs_frozen_p7": float(
                        adjusted_rand_score(reference, result.labels)
                    ),
                    "nmi_vs_frozen_p7": float(
                        normalized_mutual_info_score(reference, result.labels)
                    ),
                    "degenerate": False,
                }
            )
            print(f"{dataset} {name} done", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "descriptor_field_ablation.csv", index=False)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "fields": list(FIELDS),
                "configurations": [name for name, _ in configurations],
                "datasets": [entry[0] for entry in selected],
                "note": (
                    "Embedding model, pyramid, weighting, clustering, and K are "
                    "identical to the frozen pipeline. Only the fields present in "
                    "the canonical descriptor change."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    pd.set_option("display.width", 220)
    print(frame.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
