"""Paired bootstrap comparison of activity-vector sources.

The controlled source benchmark reports point estimates. This script attaches
paired percentile intervals to the differences between P7 and each knowledge-free
alternative, using identical resampled variants for both partitions so that the
comparison is not confounded by which variants happen to be drawn.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from llm_variant_clustering.clustering import weighted_kmeans
from llm_variant_clustering.features import (
    activity_context_embeddings,
    parse_sequences,
)
from llm_variant_clustering.semantic_compiler import ordered_semantic_pyramid
from llm_variant_clustering.uncertainty import (
    bootstrap_silhouette,
    paired_silhouette_difference,
)
from run_representation_source_benchmark import (
    DATASETS,
    MODEL_ID,
    MODEL_REVISION,
    _compact_activity_space,
    _embed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("models/huggingface"))
    parser.add_argument("--datasets", nargs="*", default=["bpi2020", "sepsis", "rtfm"])
    parser.add_argument("--resamples", type=int, default=400)
    parser.add_argument("--max-bootstrap-variants", type=int, default=2500)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/representation_sources"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = [entry for entry in DATASETS if entry[0] in set(args.datasets)]
    interval_rows = []
    paired_rows = []

    for dataset, folder, matched_k, _ in selected:
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
        sample_size = (
            args.max_bootstrap_variants
            if len(variants) > args.max_bootstrap_variants
            else None
        )

        activity_index = pd.read_parquet(
            benchmark_dir / "activity_embedding_index.parquet"
        )
        labels = activity_index["activity_label"].astype(str).tolist()
        descriptors = activity_index["canonical_descriptor"].astype(str).tolist()
        semantic_compact, semantic_rank = _compact_activity_space(
            _embed(descriptors, args.cache_dir)
        )
        raw_compact, _ = _compact_activity_space(_embed(labels, args.cache_dir))

        sources = {
            "p7_llm_semantic_bge": dict(zip(labels, semantic_compact, strict=True)),
            "raw_activity_label_bge": dict(zip(labels, raw_compact, strict=True)),
            "activity2vec_log_context_d16": activity_context_embeddings(
                variants, embedding_dim=16
            ),
            f"activity2vec_log_context_d{semantic_rank}": (
                activity_context_embeddings(variants, embedding_dim=semantic_rank)
            ),
        }

        computed = {}
        for name, activity_map in sources.items():
            matrix = ordered_semantic_pyramid(sequences, activity_map)
            distances = pairwise_distances(matrix, metric="cosine")
            result = weighted_kmeans(matrix, weights, n_clusters=matched_k)
            computed[name] = (distances, result.labels)
            interval = bootstrap_silhouette(
                distances,
                result.labels,
                weights,
                resamples=args.resamples,
                sample_size=sample_size,
            )
            interval_rows.append(
                {
                    "dataset": dataset,
                    "method": name,
                    "k": matched_k,
                    "bootstrap_sample_size": sample_size or len(variants),
                    **interval.as_dict("silhouette_"),
                }
            )
            print(f"{dataset} {name} interval done", flush=True)

        reference = "p7_llm_semantic_bge"
        for name in computed:
            if name == reference:
                continue
            difference = paired_silhouette_difference(
                *computed[reference],
                *computed[name],
                weights,
                resamples=args.resamples,
                sample_size=sample_size,
            )
            paired_rows.append(
                {
                    "dataset": dataset,
                    "first": reference,
                    "second": name,
                    "k": matched_k,
                    **difference.as_dict("difference_"),
                }
            )
            print(f"{dataset} {reference} vs {name} done", flush=True)

    pd.DataFrame(interval_rows).to_csv(
        args.output_dir / "source_silhouette_intervals.csv", index=False
    )
    pd.DataFrame(paired_rows).to_csv(
        args.output_dir / "source_paired_differences.csv", index=False
    )
    (args.output_dir / "uncertainty_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "ancillary_after_primary_artifact_lock",
                "embedding_model": MODEL_ID,
                "embedding_revision": MODEL_REVISION,
                "resamples": args.resamples,
                "confidence": 0.95,
                "datasets": [dataset for dataset, *_ in selected],
                "note": (
                    "Paired resamples use identical variant draws for both "
                    "partitions, isolating the representation difference."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output_dir / "source_paired_differences.csv")


if __name__ == "__main__":
    main()
