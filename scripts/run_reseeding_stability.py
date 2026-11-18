"""Reseeding stability of each activity-vector source, with uncertainty.

The headline stability figure averages ten pairwise ARIs from five seeds, which
is too few to report as a bare number. This script widens the seed set and
attaches a bootstrap interval over the pairwise agreements, so the claim that
LLM-normalized activity vectors yield more reproducible partitions can be stated
with a quantified margin rather than asserted from a single average.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from llm_variant_clustering.features import (
    activity_context_embeddings,
    parse_sequences,
)
from llm_variant_clustering.semantic_compiler import ordered_semantic_pyramid
from llm_variant_clustering.uncertainty import bootstrap_cluster_metric
from run_representation_source_benchmark import (
    DATASETS,
    MODEL_ID,
    MODEL_REVISION,
    _compact_activity_space,
    _embed,
)

SEEDS = (17, 29, 43, 59, 71, 83, 97, 101, 113, 127, 139, 151)


def _pairwise_agreements(labelings: list[np.ndarray]) -> np.ndarray:
    return np.array(
        [
            adjusted_rand_score(labelings[left], labelings[right])
            for left in range(len(labelings))
            for right in range(left + 1, len(labelings))
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("models/huggingface"))
    parser.add_argument("--datasets", nargs="*", default=["bpi2020", "sepsis", "rtfm"])
    parser.add_argument("--resamples", type=int, default=4000)
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help=(
            "Override the per-dataset K. Seed stability saturates at K=2, so a "
            "common larger K is required to compare logs on equal footing."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/representation_sources"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = [entry for entry in DATASETS if entry[0] in set(args.datasets)]
    rows = []

    for dataset, folder, matched_k, _ in selected:
        if args.k is not None:
            matched_k = args.k
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

        for name, activity_map in sources.items():
            matrix = ordered_semantic_pyramid(sequences, activity_map)
            labelings = [
                KMeans(
                    n_clusters=matched_k,
                    random_state=seed,
                    n_init=10,
                    algorithm="lloyd",
                ).fit_predict(matrix, sample_weight=weights)
                for seed in SEEDS
            ]
            agreements = _pairwise_agreements(labelings)
            interval = bootstrap_cluster_metric(
                agreements,
                resamples=args.resamples,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "method": name,
                    "k": matched_k,
                    "seed_count": len(SEEDS),
                    "pair_count": int(agreements.size),
                    "min_pairwise_ari": float(agreements.min()),
                    "max_pairwise_ari": float(agreements.max()),
                    "exact_reproduction_rate": float(np.mean(agreements > 0.999)),
                    **interval.as_dict("stability_"),
                }
            )
            print(
                f"{dataset} {name} stability "
                f"{interval.point_estimate:.3f} "
                f"[{interval.lower:.3f}, {interval.upper:.3f}]",
                flush=True,
            )

    pd.DataFrame(rows).to_csv(
        args.output_dir / "reseeding_stability.csv", index=False
    )
    (args.output_dir / "stability_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "ancillary_after_primary_artifact_lock",
                "embedding_model": MODEL_ID,
                "embedding_revision": MODEL_REVISION,
                "seeds": list(SEEDS),
                "k_override": args.k,
                "bootstrap_unit": "pairwise seed agreement",
                "resamples": args.resamples,
                "confidence": 0.95,
                "datasets": [dataset for dataset, *_ in selected],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output_dir / "reseeding_stability.csv")


if __name__ == "__main__":
    main()
