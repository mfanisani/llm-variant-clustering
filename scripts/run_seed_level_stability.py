"""Seed-level uncertainty for reseeding stability (v2 correction).

v1 (run_reseeding_stability.py) bootstrapped the 66 pairwise ARIs of 12 seeds
as if they were independent, although every seed's labeling enters 11 pairs.
This script recomputes the identical labelings, stores the full seed-by-seed
ARI matrix, and resamples seeds instead (U-statistic bootstrap). It also gives
paired P7-minus-comparator differences under the same seed draws.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from llm_variant_clustering.features import activity_context_embeddings, parse_sequences
from llm_variant_clustering.semantic_compiler import ordered_semantic_pyramid
from llm_variant_clustering.uncertainty import bootstrap_seed_agreement
from run_representation_source_benchmark import (
    DATASETS,
    MODEL_ID,
    MODEL_REVISION,
    _compact_activity_space,
    _embed,
)
from run_reseeding_stability import SEEDS

FIRST = "p7_llm_semantic_bge"


def _ari_matrix(labelings: list[np.ndarray]) -> np.ndarray:
    size = len(labelings)
    matrix = np.ones((size, size))
    for left in range(size):
        for right in range(left + 1, size):
            matrix[left, right] = matrix[right, left] = adjusted_rand_score(
                labelings[left], labelings[right]
            )
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("models/huggingface"))
    parser.add_argument(
        "--datasets", nargs="*", default=["bpi2020", "bpi2017", "sepsis", "rtfm"]
    )
    parser.add_argument("--resamples", type=int, default=4000)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument(
        "--v1-csv",
        type=Path,
        default=None,
        help="v1 reseeding_stability.csv used to verify point estimates",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = [entry for entry in DATASETS if entry[0] in set(args.datasets)]
    rows, paired_rows, matrices = [], [], {}
    for dataset, folder, matched_k, _ in selected:
        k = args.k if args.k is not None else matched_k
        benchmark_dir = Path("evaluation/full_benchmark") / dataset
        variants = pd.read_parquet(Path("datasets/processed") / folder / "variants.parquet")
        index_path = benchmark_dir / "embedding_index.parquet"
        if index_path.exists():
            keep = frozenset(pd.read_parquet(index_path)["variant_id"].astype(str))
            variants = variants.loc[variants["variant_id"].astype(str).isin(keep)].copy()
        variants = variants.reset_index(drop=True)
        sequences = parse_sequences(variants)
        weights = variants["frequency"].to_numpy(dtype=np.float64)
        activity_index = pd.read_parquet(benchmark_dir / "activity_embedding_index.parquet")
        labels = activity_index["activity_label"].astype(str).tolist()
        descriptors = activity_index["canonical_descriptor"].astype(str).tolist()
        semantic_compact, semantic_rank = _compact_activity_space(
            _embed(descriptors, args.cache_dir)
        )
        raw_compact, _ = _compact_activity_space(_embed(labels, args.cache_dir))
        sources = {
            FIRST: dict(zip(labels, semantic_compact, strict=True)),
            "raw_activity_label_bge": dict(zip(labels, raw_compact, strict=True)),
            "activity2vec_log_context_d16": activity_context_embeddings(
                variants, embedding_dim=16
            ),
            f"activity2vec_log_context_d{semantic_rank}": activity_context_embeddings(
                variants, embedding_dim=semantic_rank
            ),
        }
        per_method = {}
        for name, activity_map in sources.items():
            matrix_input = ordered_semantic_pyramid(sequences, activity_map)
            labelings = [
                KMeans(n_clusters=k, random_state=seed, n_init=10, algorithm="lloyd")
                .fit_predict(matrix_input, sample_weight=weights)
                for seed in SEEDS
            ]
            agreement = _ari_matrix(labelings)
            per_method[name] = agreement
            matrices[f"{dataset}__{name}"] = agreement
            interval = bootstrap_seed_agreement(agreement, resamples=args.resamples)
            rows.append(
                {
                    "dataset": dataset,
                    "method": name,
                    "k": k,
                    "seed_count": len(SEEDS),
                    "bootstrap_unit": "seed",
                    **interval.as_dict("stability_"),
                }
            )
            print(f"{dataset} {name} {interval.point_estimate:.3f} "
                  f"[{interval.lower:.3f}, {interval.upper:.3f}]", flush=True)
        for name, agreement in per_method.items():
            if name == FIRST:
                continue
            difference = bootstrap_seed_agreement(
                per_method[FIRST], second=agreement, resamples=args.resamples
            )
            paired_rows.append(
                {"dataset": dataset, "first": FIRST, "second": name, "k": k,
                 **difference.as_dict("difference_")}
            )

    frame = pd.DataFrame(rows)
    if args.v1_csv is not None and args.v1_csv.exists():
        v1 = pd.read_csv(args.v1_csv)[["dataset", "method", "stability_point_estimate",
                                        "stability_ci_lower", "stability_ci_upper"]]
        frame = frame.merge(v1, on=["dataset", "method"], how="left", suffixes=("", "_v1"))
        gap = (frame["stability_point_estimate"] - frame["stability_point_estimate_v1"]).abs()
        frame["reproduces_v1_point"] = gap < 1e-9
    frame.to_csv(args.output_dir / "reseeding_stability_seed_level.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(
        args.output_dir / "paired_stability_differences_seed_level.csv", index=False
    )
    np.savez_compressed(args.output_dir / "ari_matrices.npz", **matrices)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "ancillary_correction",
                "corrects": "pairwise-agreement bootstrap in run_reseeding_stability.py",
                "embedding_model": MODEL_ID,
                "embedding_revision": MODEL_REVISION,
                "seeds": list(SEEDS),
                "k_override": args.k,
                "bootstrap_unit": "seed (U-statistic over distinct-seed pairs)",
                "resamples": args.resamples,
                "confidence": 0.95,
                "datasets": [entry[0] for entry in selected],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
