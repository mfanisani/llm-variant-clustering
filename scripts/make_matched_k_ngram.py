"""Write the n-gram TF-IDF partitions at the matched K used by Table 3 (held-out replay).

In the original study these partitions were written by the `comparators` stage of a
separate supplementary study (GPT-5 direct clustering, not part of this release). That
stage only runs after the GPT-5 clustering stage. This helper reproduces the n-gram
partitions with the same library calls and default seeds, so Table 3 can be replicated
without GPT-5. It writes to the path that scripts/run_replay_matched_k.py reads:

    evaluation/supp_llm_direct_clustering_v1/assignments/<log>/assignments_ngram_tfidf_1_3_k<K>.parquet

Run after scripts/run_p7_scalable_benchmark.py (or run_p7_development_comparison.py for
BPI 2020), because the variant set is filtered by full_benchmark/<log>/embedding_index.parquet.

    python scripts/make_matched_k_ngram.py [--datasets bpi2020 bpi2017 bpi2019 sepsis rtfm]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from llm_variant_clustering.clustering import weighted_kmeans
from llm_variant_clustering.features import ngram_tfidf_features
from run_representation_source_benchmark import DATASETS

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "evaluation" / "full_benchmark"
OUT = ROOT / "evaluation" / "supp_llm_direct_clustering_v1" / "assignments"
METHOD = "ngram_tfidf_1_3"


def load_variants(dataset: str, folder: str) -> pd.DataFrame:
    variants = pd.read_parquet(ROOT / "datasets/processed" / folder / "variants.parquet")
    index_path = BENCH / dataset / "embedding_index.parquet"
    if index_path.exists():
        keep = frozenset(pd.read_parquet(index_path)["variant_id"].astype(str))
        variants = variants.loc[variants["variant_id"].astype(str).isin(keep)].copy()
    variants["variant_id"] = variants["variant_id"].astype(str)
    return variants.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="*", default=[entry[0] for entry in DATASETS])
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    for dataset, folder, matched_k, _ in DATASETS:
        if dataset not in args.datasets:
            continue
        variants = load_variants(dataset, folder)
        weights = variants["frequency"].to_numpy(dtype=np.float64)
        matrix = csr_matrix(ngram_tfidf_features(variants).matrix)
        labels = np.asarray(weighted_kmeans(matrix, weights, n_clusters=matched_k).labels, dtype=int)
        frame = pd.DataFrame({
            "variant_id": variants["variant_id"].tolist(),
            "cluster_label": labels,
            "cluster_id": [f"{METHOD}_k{matched_k}_c{label:02d}" for label in labels],
        })
        target = args.output_dir / dataset / f"assignments_{METHOD}_k{matched_k}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
        print(f"{dataset}: K={matched_k}, {len(frame)} variants -> {target}")


if __name__ == "__main__":
    main()
