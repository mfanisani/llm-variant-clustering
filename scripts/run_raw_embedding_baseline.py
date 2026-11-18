from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from llm_variant_clustering.clustering import weighted_kmeans
from llm_variant_clustering.embeddings import (
    embed_raw_sequences,
    embedding_feature_set,
    save_embedding_artifacts,
)
from llm_variant_clustering.evaluation import (
    evaluate_partition,
    weighted_medoid_indices,
)

MODEL_ID = "BAAI/bge-large-en-v1.5"
MODEL_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
MODEL_LICENSE = "MIT"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen raw-sequence BGE clustering baseline."
    )
    parser.add_argument("variants", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("models/huggingface"))
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    variants = pd.read_parquet(args.variants)
    embeddings, provenance = embed_raw_sequences(
        variants,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        model_license=MODEL_LICENSE,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        device="cpu",
    )
    save_embedding_artifacts(
        args.output_dir,
        variants,
        embeddings,
        provenance,
    )

    features = embedding_feature_set("raw_bge_large_en_v1_5", variants, embeddings)
    weights = variants["frequency"].to_numpy(dtype=np.float64)
    result = weighted_kmeans(
        features.matrix,
        weights,
        n_clusters=args.clusters,
    )
    metrics = evaluate_partition(
        features.matrix,
        result.labels,
        weights,
        metric=features.metric,
    )
    medoids = weighted_medoid_indices(
        features.matrix,
        result.labels,
        weights,
        metric=features.metric,
    )
    assignments = pd.DataFrame(
        {
            "variant_id": features.variant_ids,
            "cluster_id": [
                f"raw_bge_k{args.clusters}_c{label:02d}"
                for label in result.labels
            ],
            "is_medoid": [
                index == medoids[int(label)]
                for index, label in enumerate(result.labels)
            ],
        }
    )
    assignments.to_parquet(
        args.output_dir / f"assignments_raw_bge_k{args.clusters}.parquet",
        index=False,
    )
    result_record = {
        "method": features.name,
        "feature_count": features.matrix.shape[1],
        "selected_seed": result.selected_seed,
        "weighted_inertia": result.weighted_inertia,
        "mean_seed_stability_ari": result.mean_seed_stability_ari,
        **metrics,
    }
    (args.output_dir / f"metrics_k{args.clusters}.json").write_text(
        json.dumps(result_record, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

