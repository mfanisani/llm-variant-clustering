"""Controlled comparison of activity-vector sources and pyramid depth.

Pooling, clustering, weighting, and automatic-K selection are held identical.
Only the source of the per-activity vector changes, which isolates the
contribution of LLM activity normalization from the embedding model, the
temporal composer, and the clustering algorithm.
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
from llm_variant_clustering.dynamic_clustering import (
    NoEligibleKError,
    select_dynamic_k,
)
from llm_variant_clustering.embeddings import embedding_feature_set
from llm_variant_clustering.evaluation import evaluate_partition
from llm_variant_clustering.features import (
    activity_context_embeddings,
    parse_sequences,
)
from llm_variant_clustering.semantic_compiler import ordered_semantic_pyramid
from run_p7_development_comparison import _load_dynamic_config

MODEL_ID = "BAAI/bge-large-en-v1.5"
MODEL_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"

DATASETS = (
    ("bpi2020", "bpi2020_international_declarations", 9, "dynamic"),
    ("bpi2017", "bpi2017", 2, "dynamic"),
    ("bpi2019", "bpi2019", 6, "dynamic"),
    ("sepsis", "sepsis", 8, "k8"),
    ("rtfm", "rtfm", 2, "dynamic"),
)


def _compact_activity_space(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    _, singular_values, right = np.linalg.svd(
        matrix.astype(np.float64),
        full_matrices=False,
    )
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    compact = (matrix @ right[:rank].T).astype(np.float32)
    if not np.allclose(matrix @ matrix.T, compact @ compact.T, atol=2e-5):
        raise ValueError("Activity-space projection did not preserve geometry")
    return compact, rank


def _embed(texts: list[str], cache_dir: Path) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_folder=str(cache_dir),
        device="cpu",
        trust_remote_code=False,
    )
    tokens = model.tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    if max(len(item) for item in tokens["input_ids"]) > model.max_seq_length:
        raise ValueError("An activity text would require truncation")
    return model.encode(
        texts,
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def _frozen_p7_labels(
    benchmark_dir: Path,
    partition: str,
    variant_ids: list[str],
) -> np.ndarray:
    name = (
        "assignments_p7_ordered_pyramid_1_2_4_bge_dynamic.parquet"
        if partition == "dynamic"
        else "assignments_p7_ordered_pyramid_1_2_4_bge_k8.parquet"
    )
    frame = pd.read_parquet(benchmark_dir / name)
    lookup = frame.set_index(frame["variant_id"].astype(str))["cluster_label"]
    return lookup.loc[variant_ids].to_numpy()


def _evaluate(
    name: str,
    matrix: np.ndarray,
    variants: pd.DataFrame,
    weights: np.ndarray,
    matched_k: int,
    config: object,
    reference_labels: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    features = embedding_feature_set(name, variants, matrix)
    distances = pairwise_distances(features.matrix, metric=features.metric)
    matched = weighted_kmeans(features.matrix, weights, n_clusters=matched_k)
    metrics = evaluate_partition(
        features.matrix,
        matched.labels,
        weights,
        metric=features.metric,
        distances=distances,
    )

    try:
        selection = select_dynamic_k(features, weights, config, distances=distances)
    except NoEligibleKError:
        automatic_k: int | None = None
        automatic_stability: float | None = None
        automatic_silhouette: float | None = None
        automatic_status = "no_eligible_k"
    else:
        selected = selection.candidates.loc[selection.candidates["selected"]].iloc[0]
        automatic_k = int(selection.selected_k)
        automatic_stability = float(selected["mean_seed_stability_ari"])
        automatic_silhouette = float(selected["case_weighted_silhouette"])
        automatic_status = "selected"

    return {
        "method": name,
        "feature_count": int(matrix.shape[1]),
        "matched_k": matched_k,
        "matched_stability_ari": matched.mean_seed_stability_ari,
        "matched_case_weighted_silhouette": metrics["case_weighted_silhouette"],
        "matched_variant_silhouette": metrics["variant_silhouette"],
        "matched_case_entropy": metrics["case_cluster_entropy"],
        "ari_vs_frozen_p7": float(
            adjusted_rand_score(reference_labels, matched.labels)
        ),
        "nmi_vs_frozen_p7": float(
            normalized_mutual_info_score(reference_labels, matched.labels)
        ),
        "automatic_status": automatic_status,
        "automatic_k": automatic_k,
        "automatic_stability_ari": automatic_stability,
        "automatic_case_weighted_silhouette": automatic_silhouette,
    }, matched.labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("models/huggingface"))
    parser.add_argument(
        "--dynamic-config",
        type=Path,
        default=Path("configs/clustering/dynamic_k_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/representation_sources"),
    )
    parser.add_argument("--context-window", type=int, default=2)
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    selected_datasets = [
        entry
        for entry in DATASETS
        if args.datasets is None or entry[0] in set(args.datasets)
    ]
    if not selected_datasets:
        raise SystemExit("No matching datasets")

    config, config_record = _load_dynamic_config(args.dynamic_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = []
    depth_rows = []

    for dataset, folder, matched_k, partition in selected_datasets:
        benchmark_dir = Path("evaluation/full_benchmark") / dataset
        variants = pd.read_parquet(
            Path("datasets/processed") / folder / "variants.parquet"
        )
        index_path = benchmark_dir / "embedding_index.parquet"
        if index_path.exists():
            selected = frozenset(
                pd.read_parquet(index_path)["variant_id"].astype(str)
            )
            variants = variants.loc[
                variants["variant_id"].astype(str).isin(selected)
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
        descriptors = activity_index["canonical_descriptor"].astype(str).tolist()

        semantic_compact, semantic_rank = _compact_activity_space(
            _embed(descriptors, args.cache_dir)
        )
        raw_compact, _ = _compact_activity_space(_embed(labels, args.cache_dir))
        semantic_map = dict(zip(labels, semantic_compact, strict=True))
        raw_map = dict(zip(labels, raw_compact, strict=True))

        sources = {
            "p7_llm_semantic_bge": semantic_map,
            "raw_activity_label_bge": raw_map,
            "activity2vec_log_context_d16": activity_context_embeddings(
                variants,
                context_window=args.context_window,
                embedding_dim=16,
            ),
            f"activity2vec_log_context_d{semantic_rank}": activity_context_embeddings(
                variants,
                context_window=args.context_window,
                embedding_dim=semantic_rank,
            ),
        }

        for name, activity_map in sources.items():
            matrix = ordered_semantic_pyramid(sequences, activity_map)
            row, labels_out = _evaluate(
                name,
                matrix,
                variants,
                weights,
                matched_k,
                config,
                reference_labels,
            )
            row["dataset"] = dataset
            row["uses_external_knowledge"] = name.startswith(
                ("p7_", "raw_activity_label")
            )
            source_rows.append(row)
            dataset_dir = args.output_dir / dataset
            dataset_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "variant_id": variant_ids,
                    "cluster_label": labels_out,
                    "cluster_id": [
                        f"{name}_k{matched_k}_c{int(label):02d}"
                        for label in labels_out
                    ],
                }
            ).to_parquet(
                dataset_dir / f"assignments_{name}_k{matched_k}.parquet",
                index=False,
            )
            print(f"{dataset} {name} done", flush=True)

        for levels in ((1,), (1, 2), (1, 2, 4), (1, 2, 4, 8)):
            matrix = ordered_semantic_pyramid(
                sequences,
                semantic_map,
                levels=levels,
            )
            row, _ = _evaluate(
                "p7_levels_" + "_".join(str(level) for level in levels),
                matrix,
                variants,
                weights,
                matched_k,
                config,
                reference_labels,
            )
            row["dataset"] = dataset
            row["levels"] = "-".join(str(level) for level in levels)
            row["segments"] = sum(levels)
            depth_rows.append(row)
            print(f"{dataset} levels {levels} done", flush=True)

    source_frame = pd.DataFrame(source_rows)
    depth_frame = pd.DataFrame(depth_rows)
    source_frame.to_csv(args.output_dir / "activity_source_benchmark.csv", index=False)
    depth_frame.to_csv(args.output_dir / "pyramid_depth_ablation.csv", index=False)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "ancillary_after_primary_artifact_lock",
                "purpose": (
                    "Isolate the activity-vector source and pooling depth while "
                    "holding composition, weighting, and clustering identical."
                ),
                "embedding_model": MODEL_ID,
                "embedding_revision": MODEL_REVISION,
                "context_window": args.context_window,
                "dynamic_config": config_record,
                "datasets": [dataset for dataset, *_ in selected_datasets],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output_dir / "activity_source_benchmark.csv")


if __name__ == "__main__":
    main()
