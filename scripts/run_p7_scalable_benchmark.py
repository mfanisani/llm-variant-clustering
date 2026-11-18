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
from llm_variant_clustering.evaluation import (
    evaluate_partition,
    weighted_medoid_indices,
)
from llm_variant_clustering.features import FeatureSet, ngram_tfidf_features, parse_sequences
from llm_variant_clustering.semantic_compiler import (
    canonical_activity_text,
    mean_semantic_embeddings,
    ordered_semantic_pyramid,
)
from run_p7_development_comparison import _load_dynamic_config

MODEL_ID = "BAAI/bge-large-en-v1.5"
MODEL_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run geometry-preserving scalable P7 and baseline clustering on "
            "large variant tables."
        )
    )
    parser.add_argument("variants", type=Path)
    parser.add_argument("activity_semantics", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--dynamic-config",
        type=Path,
        default=Path("configs/clustering/dynamic_k_v1.json"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("models/huggingface"))
    parser.add_argument("--clusters", type=int, default=8)
    args = parser.parse_args()

    variants = pd.read_parquet(args.variants)
    sequences = parse_sequences(variants)
    weights = variants["frequency"].to_numpy(dtype=np.float64)
    labels = sorted({activity for sequence in sequences for activity in sequence})
    semantics = _load_semantics(args.activity_semantics)
    missing = sorted(set(labels).difference(semantics))
    if missing:
        raise ValueError(f"Missing frozen activity semantics: {missing}")

    semantic_texts = [canonical_activity_text(semantics[label]) for label in labels]
    semantic_embeddings, raw_label_embeddings = _embed_text_sets(
        semantic_texts,
        labels,
        cache_dir=args.cache_dir,
    )
    semantic_compact, semantic_rank = _compact_activity_space(semantic_embeddings)
    raw_compact, raw_rank = _compact_activity_space(raw_label_embeddings)
    semantic_map = dict(zip(labels, semantic_compact, strict=True))
    raw_map = dict(zip(labels, raw_compact, strict=True))

    p7_ordered = ordered_semantic_pyramid(sequences, semantic_map)
    p7_mean = mean_semantic_embeddings(sequences, semantic_map)
    raw_ordered = ordered_semantic_pyramid(sequences, raw_map)
    ngram = ngram_tfidf_features(variants)
    ngram = FeatureSet(
        name=ngram.name,
        matrix=ngram.matrix.astype(np.float32),
        variant_ids=ngram.variant_ids,
        feature_names=ngram.feature_names,
        metric=ngram.metric,
    )
    feature_sets = [
        embedding_feature_set("raw_activity_label_pyramid_bge", variants, raw_ordered),
        ngram,
        embedding_feature_set("p7_mean_activity_semantics_bge", variants, p7_mean),
        embedding_feature_set(
            "p7_ordered_pyramid_1_2_4_bge",
            variants,
            p7_ordered,
        ),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_embedding_artifacts(
        args.output_dir,
        variants,
        labels,
        semantic_texts,
        semantic_compact,
        p7_ordered,
        raw_ordered,
    )
    config, config_record = _load_dynamic_config(args.dynamic_config)
    fixed_rows = []
    dynamic_rows = []
    fixed_labels: dict[str, np.ndarray] = {}
    dynamic_labels: dict[str, np.ndarray] = {}
    for features in feature_sets:
        distances = pairwise_distances(
            features.matrix,
            metric=features.metric,
        )
        fixed = weighted_kmeans(
            features.matrix,
            weights,
            n_clusters=args.clusters,
        )
        fixed_metrics = evaluate_partition(
            features.matrix,
            fixed.labels,
            weights,
            metric=features.metric,
            distances=distances,
        )
        _save_assignments(
            args.output_dir,
            features,
            fixed.labels,
            weights,
            suffix=f"k{args.clusters}",
        )
        fixed_labels[features.name] = fixed.labels
        fixed_rows.append(
            {
                "method": features.name,
                "feature_count": features.matrix.shape[1],
                "selected_seed": fixed.selected_seed,
                "weighted_inertia": fixed.weighted_inertia,
                "mean_seed_stability_ari": fixed.mean_seed_stability_ari,
                **fixed_metrics,
            }
        )

        try:
            selection = select_dynamic_k(
                features,
                weights,
                config,
                distances=distances,
            )
        except NoEligibleKError as error:
            error.candidates.to_csv(
                args.output_dir / f"dynamic_candidates_{features.name}.csv",
                index=False,
            )
            dynamic_rows.append(
                {
                    "method": features.name,
                    "status": "no_eligible_k",
                    "selected_k": None,
                }
            )
        else:
            selection.candidates.to_csv(
                args.output_dir / f"dynamic_candidates_{features.name}.csv",
                index=False,
            )
            selected = selection.candidates.loc[
                selection.candidates["selected"]
            ].iloc[0]
            _save_assignments(
                args.output_dir,
                features,
                selection.selected_result.labels,
                weights,
                suffix="dynamic",
                selected_k=selection.selected_k,
            )
            dynamic_labels[features.name] = selection.selected_result.labels
            dynamic_rows.append(
                {
                    "method": features.name,
                    "status": "selected",
                    "selected_k": selection.selected_k,
                    "profile_votes": json.dumps(
                        selection.profile_votes,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    **{
                        key: selected[key]
                        for key in (
                            "mean_seed_stability_ari",
                            "variant_silhouette",
                            "case_weighted_silhouette",
                            "case_cluster_entropy",
                            "min_cluster_cases",
                            "max_cluster_cases",
                            "min_cluster_variants",
                            "max_cluster_variants",
                        )
                    },
                }
            )
        del distances

    pd.DataFrame(fixed_rows).to_csv(
        args.output_dir / f"metrics_fixed_k{args.clusters}.csv",
        index=False,
    )
    pd.DataFrame(dynamic_rows).to_csv(
        args.output_dir / "metrics_dynamic_k.csv",
        index=False,
    )
    _agreement(fixed_labels).to_csv(
        args.output_dir / f"partition_agreement_fixed_k{args.clusters}.csv",
        index=False,
    )
    _agreement(dynamic_labels).to_csv(
        args.output_dir / "partition_agreement_dynamic_k.csv",
        index=False,
    )
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "representation_lineage": "p7_activity_gpt41_v1",
                "scalable_implementation": {
                    "semantic_activity_rank": semantic_rank,
                    "raw_activity_rank": raw_rank,
                    "projection": (
                        "orthonormal basis spanning observed activity embeddings; "
                        "preserves all within-representation Euclidean and cosine geometry"
                    ),
                    "raw_baseline": (
                        "ordered 1-2-4 pyramid over embeddings of original activity "
                        "labels; avoids forbidden sequence truncation"
                    ),
                },
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "variant_count": len(variants),
                "case_count": int(weights.sum()),
                "activity_count": len(labels),
                "fixed_k": args.clusters,
                "dynamic_config": config_record,
                "frequency_usage": "sample_weight_in_clustering_objective_only",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_semantics(path: Path) -> dict[str, dict[str, object]]:
    frame = pd.read_parquet(path)
    if not frame["status"].eq("validated").all():
        raise ValueError("All activity semantics must be validated")
    if frame["activity_label"].duplicated().any():
        raise ValueError("Activity semantics contain duplicate labels")
    return {
        str(row.activity_label): json.loads(row.response_json)
        for row in frame.itertuples()
    }


def _embed_text_sets(
    semantic_texts: list[str],
    labels: list[str],
    *,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_folder=str(cache_dir),
        device="cpu",
        trust_remote_code=False,
    )
    texts = semantic_texts + labels
    tokens = model.tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    if max(map(len, tokens["input_ids"])) > model.max_seq_length:
        raise ValueError("An activity descriptor would require truncation")
    matrix = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    split = len(labels)
    return matrix[:split], matrix[split:]


def _compact_activity_space(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    _, singular_values, right = np.linalg.svd(
        matrix.astype(np.float64),
        full_matrices=False,
    )
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    compact = (matrix @ right[:rank].T).astype(np.float32)
    original_gram = matrix @ matrix.T
    compact_gram = compact @ compact.T
    if not np.allclose(original_gram, compact_gram, atol=2e-5):
        raise ValueError("Activity-space projection did not preserve geometry")
    return compact, rank


def _save_assignments(
    output_dir: Path,
    features: FeatureSet,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    suffix: str,
    selected_k: int | None = None,
) -> None:
    medoids = weighted_medoid_indices(
        features.matrix,
        labels,
        weights,
        metric=features.metric,
    )
    k = selected_k or len(np.unique(labels))
    pd.DataFrame(
        {
            "variant_id": features.variant_ids,
            "cluster_label": labels,
            "cluster_id": [
                f"{features.name}_{suffix}_k{k}_c{label:02d}"
                for label in labels
            ],
            "is_medoid": [
                index == medoids[int(label)]
                for index, label in enumerate(labels)
            ],
        }
    ).to_parquet(
        output_dir / f"assignments_{features.name}_{suffix}.parquet",
        index=False,
    )


def _save_embedding_artifacts(
    output_dir: Path,
    variants: pd.DataFrame,
    labels: list[str],
    semantic_texts: list[str],
    compact_activities: np.ndarray,
    p7_ordered: np.ndarray,
    raw_ordered: np.ndarray,
) -> None:
    index = variants[["variant_id"]]
    index.to_parquet(output_dir / "embedding_index.parquet", index=False)
    pd.DataFrame(
        {
            "activity_label": labels,
            "canonical_descriptor": semantic_texts,
        }
    ).to_parquet(output_dir / "activity_embedding_index.parquet", index=False)
    np.save(
        output_dir / "compact_activity_embeddings.npy",
        compact_activities,
        allow_pickle=False,
    )
    np.save(output_dir / "p7_ordered_embeddings.npy", p7_ordered, allow_pickle=False)
    np.save(output_dir / "raw_ordered_embeddings.npy", raw_ordered, allow_pickle=False)


def _agreement(assignments: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    methods = list(assignments)
    for left_index, left in enumerate(methods):
        for right in methods[left_index + 1 :]:
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "ari": adjusted_rand_score(
                        assignments[left],
                        assignments[right],
                    ),
                    "nmi": normalized_mutual_info_score(
                        assignments[left],
                        assignments[right],
                    ),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
