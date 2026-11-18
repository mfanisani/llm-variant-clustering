from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize

from llm_variant_clustering.clustering import weighted_kmeans
from llm_variant_clustering.dynamic_clustering import (
    DynamicKConfig,
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
    compose_variant_semantics,
    mean_semantic_embeddings,
    ordered_semantic_pyramid,
)

MODEL_ID = "BAAI/bge-large-en-v1.5"
MODEL_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare frozen P7 representations on development variants."
    )
    parser.add_argument("variants", type=Path)
    parser.add_argument("activity_semantics", type=Path)
    parser.add_argument("raw_embedding_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--dynamic-config",
        type=Path,
        default=Path("configs/clustering/dynamic_k_v1.json"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("models/huggingface"))
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument(
        "--phase",
        choices=("development", "recovery_validation", "protected_test"),
        default="development",
    )
    parser.add_argument(
        "--frozen-evaluation",
        action="store_true",
        help="Exclude development-rejected hybrid representations.",
    )
    args = parser.parse_args()

    variants = pd.read_parquet(args.variants)
    sequences = parse_sequences(variants)
    variant_ids = variants["variant_id"].astype(str).tolist()
    weights = variants["frequency"].to_numpy(dtype=np.float64)
    semantics = _load_activity_semantics(args.activity_semantics)
    used_activities = sorted({activity for sequence in sequences for activity in sequence})
    missing = sorted(set(used_activities).difference(semantics))
    if missing:
        raise ValueError(f"Missing P7 semantics for activities: {missing}")

    descriptors = [canonical_activity_text(semantics[label]) for label in used_activities]
    activity_embeddings = _embed_activity_descriptors(
        used_activities,
        descriptors,
        cache_dir=args.cache_dir,
    )
    mean_embeddings = mean_semantic_embeddings(sequences, activity_embeddings)
    pyramid_embeddings = ordered_semantic_pyramid(sequences, activity_embeddings)

    mean_features = embedding_feature_set(
        "p7_mean_activity_semantics_bge",
        variants,
        mean_embeddings,
    )
    pyramid_features = embedding_feature_set(
        "p7_ordered_pyramid_1_2_4_bge",
        variants,
        pyramid_embeddings,
    )
    raw_features = embedding_feature_set(
        "raw_sequence_bge",
        variants,
        _subset_raw_embeddings(args.raw_embedding_dir, variant_ids),
    )
    ngram_features = ngram_tfidf_features(variants)
    feature_sets = [
        raw_features,
        ngram_features,
        mean_features,
        pyramid_features,
    ]
    if not args.frozen_evaluation:
        feature_sets.extend(
            _hybrid_features(
                pyramid_features,
                ngram_features,
                semantic_weight=semantic_weight,
            )
            for semantic_weight in (0.25, 0.50, 0.75)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_semantic_artifacts(
        args.output_dir,
        variants,
        sequences,
        semantics,
        used_activities,
        descriptors,
        activity_embeddings,
        mean_embeddings,
        pyramid_embeddings,
    )
    fixed_metrics, fixed_assignments = _run_fixed_k(
        feature_sets,
        weights,
        args.clusters,
        args.output_dir,
    )
    fixed_metrics.to_csv(
        args.output_dir / f"metrics_fixed_k{args.clusters}.csv",
        index=False,
    )
    _partition_agreement(fixed_assignments).to_csv(
        args.output_dir / f"partition_agreement_fixed_k{args.clusters}.csv",
        index=False,
    )

    config, config_record = _load_dynamic_config(args.dynamic_config)
    dynamic_rows = []
    dynamic_assignments = {}
    for features in feature_sets:
        try:
            selection = select_dynamic_k(features, weights, config)
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
                    "profile_votes": None,
                }
            )
            continue
        selection.candidates.to_csv(
            args.output_dir / f"dynamic_candidates_{features.name}.csv",
            index=False,
        )
        selected = selection.candidates.loc[selection.candidates["selected"]].iloc[0]
        medoids = weighted_medoid_indices(
            features.matrix,
            selection.selected_result.labels,
            weights,
            metric=features.metric,
        )
        pd.DataFrame(
            {
                "variant_id": features.variant_ids,
                "cluster_label": selection.selected_result.labels,
                "cluster_id": [
                    f"{features.name}_dynamic_k{selection.selected_k}_c{label:02d}"
                    for label in selection.selected_result.labels
                ],
                "is_medoid": [
                    index == medoids[int(label)]
                    for index, label in enumerate(selection.selected_result.labels)
                ],
            }
        ).to_parquet(
            args.output_dir / f"assignments_{features.name}_dynamic.parquet",
            index=False,
        )
        dynamic_assignments[features.name] = selection.selected_result.labels
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
    pd.DataFrame(dynamic_rows).to_csv(
        args.output_dir / "metrics_dynamic_k.csv",
        index=False,
    )
    _partition_agreement(dynamic_assignments).to_csv(
        args.output_dir / "partition_agreement_dynamic_k.csv",
        index=False,
    )
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "representation_lineage": "p7_activity_gpt41_v1",
                "phase": args.phase,
                "frozen_evaluation": args.frozen_evaluation,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "fixed_k": args.clusters,
                "variant_count": len(variants),
                "case_count": int(weights.sum()),
                "activity_count": len(used_activities),
                "frequency_usage": "sample_weight_in_clustering_objective_only",
                "dynamic_config": config_record,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_activity_semantics(path: Path) -> dict[str, dict[str, object]]:
    frame = pd.read_parquet(path)
    required = {"activity_label", "status", "response_json"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Activity semantics table is missing columns: {missing}")
    if frame["activity_label"].duplicated().any():
        raise ValueError("Activity semantics table contains duplicate labels")
    if not frame["status"].eq("validated").all():
        raise ValueError("Every activity normalization must be validated")
    return {
        str(row.activity_label): json.loads(row.response_json)
        for row in frame.itertuples()
    }


def _embed_activity_descriptors(
    labels: list[str],
    descriptors: list[str],
    *,
    cache_dir: Path,
) -> dict[str, np.ndarray]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_folder=str(cache_dir),
        device="cpu",
        trust_remote_code=False,
    )
    tokenized = model.tokenizer(
        descriptors,
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    maximum = max(len(tokens) for tokens in tokenized["input_ids"])
    if maximum > model.max_seq_length:
        raise ValueError("An activity descriptor would require token truncation")
    matrix = model.encode(
        descriptors,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    return dict(zip(labels, matrix, strict=True))


def _subset_raw_embeddings(
    directory: Path,
    variant_ids: list[str],
) -> np.ndarray:
    index = pd.read_parquet(directory / "embedding_index.parquet")
    matrix = np.load(directory / "embeddings.npy", allow_pickle=False)
    positions = {
        str(variant_id): position
        for position, variant_id in enumerate(index["variant_id"])
    }
    missing = sorted(set(variant_ids).difference(positions))
    if missing:
        raise ValueError(f"Raw embedding index is missing {len(missing)} variants")
    return np.vstack([matrix[positions[variant_id]] for variant_id in variant_ids])


def _hybrid_features(
    semantic: FeatureSet,
    structural: FeatureSet,
    *,
    semantic_weight: float,
) -> FeatureSet:
    if semantic.variant_ids != structural.variant_ids:
        raise ValueError("Hybrid feature rows do not align")
    matrix = normalize(
        hstack(
            [
                semantic.matrix * np.sqrt(semantic_weight),
                structural.matrix * np.sqrt(1.0 - semantic_weight),
            ],
            format="csr",
        ),
        norm="l2",
    )
    return FeatureSet(
        name=f"p7_ordered_ngram_hybrid_a{semantic_weight:.2f}",
        matrix=matrix,
        variant_ids=semantic.variant_ids,
        feature_names=tuple(
            [f"semantic::{name}" for name in semantic.feature_names]
            + [f"structural::{name}" for name in structural.feature_names]
        ),
        metric="cosine",
    )


def _save_semantic_artifacts(
    output_dir: Path,
    variants: pd.DataFrame,
    sequences: list[list[str]],
    semantics: dict[str, dict[str, object]],
    labels: list[str],
    descriptors: list[str],
    activity_embeddings: dict[str, np.ndarray],
    mean_embeddings: np.ndarray,
    pyramid_embeddings: np.ndarray,
) -> None:
    compiled = variants[["variant_id", "frequency"]].copy()
    compiled["composed_semantics_json"] = [
        json.dumps(
            compose_variant_semantics(sequence, semantics),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for sequence in sequences
    ]
    compiled.to_parquet(output_dir / "compiled_variant_semantics.parquet", index=False)
    pd.DataFrame(
        {
            "activity_label": labels,
            "canonical_descriptor": descriptors,
        }
    ).to_parquet(output_dir / "activity_embedding_index.parquet", index=False)
    np.save(
        output_dir / "activity_embeddings.npy",
        np.vstack([activity_embeddings[label] for label in labels]),
        allow_pickle=False,
    )
    np.save(output_dir / "mean_semantic_embeddings.npy", mean_embeddings, allow_pickle=False)
    np.save(
        output_dir / "ordered_pyramid_embeddings.npy",
        pyramid_embeddings,
        allow_pickle=False,
    )
    primary_dir = output_dir / "p7_ordered_pyramid"
    primary_dir.mkdir(parents=True, exist_ok=True)
    variants[["variant_id"]].to_parquet(
        primary_dir / "embedding_index.parquet",
        index=False,
    )
    np.save(
        primary_dir / "embeddings.npy",
        pyramid_embeddings,
        allow_pickle=False,
    )


def _run_fixed_k(
    feature_sets: list[FeatureSet],
    weights: np.ndarray,
    clusters: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    assignments = {}
    for features in feature_sets:
        result = weighted_kmeans(
            features.matrix,
            weights,
            n_clusters=clusters,
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
        pd.DataFrame(
            {
                "variant_id": features.variant_ids,
                "cluster_label": result.labels,
                "cluster_id": [
                    f"{features.name}_k{clusters}_c{label:02d}"
                    for label in result.labels
                ],
                "is_medoid": [
                    index == medoids[int(label)]
                    for index, label in enumerate(result.labels)
                ],
            }
        ).to_parquet(
            output_dir / f"assignments_{features.name}_k{clusters}.parquet",
            index=False,
        )
        assignments[features.name] = result.labels
        rows.append(
            {
                "method": features.name,
                "feature_count": features.matrix.shape[1],
                "selected_seed": result.selected_seed,
                "weighted_inertia": result.weighted_inertia,
                "mean_seed_stability_ari": result.mean_seed_stability_ari,
                **metrics,
            }
        )
    return pd.DataFrame(rows), assignments


def _partition_agreement(assignments: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    names = list(assignments)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "ari": adjusted_rand_score(assignments[left], assignments[right]),
                    "nmi": normalized_mutual_info_score(
                        assignments[left],
                        assignments[right],
                    ),
                }
            )
    return pd.DataFrame(rows)


def _load_dynamic_config(path: Path) -> tuple[DynamicKConfig, dict[str, object]]:
    record = json.loads(path.read_text(encoding="utf-8"))
    config = DynamicKConfig.from_dict(record["config"])
    canonical = json.dumps(
        config.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    actual_hash = hashlib.sha256(canonical).hexdigest()
    if actual_hash != record.get("config_sha256"):
        raise ValueError("Dynamic clustering configuration hash mismatch")
    return config, record


if __name__ == "__main__":
    main()
