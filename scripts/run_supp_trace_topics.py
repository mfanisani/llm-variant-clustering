"""Supplementary baseline: Semantic Trace Topics (TF-IDF + NMF) variant encoding.

Reproduces the trace-as-document encoding of Andrienko et al. (VIPRA 2026) and
TracePrism. Variants are tokenised into activity unigrams, 2-grams and 3-grams,
weighted with TF-IDF and factorised with NMF. The topic weights then enter the
unchanged case-weighted K-means at the frozen matched K. All settings are fixed
in evaluation/supp_trace_topics_v1/PRESPEC.md. Frozen artifacts are only read.

Stages (run in order):
  silhouette  topic-count rule, matched-K and automatic-K clustering,
              silhouette with paired bootstrap against P7 and n-gram, and
              sensitivity analyses S1-S3
  stability   12-seed ARI matrices with a seed-level bootstrap
  heldout     held-out process-model F1 on the frozen split (PYTHONHASHSEED=0)
  summarize   per-log summary table and manifest with SHA-256s
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "evaluation" / "supp_trace_topics_v1"
BENCHMARK_DIR = ROOT / "evaluation" / "full_benchmark"
FROZEN_K8_STABILITY = ROOT / "evaluation/statistical_uncertainty_v2/seed_stability"
PAPER_TABLES = ROOT / "evaluation" / "paper_tables_v1"

TOPIC_GRID = (5, 21)
FALLBACK_TOPICS = 10
FIXED_TOPIC_SENSITIVITY = (5, 10, 15, 20)
NMF_CONFIGS: dict[str, dict[str, object]] = {
    # Original VIPRA Python pipeline: plain NNDSVD with the default CD solver.
    "original": {"init": "nndsvd", "solver": "cd", "max_iter": 200, "tol": 1e-4},
    # TracePrism JS port: NNDSVDa with 50 multiplicative updates.
    "traceprism": {"init": "nndsvda", "solver": "mu", "max_iter": 50, "tol": 0.0},
}
RESAMPLES = 400
SEED_RESAMPLES = 4000
BOOTSTRAP_SEED = 20240601
MAX_BOOTSTRAP_VARIANTS = 2500
STABILITY_K8_DATASETS = ("bpi2020", "bpi2017", "sepsis", "rtfm")
HELDOUT_DATASETS = ("bpi2020", "bpi2017", "bpi2019", "rtfm")
NOISE_THRESHOLD = 0.2
PRIMARY = "stt_nmf"


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------


def trace_tokens(sequence: list[str]) -> list[str]:
    """Activity unigrams, consecutive 2-grams and 3-grams of one trace."""
    tokens = [f"act:{activity}" for activity in sequence]
    for size, prefix in ((2, "2g"), (3, "3g")):
        for start in range(len(sequence) - size + 1):
            gram = sequence[start : start + size]
            tokens.append(f"{prefix}:{json.dumps(gram, ensure_ascii=False)}")
    return tokens


def token_count_matrix(
    sequences: list[list[str]],
) -> tuple[csr_matrix, tuple[str, ...]]:
    vectorizer = DictVectorizer(sparse=True, sort=True)
    counts = vectorizer.fit_transform(
        [Counter(trace_tokens(sequence)) for sequence in sequences]
    ).astype(np.float64)
    return csr_matrix(counts), tuple(vectorizer.get_feature_names_out())


def tfidf_matrix(counts: csr_matrix) -> csr_matrix:
    """Raw-count TF, smoothed IDF ln((1+n)/(1+df))+1, L2 rows (sklearn defaults)."""
    transformer = TfidfTransformer(
        norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False
    )
    return csr_matrix(transformer.fit_transform(counts))


@dataclass(frozen=True)
class NMFFit:
    weights: np.ndarray
    components: np.ndarray
    relative_error: float
    iterations: int
    converged: bool


def fit_nmf(matrix: csr_matrix, n_topics: int, config: str) -> NMFFit:
    model = NMF(
        n_components=n_topics,
        beta_loss="frobenius",
        alpha_W=0.0,
        alpha_H="same",
        l1_ratio=0.0,
        random_state=0,
        **NMF_CONFIGS[config],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        weights = model.fit_transform(matrix)
    converged = not any(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    norm = float(np.sqrt(matrix.multiply(matrix).sum()))
    return NMFFit(
        weights=np.asarray(weights, dtype=np.float64),
        components=np.asarray(model.components_, dtype=np.float64),
        relative_error=float(model.reconstruction_err_) / norm,
        iterations=int(model.n_iter_),
        converged=converged,
    )


def topic_grid(n_documents: int, vocabulary_size: int) -> list[int]:
    upper = min(TOPIC_GRID[1], min(n_documents, vocabulary_size) - 1)
    if upper < TOPIC_GRID[0]:
        raise ValueError("Too few documents or tokens for the topic grid")
    return list(range(TOPIC_GRID[0], upper + 1))


def select_topic_count(
    topic_counts: list[int], errors: list[float]
) -> tuple[int, str, np.ndarray]:
    """Knee of the reconstruction-error curve (max distance below the chord).

    Ties go to the smaller topic count. Without a decreasing curve or a point
    below the chord, the pre-declared fallback is used.
    """
    counts = np.asarray(topic_counts, dtype=np.float64)
    values = np.asarray(errors, dtype=np.float64)
    if counts.size != values.size or counts.size == 0:
        raise ValueError("topic_counts and errors must be aligned and non-empty")
    if np.any(np.diff(counts) <= 0):
        raise ValueError("topic_counts must be strictly increasing")
    fallback = int(min(FALLBACK_TOPICS, counts.max()))
    scores = np.full(counts.size, np.nan)
    if counts.size < 3 or values[0] <= values[-1]:
        return fallback, "fallback_no_decreasing_curve", scores
    x = (counts - counts[0]) / (counts[-1] - counts[0])
    y = (values[0] - values) / (values[0] - values[-1])
    scores = y - x
    best = int(np.argmax(scores))
    if scores[best] <= 0:
        return fallback, "fallback_no_knee", scores
    return int(counts[best]), "knee", scores


@dataclass(frozen=True)
class TopicEncoding:
    config: str
    n_topics: int
    weights: np.ndarray
    rule_status: str
    sweep: pd.DataFrame
    vocabulary_size: int
    fit: NMFFit


def topic_encoding(
    sequences: list[list[str]],
    config: str,
    *,
    n_topics: int | None = None,
    tfidf: csr_matrix | None = None,
) -> TopicEncoding:
    if tfidf is None:
        counts, _ = token_count_matrix(sequences)
        tfidf = tfidf_matrix(counts)
    vocabulary_size = int(tfidf.shape[1])
    if n_topics is not None:
        fit = fit_nmf(tfidf, n_topics, config)
        sweep = pd.DataFrame(
            [{"n_topics": n_topics, "relative_error": fit.relative_error,
              "iterations": fit.iterations, "converged": fit.converged,
              "knee_score": np.nan, "selected": True}]
        )
        return TopicEncoding(config, n_topics, fit.weights, "fixed", sweep,
                             vocabulary_size, fit)

    grid = topic_grid(tfidf.shape[0], vocabulary_size)
    fits = {count: fit_nmf(tfidf, count, config) for count in grid}
    chosen, status, scores = select_topic_count(
        grid, [fits[count].relative_error for count in grid]
    )
    if chosen not in fits:
        fits[chosen] = fit_nmf(tfidf, chosen, config)
    sweep = pd.DataFrame(
        [
            {
                "n_topics": count,
                "relative_error": fits[count].relative_error,
                "iterations": fits[count].iterations,
                "converged": fits[count].converged,
                "knee_score": float(score),
                "selected": count == chosen,
            }
            for count, score in zip(grid, scores, strict=True)
        ]
    )
    return TopicEncoding(config, chosen, fits[chosen].weights, status, sweep,
                         vocabulary_size, fits[chosen])


# --------------------------------------------------------------------------
# Data and comparators (existing pipeline, unchanged)
# --------------------------------------------------------------------------


def _datasets(selected: list[str]) -> list[tuple[str, str, int, str]]:
    from run_representation_source_benchmark import DATASETS

    known = {entry[0] for entry in DATASETS}
    if unknown := sorted(set(selected) - known):
        raise ValueError(f"Unknown datasets: {unknown}")
    return [entry for entry in DATASETS if entry[0] in set(selected)]


def load_variants(dataset: str, folder: str) -> pd.DataFrame:
    variants = pd.read_parquet(ROOT / "datasets/processed" / folder / "variants.parquet")
    index_path = BENCHMARK_DIR / dataset / "embedding_index.parquet"
    if index_path.exists():
        keep = frozenset(pd.read_parquet(index_path)["variant_id"].astype(str))
        variants = variants.loc[variants["variant_id"].astype(str).isin(keep)].copy()
    return variants.reset_index(drop=True)


def p7_matrix(dataset: str, sequences: list[list[str]], cache_dir: Path) -> np.ndarray:
    from llm_variant_clustering.semantic_compiler import ordered_semantic_pyramid
    from run_representation_source_benchmark import _compact_activity_space, _embed

    index = pd.read_parquet(BENCHMARK_DIR / dataset / "activity_embedding_index.parquet")
    labels = index["activity_label"].astype(str).tolist()
    descriptors = index["canonical_descriptor"].astype(str).tolist()
    compact, _ = _compact_activity_space(_embed(descriptors, cache_dir))
    return ordered_semantic_pyramid(sequences, dict(zip(labels, compact, strict=True)))


def _labels_for(path: Path, variant_ids: list[str]) -> np.ndarray:
    frame = pd.read_parquet(path)
    lookup = frame.set_index(frame["variant_id"].astype(str))["cluster_label"]
    return lookup.loc[variant_ids].to_numpy()


def _assignment_frame(variant_ids: list[str], labels: np.ndarray, method: str,
                      k: int) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=int)
    return pd.DataFrame(
        {
            "variant_id": variant_ids,
            "cluster_label": labels,
            "cluster_id": [f"{method}_k{k}_c{label:02d}" for label in labels],
        }
    )


def _sample_size(n_variants: int) -> int | None:
    return MAX_BOOTSTRAP_VARIANTS if n_variants > MAX_BOOTSTRAP_VARIANTS else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_table(name: str) -> pd.DataFrame:
    return pd.read_csv(PAPER_TABLES / name).set_index("log")


# --------------------------------------------------------------------------
# Stage: silhouette
# --------------------------------------------------------------------------


def stage_silhouette(selected: list[str], cache_dir: Path, output_dir: Path) -> None:
    from llm_variant_clustering.clustering import weighted_kmeans
    from llm_variant_clustering.dynamic_clustering import NoEligibleKError, select_dynamic_k
    from llm_variant_clustering.evaluation import evaluate_partition
    from llm_variant_clustering.features import FeatureSet, ngram_tfidf_features, parse_sequences
    from llm_variant_clustering.uncertainty import bootstrap_silhouette, paired_silhouette_difference
    from run_p7_development_comparison import _load_dynamic_config

    config, _ = _load_dynamic_config(ROOT / "configs/clustering/dynamic_k_v1.json")
    table2 = _frozen_table("table2_matched_k_silhouette.csv")
    per_log = output_dir / "per_log"
    per_log.mkdir(parents=True, exist_ok=True)

    for dataset, folder, matched_k, partition in _datasets(selected):
        variants = load_variants(dataset, folder)
        variant_ids = variants["variant_id"].astype(str).tolist()
        sequences = parse_sequences(variants)
        weights = variants["frequency"].to_numpy(dtype=np.float64)
        sample_size = _sample_size(len(variants))
        assignment_dir = output_dir / "assignments" / dataset
        assignment_dir.mkdir(parents=True, exist_ok=True)
        rows, paired_rows, checks, sweeps = [], [], [], []

        def interval_row(method, k, labels, distances, metric, matrix, extra,
                         with_ci=True, stability=None):
            metrics = evaluate_partition(csr_matrix(matrix), labels, weights,
                                         metric=metric, distances=distances)
            row = {
                "dataset": dataset, "method": method, "k": k, "metric": metric,
                "case_weighted_silhouette": metrics["case_weighted_silhouette"],
                "variant_silhouette": metrics["variant_silhouette"],
                "case_cluster_entropy": metrics["case_cluster_entropy"],
                "min_cluster_variants": metrics["min_cluster_variants"],
                "seed_stability_ari_5": stability,
                "bootstrap_sample_size": sample_size or len(variants),
                **extra,
            }
            if with_ci:
                interval = bootstrap_silhouette(
                    distances, labels, weights, resamples=RESAMPLES,
                    seed=BOOTSTRAP_SEED, sample_size=sample_size,
                )
                row.update(interval.as_dict("silhouette_"))
            rows.append(row)
            return row

        def paired(first_name, first, second_name, second, k, family):
            difference = paired_silhouette_difference(
                first[0], first[1], second[0], second[1], weights,
                resamples=RESAMPLES, seed=BOOTSTRAP_SEED, sample_size=sample_size,
            )
            record = {"dataset": dataset, "first": first_name, "second": second_name,
                      "k_first": k, "k_second": second[2],
                      "bootstrap_sample_size": sample_size or len(variants),
                      "family": family, **difference.as_dict("difference_")}
            paired_rows.append((record, difference))

        # Comparator: P7 at matched K (must reproduce the frozen partition).
        p7_features = p7_matrix(dataset, sequences, cache_dir)
        p7_distances = pairwise_distances(p7_features, metric="cosine")
        p7_result = weighted_kmeans(p7_features, weights, n_clusters=matched_k)
        p7_row = interval_row("p7_llm_semantic_bge", matched_k, p7_result.labels,
                              p7_distances, "cosine", p7_features, {},
                              stability=p7_result.mean_seed_stability_ari)
        frozen_p7 = _labels_for(
            ROOT / f"evaluation/representation_sources/{dataset}/"
            f"assignments_p7_llm_semantic_bge_k{matched_k}.parquet", variant_ids)
        frozen_table_p7 = _labels_for(
            BENCHMARK_DIR / dataset /
            f"assignments_p7_ordered_pyramid_1_2_4_bge_{partition}.parquet", variant_ids)
        checks.append({
            "dataset": dataset, "check": "p7_matched_k_reproduction",
            "recomputed": p7_row["case_weighted_silhouette"],
            "frozen": float(table2.loc[dataset, "p7"]),
            "ari_vs_representation_sources": adjusted_rand_score(frozen_p7, p7_result.labels),
            "ari_vs_full_benchmark_partition": adjusted_rand_score(frozen_table_p7, p7_result.labels),
        })
        p7 = (p7_distances, p7_result.labels, matched_k)
        print(f"{dataset} P7 {p7_row['case_weighted_silhouette']:.4f}", flush=True)

        # Comparator: n-gram TF-IDF 1-3, matched K and frozen own-K partition.
        ngram = ngram_tfidf_features(variants)
        ngram_distances = pairwise_distances(ngram.matrix, metric="cosine")
        ngram_result = weighted_kmeans(ngram.matrix, weights, n_clusters=matched_k)
        interval_row("ngram_tfidf_1_3", matched_k, ngram_result.labels, ngram_distances,
                     "cosine", ngram.matrix, {},
                     stability=ngram_result.mean_seed_stability_ari)
        ngram_own_labels = _labels_for(
            BENCHMARK_DIR / dataset / f"assignments_ngram_tfidf_1_3_{partition}.parquet",
            variant_ids)
        own_k = int(np.unique(ngram_own_labels).size)
        own_row = interval_row("ngram_tfidf_1_3_frozen_own_k", own_k, ngram_own_labels,
                               ngram_distances, "cosine", ngram.matrix, {})
        checks.append({
            "dataset": dataset, "check": "ngram_own_k_reproduction",
            "recomputed": own_row["case_weighted_silhouette"],
            "frozen": float(table2.loc[dataset, "ngram_own_k"]),
            "ari_vs_representation_sources": np.nan,
            "ari_vs_full_benchmark_partition": np.nan,
        })
        ngram_matched = (ngram_distances, ngram_result.labels, matched_k)
        ngram_own = (ngram_distances, ngram_own_labels, own_k)
        print(f"{dataset} n-gram done", flush=True)

        # Primary topic encoding and S1 (TracePrism NMF).
        counts, vocabulary = token_count_matrix(sequences)
        tfidf = tfidf_matrix(counts)
        encodings = {
            PRIMARY: topic_encoding(sequences, "original", tfidf=tfidf),
            "stt_nmf_traceprism": topic_encoding(sequences, "traceprism", tfidf=tfidf),
        }
        for name, encoding in encodings.items():
            sweep = encoding.sweep.copy()
            sweep.insert(0, "method", name)
            sweep.insert(0, "dataset", dataset)
            sweep["rule_status"] = encoding.rule_status
            sweep["vocabulary_size"] = encoding.vocabulary_size
            sweeps.append(sweep)
        np.savez_compressed(
            assignment_dir / "topic_weights.npz",
            variant_ids=np.asarray(variant_ids),
            **{name: enc.weights for name, enc in encodings.items()},
        )

        variants_to_run = {
            PRIMARY: (csr_matrix(encodings[PRIMARY].weights), "euclidean", encodings[PRIMARY]),
            "stt_nmf_traceprism": (csr_matrix(encodings["stt_nmf_traceprism"].weights), "euclidean",
                                   encodings["stt_nmf_traceprism"]),
            "stt_nmf_cosine": (csr_matrix(normalize(encodings[PRIMARY].weights, norm="l2")), "cosine",
                               encodings[PRIMARY]),
        }
        for name, (matrix, metric, encoding) in variants_to_run.items():
            distances = pairwise_distances(matrix, metric=metric)
            result = weighted_kmeans(matrix, weights, n_clusters=matched_k)
            extra = {"n_topics": encoding.n_topics, "nmf_config": encoding.config,
                     "topic_rule_status": encoding.rule_status,
                     "vocabulary_size": encoding.vocabulary_size,
                     "nmf_converged": encoding.fit.converged,
                     "nmf_iterations": encoding.fit.iterations}
            interval_row(name, matched_k, result.labels, distances, metric, matrix, extra,
                         stability=result.mean_seed_stability_ari)
            _assignment_frame(variant_ids, result.labels, name, matched_k).to_parquet(
                assignment_dir / f"assignments_{name}_k{matched_k}.parquet", index=False)
            topic = (distances, result.labels, matched_k)
            family = "primary" if name == PRIMARY else "sensitivity"
            paired(name, topic, "p7_llm_semantic_bge", p7, matched_k, family)
            paired(name, topic, "ngram_tfidf_1_3", ngram_matched, matched_k, family)
            if name == PRIMARY:
                paired(name, topic, "ngram_tfidf_1_3_frozen_own_k", ngram_own,
                       matched_k, family)
                features = FeatureSet(name=name, matrix=matrix,
                                      variant_ids=tuple(variant_ids),
                                      feature_names=tuple(f"topic_{i:02d}" for i in
                                                          range(matrix.shape[1])),
                                      metric="euclidean")
                try:
                    selection = select_dynamic_k(features, weights, config,
                                                 distances=distances)
                except NoEligibleKError as error:
                    candidates = error.candidates
                    candidates["selected"] = False
                    dynamic_status, dynamic_k = "no_eligible_k", None
                else:
                    candidates = selection.candidates
                    dynamic_status, dynamic_k = "selected", selection.selected_k
                    labels = selection.selected_result.labels
                    _assignment_frame(variant_ids, labels, f"{name}_dynamic",
                                      dynamic_k).to_parquet(
                        assignment_dir / f"assignments_{name}_dynamic.parquet",
                        index=False)
                    interval_row(f"{name}_dynamic", dynamic_k, labels, distances,
                                 metric, matrix, extra,
                                 stability=selection.selected_result.mean_seed_stability_ari)
                candidates = candidates.copy()
                candidates.insert(0, "dataset", dataset)
                candidates["status"] = dynamic_status
                candidates.to_csv(per_log / f"{dataset}_dynamic_k_candidates.csv",
                                  index=False)
            del distances
            print(f"{dataset} {name} T={encoding.n_topics} done", flush=True)

        # S3: fixed topic counts (point estimates only).
        for count in FIXED_TOPIC_SENSITIVITY:
            if count > min(tfidf.shape) - 1:
                continue
            fixed = topic_encoding(sequences, "original", n_topics=count, tfidf=tfidf)
            distances = pairwise_distances(fixed.weights, metric="euclidean")
            result = weighted_kmeans(csr_matrix(fixed.weights), weights, n_clusters=matched_k)
            interval_row(f"{PRIMARY}_fixed_t{count}", matched_k, result.labels, distances,
                         "euclidean", fixed.weights,
                         {"n_topics": count, "nmf_config": "original",
                          "topic_rule_status": "fixed",
                          "vocabulary_size": fixed.vocabulary_size,
                          "nmf_converged": fixed.fit.converged,
                          "nmf_iterations": fixed.fit.iterations},
                         with_ci=False, stability=result.mean_seed_stability_ari)
            del distances

        # Bonferroni over the three primary comparisons, read from the same samples.
        primary_family = [item for item in paired_rows if item[0]["family"] == "primary"]
        bonferroni = 1 - 0.05 / max(1, len(primary_family))
        records = []
        for record, difference in paired_rows:
            if record["family"] == "primary":
                adjusted = difference.adjusted_interval(bonferroni)
                record.update({
                    "bonferroni_family_size": len(primary_family),
                    "bonferroni_confidence": bonferroni,
                    "bonferroni_ci_lower": adjusted.lower,
                    "bonferroni_ci_upper": adjusted.upper,
                    "bonferroni_excludes_zero": bool(adjusted.lower > 0 or adjusted.upper < 0),
                })
            records.append(record)

        pd.DataFrame(rows).to_csv(per_log / f"{dataset}_silhouette.csv", index=False)
        pd.DataFrame(records).to_csv(per_log / f"{dataset}_paired_silhouette.csv",
                                     index=False)
        pd.DataFrame(checks).to_csv(per_log / f"{dataset}_reproduction_checks.csv",
                                    index=False)
        pd.concat(sweeps, ignore_index=True).to_csv(
            per_log / f"{dataset}_topic_count_sweep.csv", index=False)
        print(f"{dataset} silhouette stage complete", flush=True)


# --------------------------------------------------------------------------
# Stage: stability
# --------------------------------------------------------------------------


def ari_matrix(labelings: list[np.ndarray]) -> np.ndarray:
    size = len(labelings)
    matrix = np.ones((size, size))
    for left in range(size):
        for right in range(left + 1, size):
            matrix[left, right] = matrix[right, left] = adjusted_rand_score(
                labelings[left], labelings[right])
    return matrix


def stage_stability(selected: list[str], cache_dir: Path, output_dir: Path) -> None:
    from sklearn.cluster import KMeans

    from llm_variant_clustering.features import ngram_tfidf_features, parse_sequences
    from llm_variant_clustering.uncertainty import bootstrap_seed_agreement
    from run_reseeding_stability import SEEDS

    frozen_k8 = np.load(FROZEN_K8_STABILITY / "ari_matrices.npz")
    frozen_k9 = np.load(FROZEN_K8_STABILITY / "bpi2020_matched_k9" / "ari_matrices.npz")
    per_log = output_dir / "per_log"
    per_log.mkdir(parents=True, exist_ok=True)

    for dataset, folder, matched_k, _ in _datasets(selected):
        variants = load_variants(dataset, folder)
        variant_ids = variants["variant_id"].astype(str).tolist()
        sequences = parse_sequences(variants)
        weights = variants["frequency"].to_numpy(dtype=np.float64)
        stored = np.load(output_dir / "assignments" / dataset / "topic_weights.npz")
        if stored["variant_ids"].tolist() != variant_ids:
            raise ValueError(f"{dataset}: stored topic weights are misaligned")
        matrices = {
            PRIMARY: csr_matrix(stored[PRIMARY]),
            "p7_llm_semantic_bge": p7_matrix(dataset, sequences, cache_dir),
            "ngram_tfidf_1_3": ngram_tfidf_features(variants).matrix,
        }
        ks = [matched_k] + ([8] if dataset in STABILITY_K8_DATASETS and matched_k != 8
                            else [])
        rows, paired_rows, store = [], [], {}
        for k in ks:
            agreement = {}
            for name, matrix in matrices.items():
                labelings = [
                    KMeans(n_clusters=k, random_state=seed, n_init=10, algorithm="lloyd")
                    .fit_predict(matrix, sample_weight=weights)
                    for seed in SEEDS
                ]
                agreement[name] = ari_matrix(labelings)
                store[f"{dataset}__k{k}__{name}"] = agreement[name]
                interval = bootstrap_seed_agreement(agreement[name],
                                                    resamples=SEED_RESAMPLES)
                frozen_key = f"{dataset}__p7_llm_semantic_bge"
                frozen = None
                if name == "p7_llm_semantic_bge":
                    if k == 8 and frozen_key in frozen_k8:
                        frozen = frozen_k8[frozen_key]
                    elif dataset == "bpi2020" and k == 9 and frozen_key in frozen_k9:
                        frozen = frozen_k9[frozen_key]
                rows.append({
                    "dataset": dataset, "method": name, "k": k,
                    "seed_count": len(SEEDS), "bootstrap_unit": "seed",
                    **interval.as_dict("stability_"),
                    "matches_frozen_p7_matrix": (None if frozen is None else
                                                 bool(np.allclose(frozen, agreement[name],
                                                                  atol=1e-12))),
                })
                print(f"{dataset} k={k} {name} {interval.point_estimate:.3f}", flush=True)
            for comparator in ("p7_llm_semantic_bge", "ngram_tfidf_1_3"):
                difference = bootstrap_seed_agreement(
                    agreement[PRIMARY], second=agreement[comparator],
                    resamples=SEED_RESAMPLES)
                paired_rows.append({"dataset": dataset, "first": PRIMARY,
                                    "second": comparator, "k": k,
                                    **difference.as_dict("difference_")})
        pd.DataFrame(rows).to_csv(per_log / f"{dataset}_seed_stability.csv", index=False)
        pd.DataFrame(paired_rows).to_csv(
            per_log / f"{dataset}_paired_seed_stability.csv", index=False)
        np.savez_compressed(per_log / f"{dataset}_ari_matrices.npz", **store)


# --------------------------------------------------------------------------
# Stage: held-out process-model F1
# --------------------------------------------------------------------------


def _xes_path(folder: str) -> Path:
    matches = sorted((ROOT / "datasets/raw" / folder).glob("*.xes*"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one XES file for {folder}: {matches}")
    return matches[0]


def _aggregate(metrics: pd.DataFrame) -> list[dict]:
    rows = []
    for method, group in metrics[metrics["scope"].eq("cluster")].groupby("method", sort=True):
        weights = group["test_cases"]
        rows.append({
            "method": method, "scope": "test_case_weighted_aggregate",
            "cluster_id": "aggregate", "noise_threshold": NOISE_THRESHOLD,
            "train_variants": int(group["train_variants"].sum()),
            "test_variants": int(group["test_variants"].sum()),
            "train_cases": int(group["train_cases"].sum()),
            "test_cases": int(group["test_cases"].sum()),
            **{metric: float((group[metric] * weights).sum() / weights.sum())
               for metric in ["fitness", "precision", "fitness_precision_f1"]},
        })
    return rows


def _evaluate_split(events, case_map, variant_split, assignments) -> pd.DataFrame:
    from llm_variant_clustering.process_evaluation import (
        evaluate_holdout_model,
        evaluate_process_partition_holdout,
    )

    split_by_variant = dict(zip(variant_split["variant_id"].astype(str),
                                variant_split["split"], strict=True))
    split_by_case = dict(zip(case_map["case_id"].astype(str),
                             case_map["variant_id"].astype(str).map(split_by_variant),
                             strict=True))
    event_splits = events["case:concept:name"].astype(str).map(split_by_case)
    rows = [{
        "method": "whole_log", "scope": "whole_log", "cluster_id": "whole_log",
        "noise_threshold": NOISE_THRESHOLD,
        "train_variants": int(variant_split["split"].eq("train").sum()),
        "test_variants": int(variant_split["split"].eq("test").sum()),
        **evaluate_holdout_model(events[event_splits.eq("train")].copy(),
                                 events[event_splits.eq("test")].copy(),
                                 noise_threshold=NOISE_THRESHOLD),
    }]
    for method, frame in assignments.items():
        partition = evaluate_process_partition_holdout(
            events, case_map, frame, variant_split, method=method,
            noise_threshold=NOISE_THRESHOLD)
        rows.extend(partition.to_dict(orient="records"))
        print(f"  {method} evaluated", flush=True)
    metrics = pd.DataFrame(rows)
    return pd.concat([metrics, pd.DataFrame(_aggregate(metrics))], ignore_index=True)


def stage_heldout(selected: list[str], output_dir: Path) -> None:
    import pm4py

    from llm_variant_clustering.process_evaluation import (
        _clusters_cover_both_splits,
        select_variant_holdout_split,
    )

    heldout_dir = output_dir / "heldout"
    heldout_dir.mkdir(parents=True, exist_ok=True)
    for dataset, folder, matched_k, _ in _datasets(selected):
        if dataset not in HELDOUT_DATASETS:
            print(f"{dataset}: held-out F1 not evaluated (frozen split abstained)")
            continue
        bench = BENCHMARK_DIR / dataset
        manifest = json.loads(
            (bench / "process_model_metrics_dynamic_heldout_manifest.json").read_text(
                encoding="utf-8"))
        frozen_split = pd.read_parquet(
            bench / "process_model_metrics_dynamic_heldout_variant_split.parquet")
        frozen_split["variant_id"] = frozen_split["variant_id"].astype(str)
        comparators = {method: pd.read_parquet(bench / f"assignments_{method}.parquet")
                       for method in manifest["methods"]}
        topic_dir = output_dir / "assignments" / dataset
        topic = {}
        for name, path in ((f"{PRIMARY}_dynamic", topic_dir / f"assignments_{PRIMARY}_dynamic.parquet"),
                           (f"{PRIMARY}_k{matched_k}", topic_dir / f"assignments_{PRIMARY}_k{matched_k}.parquet")):
            if path.exists():
                topic[name] = pd.read_parquet(path)
        for frame in [*comparators.values(), *topic.values()]:
            frame["variant_id"] = frame["variant_id"].astype(str)

        selected_variants = frozenset(frozen_split["variant_id"])
        for name, frame in {**comparators, **topic}.items():
            if frozenset(frame["variant_id"]) != selected_variants:
                raise ValueError(f"{dataset} {name}: variants differ from the frozen split")
        coverage = {name: bool(_clusters_cover_both_splits(frame, frozen_split))
                    for name, frame in topic.items()}

        case_map = pd.read_parquet(ROOT / "datasets/processed" / folder / "case_variant_map.parquet")
        case_map = case_map[case_map["variant_id"].astype(str).isin(selected_variants)].copy()
        events = pm4py.read_xes(str(_xes_path(folder)), return_legacy_log_object=False,
                                show_progress_bar=False)
        selected_cases = set(case_map["case_id"].astype(str))
        events = events[events["case:concept:name"].astype(str).isin(selected_cases)].copy()
        if events["case:concept:name"].astype(str).nunique() != len(selected_cases):
            raise ValueError(f"{dataset}: events do not cover every selected case")
        print(f"{dataset}: {len(events)} events loaded", flush=True)

        feasible = {name: frame for name, frame in topic.items() if coverage[name]}
        metrics = _evaluate_split(events, case_map, frozen_split, {**comparators, **feasible})
        metrics.insert(0, "split", "frozen")
        metrics.insert(0, "dataset", dataset)
        outputs = [metrics]
        status = {name: ("evaluated_on_frozen_split" if covered else "infeasible_on_frozen_split")
                  for name, covered in coverage.items()}
        fresh_seed = None
        infeasible = {name: frame for name, frame in topic.items() if not coverage[name]}
        if infeasible:
            try:
                fresh_split, fresh_seed = select_variant_holdout_split(
                    case_map, {**comparators, **infeasible}, test_fraction=0.2,
                    case_fraction_tolerance=0.05)
            except ValueError as error:
                for name in infeasible:
                    status[name] += f"; fresh split infeasible: {error}"
            else:
                fresh = _evaluate_split(events, case_map, fresh_split,
                                        {**comparators, **infeasible})
                fresh.insert(0, "split", f"fresh_seed{fresh_seed}")
                fresh.insert(0, "dataset", dataset)
                outputs.append(fresh)
                fresh_split.to_parquet(heldout_dir / f"{dataset}_fresh_variant_split.parquet",
                                       index=False)
                for name in infeasible:
                    status[name] += f"; evaluated on fresh split seed {fresh_seed}"
        pd.concat(outputs, ignore_index=True).to_csv(heldout_dir / f"{dataset}_heldout.csv",
                                                     index=False)
        (heldout_dir / f"{dataset}_heldout_status.json").write_text(json.dumps({
            "dataset": dataset, "frozen_split_seed": manifest.get("split_seed"),
            "frozen_methods": manifest["methods"], "topic_status": status,
            "fresh_split_seed": fresh_seed,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "noise_threshold": NOISE_THRESHOLD,
        }, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Stage: summarize
# --------------------------------------------------------------------------


def _read_all(output_dir: Path, suffix: str) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted((output_dir / "per_log").glob(f"*_{suffix}.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def stage_summarize(selected: list[str], output_dir: Path) -> None:
    from llm_variant_clustering.uncertainty import bootstrap_cluster_metric

    silhouette = _read_all(output_dir, "silhouette")
    paired = _read_all(output_dir, "paired_silhouette")
    checks = _read_all(output_dir, "reproduction_checks")
    sweeps = _read_all(output_dir, "topic_count_sweep")
    stability = _read_all(output_dir, "seed_stability")
    paired_stability = _read_all(output_dir, "paired_seed_stability")
    for name, frame in (("silhouette_all_methods", silhouette),
                        ("paired_silhouette_differences", paired),
                        ("reproduction_checks", checks), ("topic_count_sweep", sweeps),
                        ("seed_stability", stability),
                        ("paired_seed_stability_differences", paired_stability)):
        frame.to_csv(output_dir / f"{name}.csv", index=False)

    heldout_frames = [pd.read_csv(path) for path in
                      sorted((output_dir / "heldout").glob("*_heldout.csv"))]
    heldout = pd.concat(heldout_frames, ignore_index=True) if heldout_frames else pd.DataFrame()
    f1_rows = []
    for (dataset, split, method), group in (heldout.groupby(["dataset", "split", "method"])
                                            if not heldout.empty else []):
        clusters = group[group["scope"].eq("cluster")]
        aggregate = group[group["scope"].isin(["test_case_weighted_aggregate", "whole_log"])]
        row = {"dataset": dataset, "split": split, "method": method,
               "clusters": len(clusters),
               "test_case_weighted_f1": float(aggregate["fitness_precision_f1"].iloc[0])}
        if len(clusters) >= 2:
            interval = bootstrap_cluster_metric(
                clusters["fitness_precision_f1"].to_numpy(dtype=float),
                clusters["test_cases"].to_numpy(dtype=float),
                resamples=RESAMPLES, seed=BOOTSTRAP_SEED)
            row.update(interval.as_dict("f1_"))
        f1_rows.append(row)
    f1 = pd.DataFrame(f1_rows)
    f1.to_csv(output_dir / "heldout_f1_intervals.csv", index=False)

    table2 = _frozen_table("table2_matched_k_silhouette.csv")
    table3 = _frozen_table("table3_heldout_f1.csv")
    summary = []
    for dataset, _, matched_k, _ in _datasets(selected):
        sil = silhouette[silhouette["dataset"].eq(dataset)].set_index("method")
        pair = paired[paired["dataset"].eq(dataset)].set_index(["first", "second"])
        row = {"log": dataset, "matched_k": matched_k}
        if PRIMARY in sil.index:
            topic = sil.loc[PRIMARY]
            row.update({
                "topic_n_topics": int(topic["n_topics"]),
                "topic_silhouette": topic["case_weighted_silhouette"],
                "topic_silhouette_ci_lower": float(topic["silhouette_ci_lower"]),
                "topic_silhouette_ci_upper": float(topic["silhouette_ci_upper"]),
                "p7_silhouette_frozen": table2.loc[dataset, "p7"],
                "p7_silhouette_recomputed": sil.loc["p7_llm_semantic_bge", "case_weighted_silhouette"],
                "raw_silhouette_frozen": table2.loc[dataset, "raw_label"],
                "a2v16_silhouette_frozen": table2.loc[dataset, "activity2vec_d16"],
                "ngram_matched_k_silhouette": sil.loc["ngram_tfidf_1_3", "case_weighted_silhouette"],
                "ngram_own_k_silhouette_frozen": table2.loc[dataset, "ngram_own_k"],
                "ngram_own_k": int(table2.loc[dataset, "ngram_k"]),
            })
            for comparator, label in (("p7_llm_semantic_bge", "vs_p7"),
                                      ("ngram_tfidf_1_3", "vs_ngram_matched_k"),
                                      ("ngram_tfidf_1_3_frozen_own_k", "vs_ngram_own_k")):
                if (PRIMARY, comparator) in pair.index:
                    item = pair.loc[(PRIMARY, comparator)]
                    row[f"silhouette_diff_{label}"] = item["difference_point_estimate"]
                    row[f"silhouette_diff_{label}_ci_lower"] = float(item["difference_ci_lower"])
                    row[f"silhouette_diff_{label}_ci_upper"] = float(item["difference_ci_upper"])
            if f"{PRIMARY}_dynamic" in sil.index:
                row["topic_dynamic_k"] = int(sil.loc[f"{PRIMARY}_dynamic", "k"])
                row["topic_dynamic_silhouette"] = sil.loc[f"{PRIMARY}_dynamic",
                                                          "case_weighted_silhouette"]
            row["topic_seed_stability_ari_5"] = topic["seed_stability_ari_5"]
            row["p7_seed_stability_ari_5"] = sil.loc["p7_llm_semantic_bge", "seed_stability_ari_5"]
            row["ngram_seed_stability_ari_5"] = sil.loc["ngram_tfidf_1_3", "seed_stability_ari_5"]
        if not f1.empty:
            frozen_f1 = f1[f1["dataset"].eq(dataset) & f1["split"].eq("frozen")].set_index("method")
            if f"{PRIMARY}_dynamic" in frozen_f1.index:
                item = frozen_f1.loc[f"{PRIMARY}_dynamic"]
                row["topic_dynamic_heldout_f1"] = item["test_case_weighted_f1"]
                row["topic_dynamic_heldout_f1_ci_lower"] = float(item["f1_ci_lower"])
                row["topic_dynamic_heldout_f1_ci_upper"] = float(item["f1_ci_upper"])
            if f"{PRIMARY}_k{matched_k}" in frozen_f1.index:
                row["topic_matched_k_heldout_f1"] = frozen_f1.loc[f"{PRIMARY}_k{matched_k}",
                                                                  "test_case_weighted_f1"]
            for method in frozen_f1.index:
                if method.startswith("p7_"):
                    row["p7_heldout_f1_same_run"] = frozen_f1.loc[method, "test_case_weighted_f1"]
                elif method.startswith("ngram_"):
                    row["ngram_heldout_f1_same_run"] = frozen_f1.loc[method, "test_case_weighted_f1"]
                elif method.startswith("raw_"):
                    row["raw_heldout_f1_same_run"] = frozen_f1.loc[method, "test_case_weighted_f1"]
                elif method == "whole_log":
                    row["whole_log_heldout_f1_same_run"] = frozen_f1.loc[method, "test_case_weighted_f1"]
            if dataset in table3.index:
                row.update({"p7_heldout_f1_frozen": table3.loc[dataset, "p7"],
                            "ngram_heldout_f1_frozen": table3.loc[dataset, "ngram"],
                            "raw_heldout_f1_frozen": table3.loc[dataset, "raw_bge"],
                            "whole_log_heldout_f1_frozen": table3.loc[dataset, "whole_log"]})
        summary.append(row)
    pd.DataFrame(summary).to_csv(output_dir / "per_log_summary.csv", index=False)
    write_manifest(output_dir)


def write_manifest(output_dir: Path) -> None:
    import sklearn

    inputs = [
        ROOT / "scripts/run_supp_trace_topics.py",
        ROOT / "tests/test_supp_trace_topics.py",
        ROOT / "configs/clustering/dynamic_k_v1.json",
        PAPER_TABLES / "table2_matched_k_silhouette.csv",
        PAPER_TABLES / "table3_heldout_f1.csv",
        FROZEN_K8_STABILITY / "ari_matrices.npz",
        FROZEN_K8_STABILITY / "bpi2020_matched_k9" / "ari_matrices.npz",
    ]
    for dataset, folder, matched_k, partition in _datasets(
            ["bpi2020", "bpi2017", "bpi2019", "sepsis", "rtfm"]):
        bench = BENCHMARK_DIR / dataset
        inputs += [
            ROOT / "datasets/processed" / folder / "variants.parquet",
            ROOT / "datasets/processed" / folder / "case_variant_map.parquet",
            bench / "embedding_index.parquet",
            bench / "activity_embedding_index.parquet",
            bench / f"assignments_ngram_tfidf_1_3_{partition}.parquet",
            bench / f"assignments_p7_ordered_pyramid_1_2_4_bge_{partition}.parquet",
            ROOT / f"evaluation/representation_sources/{dataset}/assignments_p7_llm_semantic_bge_k{matched_k}.parquet",
            bench / "process_model_metrics_dynamic_heldout_variant_split.parquet",
            bench / "process_model_metrics_dynamic_heldout_manifest.json",
        ]
    outputs = sorted(path for path in output_dir.rglob("*")
                     if path.is_file() and path.name != "manifest.json")
    manifest = {
        "schema_version": 1,
        "status": "supplementary_baseline_not_part_of_frozen_study",
        "prespec": {"path": "PRESPEC.md", "sha256": _sha256(output_dir / "PRESPEC.md")},
        "method": "Semantic Trace Topics (Andrienko et al., VIPRA 2026; TracePrism arXiv 2610.28532)",
        "vocabulary": ["act", "2g", "3g"],
        "tfidf": "raw tf, smooth idf, l2 rows; df over unique variants",
        "nmf_configs": NMF_CONFIGS,
        "topic_grid": list(TOPIC_GRID),
        "fallback_topics": FALLBACK_TOPICS,
        "bootstrap": {"resamples": RESAMPLES, "seed": BOOTSTRAP_SEED,
                      "max_bootstrap_variants": MAX_BOOTSTRAP_VARIANTS,
                      "seed_resamples": SEED_RESAMPLES, "confidence": 0.95},
        "software": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "inputs": {str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
                   for path in inputs if path.exists()},
        "outputs": {str(path.relative_to(output_dir)).replace("\\", "/"): _sha256(path)
                    for path in outputs},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                              encoding="utf-8")


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["silhouette", "stability", "heldout", "summarize"])
    parser.add_argument("--datasets", nargs="+",
                        default=["bpi2020", "bpi2017", "bpi2019", "sepsis", "rtfm"])
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "models/huggingface")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "heldout" and os.environ.get("PYTHONHASHSEED") != "0":
        # Inductive Miner tie-breaks depend on the hash seed; fix it as pre-declared.
        completed = subprocess.run([sys.executable, *sys.argv],
                                   env={**os.environ, "PYTHONHASHSEED": "0"})
        raise SystemExit(completed.returncode)
    if args.stage == "silhouette":
        stage_silhouette(args.datasets, args.cache_dir, args.output_dir)
    elif args.stage == "stability":
        stage_stability(args.datasets, args.cache_dir, args.output_dir)
    elif args.stage == "heldout":
        stage_heldout(args.datasets, args.output_dir)
    else:
        stage_summarize(args.datasets, args.output_dir)


if __name__ == "__main__":
    main()
