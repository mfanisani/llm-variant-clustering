"""Review 3 computational fixes: singleton silhouette convention and exact bootstraps.

Writes only to evaluation/review3_fixes_v1/. Frozen results under evaluation/ are
read, never written. Bootstrap logic is copied from
evaluation/review_bootstrap_tails_v1/run.py (same seed, same m-out-of-n cap,
same skip rule); only the partitions and the replicate count change.

  python scripts/run_review3_fixes.py embed            # isolated BGE subprocess
  python scripts/run_review3_fixes.py audit
  python scripts/run_review3_fixes.py bootstrap --logs rtfm sepsis bpi2020 bpi2017 bpi2019 bpi2012
  python scripts/run_review3_fixes.py summarize
  python scripts/run_review3_fixes.py manifest

Each partition's replicates are stored with both the previous ("old") and the
corrected ("new") singleton convention, computed from one silhouette pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.preprocessing import normalize

from llm_variant_clustering.evaluation import frequency_weighted_silhouette
from llm_variant_clustering.features import (
    activity_context_embeddings,
    ngram_tfidf_features,
    parse_sequences,
)
from llm_variant_clustering.semantic_compiler import ordered_semantic_pyramid
from llm_variant_clustering.uncertainty import _percentile_interval, _resample_indices
from run_representation_source_benchmark import _compact_activity_space
from run_statistical_uncertainty import DATASET_FOLDERS

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"
OUT = EVAL / "review3_fixes_v1"
CACHE = OUT / "cache"
REPS = OUT / "replicates"
REFIT = OUT / "partitions"
SCRATCH = OUT / "_scratch"
SEED = 20240601
CAP = 2500
RESAMPLES = 5000
TABLE2_RESAMPLES = 300
SUPP_RESAMPLES = 400
CONFIDENCE = 0.95
FAMILY = ("bpi2017", "bpi2019", "sepsis", "rtfm")
BONFERRONI = 1.0 - (1.0 - CONFIDENCE) / len(FAMILY)
P7_BENCH = "p7_ordered_pyramid_1_2_4_bge"
RAW_BENCH = "raw_activity_label_pyramid_bge"
# log, matched K (Table 2), frozen full-benchmark partition suffix
LOGS = (
    ("bpi2020", 9, "dynamic"),
    ("bpi2017", 2, "dynamic"),
    ("bpi2019", 6, "dynamic"),
    ("sepsis", 8, "k8"),
    ("rtfm", 2, "dynamic"),
)
MATCHED_K = {log: k for log, k, _ in LOGS}
SUFFIX = {log: s for log, _, s in LOGS}

_STATE: dict = {}


# --------------------------------------------------------------------------
# silhouette: previous convention and corrected convention in one pass
# --------------------------------------------------------------------------


def silhouette_old_new(distances, labels, weights) -> tuple[float, float, float]:
    """Return (old, new, min cluster case weight).

    Statement-for-statement copy of frequency_weighted_silhouette as it was
    before the fix ("old"); "new" additionally zeroes points whose own cluster
    has total case weight <= 1, which is the line added to the library.
    """
    distances = np.asarray(distances, dtype=np.float64)
    labels = np.asarray(labels)
    weights = np.asarray(weights, dtype=np.float64)
    size = len(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("At least two clusters are required")
    cluster_distances = np.empty((size, len(unique_labels)), dtype=np.float64)
    cluster_weights = np.empty(len(unique_labels), dtype=np.float64)
    encoded_labels = np.empty(size, dtype=np.int64)
    for position, cluster in enumerate(unique_labels):
        mask = labels == cluster
        cluster_weights[position] = weights[mask].sum()
        cluster_distances[:, position] = (
            distances[:, mask] @ weights[mask]
        ) / cluster_weights[position]
        encoded_labels[mask] = position
    row_indices = np.arange(size)
    own_weights = cluster_weights[encoded_labels]
    within = (
        cluster_distances[row_indices, encoded_labels]
        * own_weights
        / np.maximum(own_weights - 1.0, 1.0)
    )
    within[own_weights <= 1.0] = 0.0
    cluster_distances[row_indices, encoded_labels] = np.inf
    nearest = cluster_distances.min(axis=1)
    denominator = np.maximum(within, nearest)
    silhouettes = np.divide(
        nearest - within,
        denominator,
        out=np.zeros(size, dtype=np.float64),
        where=denominator > 0,
    )
    old = float(np.average(silhouettes, weights=weights))
    single = own_weights <= 1.0
    if single.any():
        silhouettes[single] = 0.0
        new = float(np.average(silhouettes, weights=weights))
    else:
        new = old
    return old, new, float(cluster_weights.min())


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def log_frame(log: str) -> tuple[pd.DataFrame, Path]:
    """Variant order of the frozen embeddings (identical to every source script)."""
    if log == "bpi2012":
        bench = EVAL / "supp_extra_logs_v1" / "bpi2012"
        variants = pd.read_parquet(ROOT / "datasets/processed/bpi2012/variants.parquet")
    else:
        bench = EVAL / "full_benchmark" / log
        variants = pd.read_parquet(
            ROOT / "datasets/processed" / DATASET_FOLDERS[log] / "variants.parquet"
        )
    variants = variants.assign(variant_id=variants["variant_id"].astype(str))
    index_path = bench / "embedding_index.parquet"
    if index_path.exists():
        keep = set(pd.read_parquet(index_path)["variant_id"].astype(str))
        filtered = variants.loc[variants["variant_id"].isin(keep)].reset_index(drop=True)
        order = pd.read_parquet(index_path)["variant_id"].astype(str).tolist()
        if filtered["variant_id"].tolist() != order:
            raise SystemExit(f"{log}: embedding_index order differs from filtered variants order")
        variants = filtered
    return variants.reset_index(drop=True), bench


def labels_from(path: Path, order: list[str]) -> np.ndarray:
    frame = pd.read_parquet(path)
    lookup = frame.set_index(frame["variant_id"].astype(str))["cluster_label"]
    return lookup.loc[order].to_numpy()


def cosine(matrix) -> np.ndarray:
    return pairwise_distances(matrix, metric="cosine")


def _activity_index(bench: Path) -> tuple[list[str], list[str]]:
    index = pd.read_parquet(bench / "activity_embedding_index.parquet")
    return index["activity_label"].astype(str).tolist(), index["canonical_descriptor"].astype(str).tolist()


def _pyramid(sequences, labels, vectors) -> np.ndarray:
    compact, _ = _compact_activity_space(vectors)
    return ordered_semantic_pyramid(sequences, dict(zip(labels, compact, strict=True)))


def _vectors(log: str) -> dict:
    path = CACHE / f"activity_vectors_{log}.npz"
    if not path.exists():
        raise SystemExit(f"missing {path}; run the embed step first")
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def rebuilt_matrix(log: str, key: str, variants: pd.DataFrame, sequences) -> np.ndarray:
    """Dense pyramid exactly as run_representation_uncertainty.py builds it."""
    if key == "a2v16":
        return ordered_semantic_pyramid(sequences, activity_context_embeddings(variants, embedding_dim=16))
    vec = _vectors(log)
    return _pyramid(sequences, vec["labels"].tolist(), vec[key])

def distance_sources(log: str) -> dict:
    """name -> callable returning the distance matrix exactly as the source script built it."""
    variants, bench = log_frame(log)
    sequences = parse_sequences(variants)
    sources = {}
    if log == "bpi2012":
        sources["p7"] = lambda: cosine(np.load(bench / "p7_ordered_embeddings.npy"))
        sources["raw"] = lambda: cosine(np.load(bench / "raw_ordered_embeddings.npy"))
    else:
        def from_cache(key):
            def build():
                return cosine(rebuilt_matrix(log, key, variants, sequences))
            return build
        # Table 2 intervals (run_representation_uncertainty.py), Supp A tagger and the
        # BPI 2020 secondary contrast all rebuild the pyramid from the encoder output.
        sources["p7"] = from_cache("p7")
        sources["raw_t2"] = from_cache("raw")
        sources["tagger"] = from_cache("tagger")
        if log != "bpi2020":
            # Frozen full-benchmark matrices used by the 5,000-replicate review run and
            # by Supp A. P7 is bit-identical to the rebuilt one; raw differs by <1e-6.
            sources["p7"] = lambda: cosine(np.load(bench / "p7_ordered_embeddings.npy"))
            sources["raw"] = lambda: cosine(np.load(bench / "raw_ordered_embeddings.npy"))
    sources["a2v16"] = lambda: cosine(rebuilt_matrix(log, "a2v16", variants, sequences))
    if log != "bpi2012":
        sources["ngram"] = lambda: cosine(ngram_tfidf_features(variants).matrix)
        topic_dir = EVAL / "supp_trace_topics_v1" / "assignments" / log

        def topic(key, metric):
            def build():
                data = np.load(topic_dir / "topic_weights.npz", allow_pickle=True)
                ids = [str(x) for x in data["variant_ids"]]
                if ids != variants["variant_id"].tolist():
                    raise SystemExit(f"{log}: topic weight order differs")
                weights = data[key]
                if metric == "cosine":
                    return pairwise_distances(csr_matrix(normalize(weights, norm="l2")), metric="cosine")
                return pairwise_distances(csr_matrix(weights), metric="euclidean")
            return build
        sources["topic_stt_nmf"] = topic("stt_nmf", "euclidean")
        sources["topic_stt_nmf_traceprism"] = topic("stt_nmf_traceprism", "euclidean")
        sources["topic_stt_nmf_cosine"] = topic("stt_nmf", "cosine")
    return sources


def partitions(log: str) -> list[dict]:
    """Every audited partition: name, distance source, assignment file, bootstrap replicates."""
    variants, bench = log_frame(log)
    order = variants["variant_id"].tolist()
    rows = []

    def add(name, source, path, reps, family, published=None):
        rows.append({
            "log": log, "partition": name, "distance": source, "path": path,
            "labels": labels_from(path, order), "replicates": reps,
            "family": family, "published_silhouette": published,
        })

    if log == "bpi2012":
        src = bench / "sources"
        bench_csv = pd.read_csv(src / "activity_source_benchmark.csv").set_index("method")
        for name, source, method in (
            ("p7_matched_k8", "p7", "p7_llm_semantic_bge"),
            ("raw_matched_k8", "raw", "raw_activity_label_bge"),
            ("a2v16_matched_k8", "a2v16", "activity2vec_log_context_d16"),
        ):
            add(name, source, src / f"assignments_{method}_k8.parquet", SUPP_RESAMPLES,
                "supp_extra_logs_bpi2012", float(bench_csv.loc[method, "matched_case_weighted_silhouette"]))
        return rows

    k = MATCHED_K[log]
    suffix = SUFFIX[log]
    rs = EVAL / "representation_sources" / log
    bench_csv = pd.read_csv(EVAL / "representation_sources/activity_source_benchmark.csv")
    bench_csv = bench_csv.loc[bench_csv["dataset"].eq(log)].set_index("method")
    pub = lambda m: float(bench_csv.loc[m, "matched_case_weighted_silhouette"])  # noqa: E731
    folder = "representation_sources" if log in ("bpi2020", "sepsis", "rtfm") else "representation_sources_extra"
    intervals = pd.read_csv(EVAL / folder / "source_silhouette_intervals.csv")
    intervals = intervals.loc[intervals["dataset"].eq(log)].set_index("method")
    ipub = lambda m: float(intervals.loc[m, "silhouette_point_estimate"])  # noqa: E731
    # Partitions behind the Table 2 intervals: dense refit as in run_representation_uncertainty.py
    # bpi2017 P7: the local dense refit moves one variant (silhouette 0.628121); the saved
    # assignment reproduces the published interval point (0.628116195) to 1e-9, so use it.
    p7_file = (rs / f"assignments_p7_llm_semantic_bge_k{k}.parquet" if log == "bpi2017"
               else REFIT / f"{log}__p7_table2_k{k}.parquet")
    add(f"p7_table2_k{k}", "p7", p7_file, RESAMPLES, "table2_interval", ipub("p7_llm_semantic_bge"))
    add(f"raw_table2_k{k}", "raw_t2", REFIT / f"{log}__raw_table2_k{k}.parquet",
        RESAMPLES, "table2_interval", ipub("raw_activity_label_bge"))
    add(f"a2v16_table2_k{k}", "a2v16", REFIT / f"{log}__a2v16_table2_k{k}.parquet",
        TABLE2_RESAMPLES, "table2_interval", ipub("activity2vec_log_context_d16"))
    # Saved assignments behind the Table 2 silhouette columns (sparse-input fit)
    add(f"p7_saved_k{k}", "p7", rs / f"assignments_p7_llm_semantic_bge_k{k}.parquet",
        0, "table2_column", pub("p7_llm_semantic_bge"))
    add(f"raw_saved_k{k}", "raw_t2", rs / f"assignments_raw_activity_label_bge_k{k}.parquet",
        0, "table2_column", pub("raw_activity_label_bge"))
    add(f"a2v16_saved_k{k}", "a2v16", rs / f"assignments_activity2vec_log_context_d16_k{k}.parquet",
        0, "table2_column", pub("activity2vec_log_context_d16"))
    dyn = pd.read_csv(bench / ("metrics_fixed_k8.csv" if suffix == "k8" else "metrics_dynamic_k.csv"))
    dyn = dyn.set_index("method")
    if log != "bpi2020":
        add(f"p7_fullbench_{suffix}", "p7", bench / f"assignments_{P7_BENCH}_{suffix}.parquet",
            RESAMPLES, "review_5000_and_suppA", float(dyn.loc[P7_BENCH, "case_weighted_silhouette"]))
        add(f"raw_fullbench_{suffix}", "raw", bench / f"assignments_{RAW_BENCH}_{suffix}.parquet",
            RESAMPLES, "review_5000_and_suppA", float(dyn.loc[RAW_BENCH, "case_weighted_silhouette"]))
    ngram_pub = float(dyn.loc["ngram_tfidf_1_3", "case_weighted_silhouette"])
    add("ngram_own_k", "ngram", bench / f"assignments_ngram_tfidf_1_3_{suffix}.parquet", 0,
        "table2", ngram_pub)
    tagger_pub = pd.read_csv(EVAL / "supp_semantic_tagger_v1/partition_metrics.csv").set_index("dataset")
    tagger_file = EVAL / f"supp_semantic_tagger_v1/assignments/assignments_p7_tagger_{log}_k{k}.parquet"
    if log == "bpi2020":
        # the BPI 2020 secondary contrast refits the tagger partition on the dense matrix
        add(f"tagger_k{k}", "tagger", REFIT / f"{log}__tagger_k{k}.parquet", RESAMPLES,
            "suppA_tagger", float(tagger_pub.loc[log, "tagger_silhouette"]))
        add(f"tagger_saved_k{k}", "tagger", tagger_file, 0, "suppA_tagger",
            float(tagger_pub.loc[log, "tagger_silhouette"]))
    else:
        add(f"tagger_k{k}", "tagger", tagger_file, RESAMPLES, "suppA_tagger",
            float(tagger_pub.loc[log, "tagger_silhouette"]))
    topic_dir = EVAL / "supp_trace_topics_v1" / "assignments" / log
    topic_pub = pd.read_csv(EVAL / f"supp_trace_topics_v1/per_log/{log}_silhouette.csv").set_index("method")
    for name, source, file, method in (
        (f"topic_stt_nmf_k{k}", "topic_stt_nmf", f"assignments_stt_nmf_k{k}.parquet", "stt_nmf"),
        (f"topic_stt_nmf_traceprism_k{k}", "topic_stt_nmf_traceprism",
         f"assignments_stt_nmf_traceprism_k{k}.parquet", "stt_nmf_traceprism"),
        (f"topic_stt_nmf_cosine_k{k}", "topic_stt_nmf_cosine",
         f"assignments_stt_nmf_cosine_k{k}.parquet", "stt_nmf_cosine"),
        ("topic_stt_nmf_dynamic", "topic_stt_nmf", "assignments_stt_nmf_dynamic.parquet", "stt_nmf_dynamic"),
    ):
        path = topic_dir / file
        if path.exists():
            published = float(topic_pub.loc[method, "case_weighted_silhouette"]) if method in topic_pub.index else None
            add(name, source, path, 0, "suppB_topics", published)
    return rows


# --------------------------------------------------------------------------
# embed (isolated subprocess: torch is only imported here)
# --------------------------------------------------------------------------


def embed(_: argparse.Namespace) -> None:
    from run_representation_source_benchmark import _embed

    CACHE.mkdir(parents=True, exist_ok=True)
    tagged = pd.read_parquet(EVAL / "supp_semantic_tagger_v1/tagger/activity_semantics.parquet")
    for log, _, _ in LOGS:
        target = CACHE / f"activity_vectors_{log}.npz"
        if target.exists():
            print(f"{log}: cached", flush=True)
            continue
        labels, descriptors = _activity_index(EVAL / "full_benchmark" / log)
        frame = tagged.loc[tagged["dataset"].eq(log)]
        tagger_desc = dict(zip(frame["activity_label"].astype(str), frame["canonical_descriptor"].astype(str)))
        out = {
            "labels": np.asarray(labels),
            "p7": _embed(descriptors, ROOT / "models/huggingface"),
            "raw": _embed(labels, ROOT / "models/huggingface"),
            "tagger": _embed([tagger_desc[x] for x in labels], ROOT / "models/huggingface"),
        }
        np.savez(target, **out)
        print(f"{log}: embedded {len(labels)} activities", flush=True)


# --------------------------------------------------------------------------
# audit: point estimates under both conventions
# --------------------------------------------------------------------------


def _degenerate_sepsis() -> dict:
    """Re-create the Sepsis only_state/only_role ablation partition without an encoder.

    One distinct descriptor gives one activity vector, so the compact activity space
    has rank 1 and every activity maps to the same scalar; any constant unit vector
    reproduces the geometry.
    """
    from llm_variant_clustering.clustering import weighted_kmeans
    from llm_variant_clustering.embeddings import embedding_feature_set

    variants, bench = log_frame("sepsis")
    sequences = parse_sequences(variants)
    weights = variants["frequency"].to_numpy(dtype=np.float64)
    labels, _ = _activity_index(bench)
    vectors = np.tile(np.eye(1, 1024, 0, dtype=np.float32), (len(labels), 1))
    matrix = _pyramid(sequences, labels, vectors)
    features = embedding_feature_set("only_state", variants, matrix)
    distances = pairwise_distances(features.matrix, metric=features.metric)
    result = weighted_kmeans(features.matrix, weights, n_clusters=8)
    old, new, min_weight = silhouette_old_new(distances, result.labels, weights)
    frozen_p7 = labels_from(bench / f"assignments_{P7_BENCH}_k8.parquet", variants["variant_id"].tolist())
    cluster_weights = pd.Series(weights).groupby(result.labels).sum()
    lengths = pd.Series([len(s) for s in sequences])
    small = cluster_weights.idxmin()
    return {
        "log": "sepsis", "partition": "ablation_only_state_or_only_role_k8",
        "family": "descriptor_field_ablation", "distance": "rank1_constant_activity_vector",
        "n_variants": len(weights), "n_clusters": int(np.unique(result.labels).size),
        "occupied_clusters": int(cluster_weights.size),
        "min_cluster_case_weight": min_weight,
        "clusters_with_weight_le_1": int((cluster_weights <= 1).sum()),
        "min_cluster_variants": int(pd.Series(result.labels).value_counts().min()),
        "old_silhouette": old, "new_silhouette": new,
        "published_silhouette": 0.9999992245395636,
        "ari_vs_frozen_p7": float(adjusted_rand_score(frozen_p7, result.labels)),
        "note": (
            f"smallest cluster: {int((result.labels == small).sum())} variant(s), "
            f"case weight {cluster_weights[small]:.0f}, sequence lengths "
            f"{sorted(set(lengths[result.labels == small]))}; distinct pyramid rows "
            f"{np.unique(np.round(np.asarray(matrix), 6), axis=0).shape[0]}"
        ),
    }


def refit(args: argparse.Namespace) -> None:
    """Dense weighted k-means refits exactly as run_representation_uncertainty.py
    (and, for the BPI 2020 tagger, _bpi2020_secondary in run_supp_semantic_tagger.py)."""
    from llm_variant_clustering.clustering import weighted_kmeans

    REFIT.mkdir(parents=True, exist_ok=True)
    for log in args.logs:
        variants, _ = log_frame(log)
        sequences = parse_sequences(variants)
        weights = variants["frequency"].to_numpy(dtype=np.float64)
        k = MATCHED_K[log]
        keys = [("p7", "p7"), ("raw", "raw"), ("a2v16", "a2v16")] + ([("tagger", "tagger")] if log == "bpi2020" else [])
        for name, key in keys:
            target = REFIT / (f"{log}__{name}_k{k}.parquet" if name == "tagger" else f"{log}__{name}_table2_k{k}.parquet")
            if target.exists():
                continue
            started = time.time()
            matrix = rebuilt_matrix(log, key, variants, sequences)
            labels = weighted_kmeans(matrix, weights, n_clusters=k).labels
            pd.DataFrame({"variant_id": variants["variant_id"], "cluster_label": labels.astype("int32")}).to_parquet(
                target, index=False)
            print(f"{log} {name}: refit K={k} in {time.time() - started:.0f}s", flush=True)


def audit(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, agreement = [], []
    for log in args.logs:
        variants, _ = log_frame(log)
        weights = variants["frequency"].to_numpy(dtype=np.float64)
        parts = partitions(log)
        sources = distance_sources(log)
        for source in dict.fromkeys(p["distance"] for p in parts):
            started = time.time()
            distances = sources[source]()
            for part in [p for p in parts if p["distance"] == source]:
                labels = part["labels"]
                old, new, min_weight = silhouette_old_new(distances, labels, weights)
                library = frequency_weighted_silhouette(distances, labels, weights)
                if abs(library - new) > 1e-12:
                    raise SystemExit(f"{log} {part['partition']}: copy != library ({new} vs {library})")
                cluster_weights = pd.Series(weights).groupby(labels).sum()
                counts = pd.Series(labels).value_counts()
                pub = part["published_silhouette"]
                rows.append({
                    "log": log, "partition": part["partition"], "family": part["family"],
                    "distance": source, "n_variants": len(weights),
                    "n_clusters": int(np.unique(labels).size),
                    "min_cluster_case_weight": min_weight,
                    "clusters_with_weight_le_1": int((cluster_weights <= 1).sum()),
                    "min_cluster_variants": int(counts.min()),
                    "single_variant_clusters": int((counts == 1).sum()),
                    "old_silhouette": old, "new_silhouette": new,
                    "new_minus_old": new - old,
                    "published_silhouette": pub,
                    "old_minus_published": None if pub is None else old - pub,
                    "assignment_file": str(part["path"].relative_to(ROOT)),
                })
                print(f"{log:8s} {part['partition']:34s} K={rows[-1]['n_clusters']} "
                      f"minw={min_weight:g} old={old:.6f} new={new:.6f} pub={pub}", flush=True)
            del distances
            print(f"  {log} {source}: {time.time() - started:.0f}s", flush=True)
        by_name = {p["partition"]: p["labels"] for p in parts}
        if log != "bpi2012":
            k = MATCHED_K[log]
            s = SUFFIX[log]
            pairs = [
                (f"raw_saved_k{k}", f"raw_table2_k{k}"),
                (f"p7_saved_k{k}", f"p7_table2_k{k}"),
                (f"a2v16_saved_k{k}", f"a2v16_table2_k{k}"),
            ]
            if log == "bpi2020":
                pairs.append((f"tagger_saved_k{k}", f"tagger_k{k}"))
            else:
                pairs += [
                    (f"raw_fullbench_{s}", f"raw_table2_k{k}"),
                    (f"raw_fullbench_{s}", f"raw_saved_k{k}"),
                    (f"p7_fullbench_{s}", f"p7_table2_k{k}"),
                    (f"tagger_k{k}", f"p7_fullbench_{s}"),
                ]
            for first, second in pairs:
                a, b = by_name[first], by_name[second]
                agreement.append({
                    "log": log, "first": first, "second": second,
                    "k_first": int(np.unique(a).size), "k_second": int(np.unique(b).size),
                    "ari": float(adjusted_rand_score(a, b)),
                    "variants_reassigned": int(len(a) - pd.crosstab(a, b).max(axis=1).sum()),
                    "identical_partition": bool(np.array_equal(_canonical(a), _canonical(b))),
                })
            if log == "bpi2017":
                refit_p7 = labels_from(REFIT / f"{log}__p7_table2_k{k}.parquet", variants["variant_id"].tolist())
                saved = by_name[f"p7_saved_k{k}"]
                agreement.append({
                    "log": log, "first": f"p7_saved_k{k}", "second": "p7 dense refit on this machine (not used)",
                    "k_first": int(np.unique(saved).size), "k_second": int(np.unique(refit_p7).size),
                    "ari": float(adjusted_rand_score(saved, refit_p7)),
                    "variants_reassigned": int(len(saved) - pd.crosstab(saved, refit_p7).max(axis=1).sum()),
                    "identical_partition": bool(np.array_equal(_canonical(saved), _canonical(refit_p7))),
                })
        if log == "bpi2020":
            other = labels_from(
                EVAL / "representation_sources_bpi2020/bpi2020/assignments_raw_activity_label_bge_k9.parquet",
                variants["variant_id"].tolist(),
            )
            agreement.append({
                "log": log, "first": "representation_sources_bpi2020 raw k9", "second": "raw_saved_k9",
                "k_first": int(np.unique(other).size), "k_second": int(np.unique(by_name["raw_saved_k9"]).size),
                "ari": float(adjusted_rand_score(other, by_name["raw_saved_k9"])),
                "identical_partition": bool(np.array_equal(_canonical(other), _canonical(by_name["raw_saved_k9"]))),
            })
    if "sepsis" in args.logs:
        rows.append(_degenerate_sepsis())
    frame = pd.DataFrame(rows)
    name = "audit_point_estimates.csv" if args.logs == ALL_LOGS else f"audit_point_estimates_{'_'.join(args.logs)}.csv"
    frame.to_csv(OUT / name, index=False)
    if agreement:
        pd.DataFrame(agreement).to_csv(OUT / name.replace("audit_point_estimates", "partition_identity"), index=False)
    print(frame.drop(columns=["assignment_file"], errors="ignore").round(6).to_string(index=False))


# --------------------------------------------------------------------------
# bootstrap: per-partition replicate vectors under both conventions
# --------------------------------------------------------------------------


def _init(paths: dict) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    _STATE["distances"] = np.load(paths["distances"], mmap_mode="r")
    _STATE["labels"] = {k: np.load(v, allow_pickle=True) for k, v in paths["labels"].items()}
    _STATE["reps"] = paths["reps"]
    _STATE["weights"] = np.load(paths["weights"])


def _chunk(task):
    start, draws = task
    weights = _STATE["weights"]
    out = []
    for offset, indices in enumerate(draws):
        row = start + offset
        needed = [k for k, r in _STATE["reps"].items() if row < r]
        valid = {k: np.unique(_STATE["labels"][k][indices]).size >= 2 for k in needed}
        sub = None
        if any(valid.values()):
            sub = _STATE["distances"][np.ix_(indices, indices)]
        record = {}
        for key in needed:
            if not valid[key]:
                record[key] = (np.nan, np.nan, np.nan)
                continue
            record[key] = silhouette_old_new(sub, _STATE["labels"][key][indices], weights[indices])
        out.append((row, record))
    return out


def _canonical(labels: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(labels, return_inverse=True)
    first_seen = {}
    return np.asarray([first_seen.setdefault(x, len(first_seen)) for x in inverse])


def bootstrap(args: argparse.Namespace) -> None:
    REPS.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for log in args.logs:
        variants, _ = log_frame(log)
        weights = variants["frequency"].to_numpy(dtype=np.float64)
        n = len(weights)
        cap = CAP if n > CAP else None
        parts = [p for p in partitions(log) if p["replicates"] > 0]
        sources = distance_sources(log)
        max_reps = max(p["replicates"] for p in parts)
        draws = _resample_indices(n, max_reps, SEED, cap)
        for source in dict.fromkeys(p["distance"] for p in parts):
            group = [p for p in parts if p["distance"] == source]
            todo, aliases = [], {}
            for part in group:
                target = REPS / f"{log}__{part['partition']}.npz"
                if target.exists():
                    continue
                twin = next((t for t in todo if np.array_equal(_canonical(t["labels"]), _canonical(part["labels"]))
                             and t["replicates"] >= part["replicates"]), None)
                if twin is not None:
                    aliases[part["partition"]] = twin["partition"]
                else:
                    todo.append(part)
            if not todo:
                continue
            started = time.time()
            distances = sources[source]()
            stem = SCRATCH / f"{log}_{source}"
            np.save(f"{stem}_distances.npy", distances)
            np.save(f"{stem}_weights.npy", weights)
            label_paths = {}
            for part in todo:
                np.save(f"{stem}_{part['partition']}_labels.npy", part["labels"])
                label_paths[part["partition"]] = f"{stem}_{part['partition']}_labels.npy"
            # one serial check: the copy's "new" equals the library on the first draws
            for part in todo:
                for indices in draws[:3]:
                    labels = part["labels"][indices]
                    if np.unique(labels).size < 2:
                        continue
                    sub = distances[np.ix_(indices, indices)]
                    lib = frequency_weighted_silhouette(sub, labels, weights[indices])
                    if abs(lib - silhouette_old_new(sub, labels, weights[indices])[1]) > 1e-12:
                        raise SystemExit("copy differs from library on a bootstrap draw")
            del distances
            paths = {
                "distances": f"{stem}_distances.npy", "weights": f"{stem}_weights.npy",
                "labels": label_paths, "reps": {p["partition"]: p["replicates"] for p in todo},
            }
            reps = max(p["replicates"] for p in todo)
            chunk = 25
            tasks = [(s, draws[s: min(s + chunk, reps)]) for s in range(0, reps, chunk)]
            results = {p["partition"]: np.full((p["replicates"], 3), np.nan) for p in todo}
            print(f"{log} {source}: n={n} m={cap or n} reps={reps} partitions={list(results)} "
                  f"aliases={aliases}", flush=True)
            with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(paths,)) as pool:
                for done, part in enumerate(pool.map(_chunk, tasks), start=1):
                    for row, record in part:
                        for key, value in record.items():
                            results[key][row] = value
                    if done % max(1, len(tasks) // 10) == 0:
                        print(f"  {log} {source}: {done}/{len(tasks)} chunks "
                              f"{time.time() - started:.0f}s", flush=True)
            for part in group:
                key = aliases.get(part["partition"], part["partition"])
                if key not in results:
                    continue
                values = results[key][: part["replicates"]]
                np.savez(
                    REPS / f"{log}__{part['partition']}.npz",
                    old=values[:, 0], new=values[:, 1], min_cluster_weight=values[:, 2],
                    valid=~np.isnan(values[:, 0]), sample_size=cap or n, seed=SEED,
                    computed_as=key,
                )
            for path in SCRATCH.glob(f"{log}_{source}_*"):
                path.unlink()
            print(f"  {log} {source}: done in {time.time() - started:.0f}s", flush=True)


# --------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------


def _load_reps(log: str, partition: str) -> dict:
    data = np.load(REPS / f"{log}__{partition}.npz")
    return {k: data[k] for k in data.files}


def contrast(log, first, second, reps, convention, point_first, point_second, bonferroni=True):
    a, b = _load_reps(log, first), _load_reps(log, second)
    valid = a["valid"][:reps] & b["valid"][:reps]
    diff = (a[convention][:reps] - b[convention][:reps])[valid]
    single = ((a["min_cluster_weight"][:reps] <= 1) | (b["min_cluster_weight"][:reps] <= 1))[valid]
    point = point_first - point_second
    nominal = _percentile_interval(point, diff, CONFIDENCE)
    record = {
        "log": log, "first": first, "second": second, "convention": convention,
        "replicates_requested": reps, "valid_replicates": int(diff.size),
        "skipped_replicates": int(reps - diff.size),
        "replicates_with_cluster_weight_le_1": int(single.sum()),
        "point_estimate": point,
        "ci95_lower": nominal.lower, "ci95_upper": nominal.upper,
        "share_gt_0": float(np.mean(diff > 0)),
    }
    if bonferroni:
        adjusted = _percentile_interval(point, diff, BONFERRONI)
        record.update(bonf9875_lower=adjusted.lower, bonf9875_upper=adjusted.upper)
    return record, diff


ALL_LOGS = ["bpi2020", "bpi2017", "bpi2019", "sepsis", "rtfm", "bpi2012"]


def summarize(_: argparse.Namespace) -> None:
    audit_frame = pd.read_csv(OUT / "audit_point_estimates.csv")
    point = audit_frame.set_index(["log", "partition"])[["old_silhouette", "new_silhouette"]]

    def p(log, part, convention):
        return float(point.loc[(log, part), f"{convention}_silhouette"])

    rows = []
    # Task 2a: exact Table 2 partitions, 5,000 replicates
    for log in ("bpi2017", "bpi2019", "sepsis", "rtfm", "bpi2020"):
        k = MATCHED_K[log]
        for conv in ("old", "new"):
            for reps in (RESAMPLES, TABLE2_RESAMPLES):
                for second in (f"raw_table2_k{k}", f"a2v16_table2_k{k}"):
                    if second.startswith("a2v") and reps != TABLE2_RESAMPLES:
                        continue
                    rec, _ = contrast(log, f"p7_table2_k{k}", second, reps, conv,
                                      p(log, f"p7_table2_k{k}", conv), p(log, second, conv))
                    rec["analysis"] = ("task2_exact_table2_5000" if reps == RESAMPLES
                                       else "task2_table2_nominal_300")
                    rec["role"] = "secondary" if log == "bpi2020" else "family"
                    rows.append(rec)
        if log != "bpi2020":
            for conv in ("old", "new"):
                rec, _ = contrast(log, f"p7_fullbench_{SUFFIX[log]}", f"raw_fullbench_{SUFFIX[log]}",
                                  RESAMPLES, conv, p(log, f"p7_fullbench_{SUFFIX[log]}", conv),
                                  p(log, f"raw_fullbench_{SUFFIX[log]}", conv))
                rec["analysis"] = "review_bootstrap_tails_v1_partitions_5000"
                rec["role"] = "family"
                rows.append(rec)
    # Task 3: tagger contrasts with Supp A partitions
    for log in ("bpi2017", "bpi2019", "sepsis", "rtfm", "bpi2020"):
        k = MATCHED_K[log]
        if log == "bpi2020":
            pairs = [("tagger_minus_raw", f"raw_table2_k{k}"), ("tagger_minus_p7", f"p7_table2_k{k}")]
            published_reps = TABLE2_RESAMPLES
        else:
            pairs = [("tagger_minus_raw", f"raw_fullbench_{SUFFIX[log]}"),
                     ("tagger_minus_p7", f"p7_fullbench_{SUFFIX[log]}")]
            published_reps = SUPP_RESAMPLES
        for name, second in pairs:
            for conv in ("old", "new"):
                for reps in (RESAMPLES, published_reps):
                    rec, _ = contrast(log, f"tagger_k{k}", second, reps, conv,
                                      p(log, f"tagger_k{k}", conv), p(log, second, conv))
                    rec["analysis"] = f"task3_{name}_{reps}"
                    rec["role"] = "secondary" if log == "bpi2020" else "family"
                    rows.append(rec)
    # BPI 2012 (Supp extra logs)
    for conv in ("old", "new"):
        for second in ("raw_matched_k8", "a2v16_matched_k8"):
            rec, _ = contrast("bpi2012", "p7_matched_k8", second, SUPP_RESAMPLES, conv,
                              p("bpi2012", "p7_matched_k8", conv), p("bpi2012", second, conv))
            rec["analysis"] = "supp_extra_logs_bpi2012_400"
            rec["role"] = "family"
            rows.append(rec)
    frame = pd.DataFrame(rows)
    lead = ["analysis", "role", "log", "first", "second", "convention", "replicates_requested"]
    frame = frame[lead + [c for c in frame.columns if c not in lead]]
    frame.to_csv(OUT / "bootstrap_contrasts_all.csv", index=False)

    # single-case-cluster prevalence per partition
    prevalence = []
    for path in sorted(REPS.glob("*.npz")):
        log, partition = path.stem.split("__")
        data = _load_reps(log, partition)
        valid = data["valid"]
        prevalence.append({
            "log": log, "partition": partition, "replicates": int(valid.size),
            "valid": int(valid.sum()),
            "replicates_with_cluster_weight_le_1": int((data["min_cluster_weight"][valid] <= 1).sum()),
            "max_abs_new_minus_old": float(np.nanmax(np.abs(data["new"] - data["old"]))) if valid.any() else 0.0,
            "computed_as": str(data["computed_as"]),
        })
    pd.DataFrame(prevalence).to_csv(OUT / "bootstrap_singleton_prevalence.csv", index=False)
    pd.set_option("display.width", 250)
    print(frame.round(5).to_string(index=False))
    print(pd.DataFrame(prevalence).to_string(index=False))


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest(_: argparse.Namespace) -> None:
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "manifest.json")
    extra = [ROOT / "scripts/run_review3_fixes.py", ROOT / "src/llm_variant_clustering/evaluation.py",
             ROOT / "tests/test_silhouette_singleton.py"]
    import sklearn
    record = {
        "schema_version": 1,
        "created_by": "scripts/run_review3_fixes.py",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "bootstrap": {"seed": SEED, "cap_variants": CAP, "replicates": RESAMPLES,
                      "table2_replicates": TABLE2_RESAMPLES, "supp_replicates": SUPP_RESAMPLES,
                      "confidence": CONFIDENCE, "bonferroni_confidence": BONFERRONI},
        "threads": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS")},
        "outputs": {str(p.relative_to(OUT)).replace("\\", "/"): {"sha256": _sha(p), "bytes": p.stat().st_size}
                    for p in files},
        "code": {str(p.relative_to(ROOT)).replace("\\", "/"): _sha(p) for p in extra},
    }
    (OUT / "manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {len(files)} outputs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("embed")
    rf = sub.add_parser("refit")
    rf.add_argument("--logs", nargs="+", default=[log for log, _, _ in LOGS])
    a = sub.add_parser("audit")
    a.add_argument("--logs", nargs="+", default=ALL_LOGS)
    b = sub.add_parser("bootstrap")
    b.add_argument("--logs", nargs="+", default=ALL_LOGS)
    b.add_argument("--workers", type=int, default=2)
    sub.add_parser("summarize")
    sub.add_parser("manifest")
    args = parser.parse_args()
    {"embed": embed, "refit": refit, "audit": audit, "bootstrap": bootstrap,
     "summarize": summarize, "manifest": manifest}[args.command](args)


if __name__ == "__main__":
    main()
