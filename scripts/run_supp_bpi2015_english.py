"""Supplementary follow-up: BPI 2015 Municipality 1 with English activity names.

New, review-motivated work (see evaluation/supp_bpi2015_english_v1/PRESPEC.md).
The original code-only result of evaluation/supp_extra_logs_v1 is preserved
and not modified. Activity identity stays the ``concept:name`` code, so the
case and variant set equal Supp C; only the text given to the text-based
methods changes from the code to its event-modal ``activityNameEN``.

Stages: prepare, normalize, tag, cluster, heldout, aggregate, manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# The override exists only for code-only smoke tests in a scratch folder.
OUT = Path(os.environ.get("SUPP_BPI2015_EN_OUT", ROOT / "evaluation/supp_bpi2015_english_v1"))
SUPP_C = ROOT / "evaluation/supp_extra_logs_v1/bpi2015_m1"
FROZEN = ROOT / "configs/frozen/p7_activity_gpt41_v1"
DYNAMIC_CONFIG = ROOT / "configs/clustering/dynamic_k_v1.json"
DATA = ROOT / "datasets/processed/bpi2015_municipality1"
XES = ROOT / "datasets/raw/bpi2015_municipality1/BPIC15_1.xes"
VARIANTS = DATA / "variants.parquet"
CASE_MAP = DATA / "case_variant_map.parquet"

ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
DEPLOYMENT = "gpt-4.1"
MODEL_VERSION = "2025-04-14"
HF_CACHE = ROOT / "models/huggingface"

CODE_KEY = "concept:name"
ENGLISH_KEY = "activityNameEN"
CASE_KEY = "case:concept:name"

P7_EN = "p7_english_pyramid_bge"
RAW_EN = "raw_english_label_pyramid_bge"
TAGGER_EN = "tagger_english_pyramid_bge"
RAW_CODE = "raw_code_label_pyramid_bge"
A2V = "activity2vec_log_context_d16"
NGRAM = "ngram_tfidf_1_3"
METHODS = (P7_EN, RAW_EN, TAGGER_EN, RAW_CODE, A2V, NGRAM)
TEXT_METHODS = (P7_EN, RAW_EN, TAGGER_EN)
REPLAY_METHODS = (P7_EN, RAW_EN, RAW_CODE, TAGGER_EN)
WHOLE_LOG = "whole_log"

MATCHED_K = 8
RESAMPLES = 400
SEED_RESAMPLES = 4000
F1_RESAMPLES = 2000
CONFIDENCE = 0.95
FAMILY = (("Q2", P7_EN, RAW_EN), ("Q3", TAGGER_EN, RAW_EN))
FAMILY_SIZE = len(FAMILY)
DESCRIPTIVE_PAIRS = (
    ("Q1_own_space_not_comparable", RAW_EN, RAW_CODE),
    ("descriptive", P7_EN, RAW_CODE),
    ("descriptive", P7_EN, A2V),
    ("descriptive", P7_EN, NGRAM),
    ("descriptive", P7_EN, TAGGER_EN),
)
REFERENCE_GEOMETRIES = (A2V, NGRAM)
REFERENCE_PAIRS = (
    ("Q1", RAW_EN, RAW_CODE),
    ("Q2_reference", P7_EN, RAW_EN),
    ("Q3_reference", TAGGER_EN, RAW_EN),
)
REPLAY_CAP_SECONDS = 4 * 3600
NOISE_THRESHOLD = 0.2
HASH_SEED = "0"
CHUNK_ROWS = 256
FIELDS = ("action", "business_object", "process_stage", "explicit_state", "explicit_role")
EMPTY_VALUES = {"explicit_role": "none explicit"}
# Approximate list price for GPT-4.1 (USD per 1M tokens), used only for a cost estimate.
PRICE_PER_M = {"prompt": 2.0, "completion": 8.0}


# --------------------------------------------------------------------------
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------


def modal_english_names(events: pd.DataFrame) -> pd.DataFrame:
    """Per code: event-modal English name, ties broken lexicographically.

    Missing or blank English names fall back to the code string (flagged).
    """
    frame = events[[CODE_KEY, ENGLISH_KEY]].copy()
    frame[CODE_KEY] = frame[CODE_KEY].astype(str)
    english = frame[ENGLISH_KEY].astype("string").str.strip()
    frame["_missing"] = english.isna() | english.eq("")
    frame["_english"] = english.where(~frame["_missing"], None)
    rows = []
    for code, group in frame.groupby(CODE_KEY, sort=True):
        names = Counter(group.loc[~group["_missing"], "_english"].astype(str))
        if names:
            top = max(names.values())
            modal = sorted(name for name, count in names.items() if count == top)[0]
            fallback = False
        else:
            modal, top, fallback = code, 0, True
        rows.append({
            "code": code,
            "english_text": modal,
            "events": int(len(group)),
            "events_missing_english": int(group["_missing"].sum()),
            "events_with_modal_name": int(top),
            "distinct_english_names": len(names),
            "tie_at_mode": sum(count == top for count in names.values()) > 1 if names else False,
            "fallback_to_code": fallback,
            "english_names_json": json.dumps(dict(sorted(names.items())), ensure_ascii=False),
        })
    result = pd.DataFrame(rows)
    shared = result.groupby("english_text")["code"].transform("size")
    result["codes_sharing_text"] = shared.astype(int)
    return result


def field_fill_rates(semantics: dict[str, dict], weights: dict[str, int] | None = None) -> dict:
    """Share of descriptors whose field is informative (not unknown / none explicit)."""
    keys = sorted(semantics)
    w = np.array([1.0 if weights is None else float(weights[k]) for k in keys])
    total = w.sum()
    out: dict[str, float] = {"units": len(keys)}
    for field in FIELDS:
        empty = EMPTY_VALUES.get(field, "unknown")
        values = [str(semantics[k][field]).strip().lower() for k in keys]
        filled = np.array([v not in (empty, "unknown") for v in values], dtype=float)
        out[f"{field}_fill"] = float((filled * w).sum() / total)
        if field in ("process_stage", "explicit_state"):
            other = np.array([v == "other" for v in values], dtype=float)
            out[f"{field}_other"] = float((other * w).sum() / total)
    ambiguity = np.array(
        [str(semantics[k]["ambiguity"]).strip().lower() == "none" for k in keys], dtype=float
    )
    out["ambiguity_none"] = float((ambiguity * w).sum() / total)
    all_unknown = np.array(
        [
            str(semantics[k]["action"]).lower() == "unknown"
            and str(semantics[k]["business_object"]).lower() == "unknown"
            for k in keys
        ],
        dtype=float,
    )
    out["action_and_object_unknown"] = float((all_unknown * w).sum() / total)
    return out


def chunked_distances(matrix, metric: str = "cosine") -> np.ndarray:
    """Full pairwise distance matrix assembled from row chunks (bounded peak memory)."""
    from sklearn.metrics import pairwise_distances

    n = matrix.shape[0]
    out = np.empty((n, n), dtype=np.float64)
    for start in range(0, n, CHUNK_ROWS):
        stop = min(n, start + CHUNK_ROWS)
        out[start:stop] = pairwise_distances(matrix[start:stop], matrix, metric=metric)
    np.fill_diagonal(out, 0.0)
    return out


def support_label(lower: float, upper: float, adj_lower: float, adj_upper: float) -> str:
    """PRESPEC wording for Q2/Q3 paired differences."""
    if lower > 0:
        return "higher_robust" if adj_lower > 0 else "higher"
    if upper < 0:
        return "lower_robust" if adj_upper < 0 else "lower"
    return "not_separated"


def q1_reading(f1_difference: float | None, reference_lowers: list[float],
               reference_uppers: list[float]) -> str:
    """PRESPEC Q1 decision: held-out F1 difference plus both reference geometries."""
    if len(reference_lowers) != len(REFERENCE_GEOMETRIES):
        return "not_assessable"
    positive = all(lower > 0 for lower in reference_lowers)
    negative = all(upper < 0 for upper in reference_uppers)
    if f1_difference is None:
        if positive:
            return "partial_positive_f1_not_completed"
        if negative:
            return "partial_negative_f1_not_completed"
        return "partial_mixed_or_not_separated_f1_not_completed"
    if f1_difference >= 0.01 and positive:
        return "consistent_help"
    if f1_difference <= -0.01 and negative:
        return "consistent_against"
    return "mixed_or_not_separated"


# --------------------------------------------------------------------------
# Infrastructure
# --------------------------------------------------------------------------


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def require_prespec() -> str:
    prespec, recorded = OUT / "PRESPEC.md", OUT / "PRESPEC.sha256"
    if not prespec.exists() or not recorded.exists():
        raise SystemExit("PRESPEC.md and PRESPEC.sha256 must exist before this stage")
    actual = sha256(prespec)
    if recorded.read_text(encoding="utf-8").split()[0] != actual:
        raise SystemExit("PRESPEC.md differs from its recorded SHA-256; log changes in AMENDMENTS.md")
    return actual


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "scripts")])
    env["PYTHONHASHSEED"] = HASH_SEED
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env.setdefault("HF_HUB_OFFLINE", "1")
    return env


def _read_events() -> pd.DataFrame:
    import pm4py

    return pm4py.read_xes(str(XES), return_legacy_log_object=False, show_progress_bar=False)


def _variants() -> pd.DataFrame:
    variants = pd.read_parquet(VARIANTS)
    keep = frozenset(pd.read_parquet(SUPP_C / "embedding_index.parquet")["variant_id"].astype(str))
    variants = variants.loc[variants["variant_id"].astype(str).isin(keep)].reset_index(drop=True)
    if len(variants) != len(keep):
        raise ValueError("Variant set differs from Supp C")
    return variants


def _english_map() -> dict[str, str]:
    frame = pd.read_csv(OUT / "data/code_english_map.csv", dtype={"code": str})
    return dict(zip(frame["code"], frame["english_text"].astype(str), strict=True))


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------


def stage_prepare() -> None:
    require_prespec()
    from llm_variant_clustering.features import parse_sequences

    out = OUT / "data"
    out.mkdir(parents=True, exist_ok=True)
    events = _read_events()
    mapping = modal_english_names(events)
    variants = _variants()
    codes = sorted({a for s in parse_sequences(variants) for a in s})
    if sorted(mapping["code"]) != codes:
        raise ValueError("Codes in the XES differ from the codes in the Supp C variants")
    mapping.to_csv(out / "code_english_map.csv", index=False)
    english = dict(zip(mapping["code"], mapping["english_text"], strict=True))
    helper = variants[["variant_id", "frequency"]].copy()
    helper["activity_sequence"] = [
        json.dumps([english[a] for a in s], ensure_ascii=False) for s in parse_sequences(variants)
    ]
    helper.to_parquet(out / "english_variants_for_normalizer.parquet", index=False)
    shared = mapping.loc[mapping["codes_sharing_text"] > 1]
    events_total = int(mapping["events"].sum())
    events_off_mode = int((mapping["events"] - mapping["events_missing_english"]
                           - mapping["events_with_modal_name"]).sum())
    facts = {
        "events": events_total,
        "cases": int(events[CASE_KEY].nunique()),
        "variants": len(variants),
        "codes": len(mapping),
        "distinct_english_texts": int(mapping["english_text"].nunique()),
        "codes_with_multiple_english_names": int((mapping["distinct_english_names"] > 1).sum()),
        "codes_with_tie_at_mode": int(mapping["tie_at_mode"].sum()),
        "events_with_non_modal_english_name": events_off_mode,
        "share_events_non_modal": events_off_mode / events_total,
        "events_missing_english": int(mapping["events_missing_english"].sum()),
        "codes_fallback_to_code": int(mapping["fallback_to_code"].sum()),
        "english_texts_shared_by_multiple_codes": int(shared["english_text"].nunique()),
        "codes_in_shared_texts": len(shared),
        "max_english_text_length": int(mapping["english_text"].str.len().max()),
    }
    _write_json(out / "data_facts.json", facts)
    print(json.dumps(facts, indent=2))


# --------------------------------------------------------------------------
# normalize (frozen GPT-4.1 normaliser, unchanged prompt and schema)
# --------------------------------------------------------------------------


def stage_normalize() -> None:
    require_prespec()
    cmd = [
        sys.executable, "scripts/run_p7_activity_normalization.py",
        str(OUT / "data/english_variants_for_normalizer.parquet"),
        str(OUT / "p7/activity_semantics_english.parquet"),
        "--cache-dir", str(OUT / "azure_activity_cache"),
        "--endpoint", ENDPOINT,
        "--deployment", DEPLOYMENT,
        "--model-version", MODEL_VERSION,
        "--prompt", str(FROZEN / "prompt.txt"),
        "--schema", str(FROZEN / "schema.json"),
    ]
    (OUT / "p7").mkdir(parents=True, exist_ok=True)
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT, env=_env())


# --------------------------------------------------------------------------
# tag (non-LLM Rebmann & van der Aa tagger, unchanged worker and mapping)
# --------------------------------------------------------------------------


def stage_tag() -> None:
    require_prespec()
    import run_supp_semantic_tagger as tagger
    from llm_variant_clustering.semantic_compiler import (
        canonical_activity_text,
        validate_activity_semantics,
    )

    out = OUT / "tagger"
    out.mkdir(parents=True, exist_ok=True)
    texts = sorted(set(_english_map().values()))
    cleaned = {text: tagger.clean_label(text) for text in texts}
    unique = sorted({c for c in cleaned.values() if c.split()})
    request, response = out / "tagger_input.json", out / "tagger_output.json"
    _write_json(request, {"labels": unique})
    tmp = tagger.TAGGER_ROOT / "cache/tmp"
    env = {**os.environ, "PYTHONHASHSEED": tagger.HASH_SEED, "PYTHONPATH": "",
           "TF_CPP_MIN_LOG_LEVEL": "2", "TMP": str(tmp), "TEMP": str(tmp)}
    started = time.time()
    subprocess.run(
        [str(tagger.TAGGER_PYTHON), str(ROOT / "scripts/run_supp_semantic_tagger.py"),
         "_worker", str(request), str(response)],
        check=True, env=env, cwd=str(tagger.TAGGER_SRC),
    )
    seconds = time.time() - started
    result = json.loads(response.read_text(encoding="utf-8"))
    if result["labels"] != unique:
        raise SystemExit("Tagger output does not echo the requested labels")
    tagged = {label: (tags, conf) for label, tags, conf in zip(
        result["labels"], result["tags"], result["confidence_mean_max_logit"])}
    rows = []
    for text, clean in cleaned.items():
        split = clean.split()
        tags, conf = tagged.get(clean, ([], None))
        status, response_json, canonical, error = "validated", None, None, None
        try:
            semantics = tagger.map_to_p7(text, split, list(tags))
            validate_activity_semantics(semantics, schema_path=FROZEN / "schema.json",
                                        expected_activity_label=text)
            response_json = json.dumps(semantics, sort_keys=True)
            canonical = canonical_activity_text(semantics)
        except (ValueError, AssertionError) as exc:
            status, error = "failed", str(exc)
        rows.append({"activity_label": text, "cleaned_label": clean,
                     "tokens_json": json.dumps(split), "tags_json": json.dumps(list(tags)),
                     "confidence_mean_max_logit": conf, "status": status, "error": error,
                     "response_json": response_json, "canonical_descriptor": canonical})
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "activity_semantics_english_tagger.parquet", index=False)
    frame.drop(columns=["response_json"]).to_csv(out / "activity_semantics_english_tagger.csv",
                                                 index=False)
    _write_json(out / "run_manifest.json", {
        "normalizer": "rebmann_vanderaa_2022_bert_tagger", "commit": tagger.TAGGER_COMMIT,
        "model_zip_sha256": tagger.TAGGER_MODEL_ZIP_SHA256, "environment": result["environment"],
        "english_texts": len(texts), "unique_cleaned_labels_tagged": len(unique),
        "validated": int(frame["status"].eq("validated").sum()),
        "tagger_seconds": round(seconds, 1),
    })
    print(frame["status"].value_counts().to_string())


# --------------------------------------------------------------------------
# cluster
# --------------------------------------------------------------------------


def _load_semantics(path: Path) -> tuple[dict[str, dict] | None, list[str]]:
    frame = pd.read_parquet(path)
    failed = frame.loc[~frame["status"].eq("validated"), "activity_label"].astype(str).tolist()
    if failed:
        return None, failed
    return {str(r.activity_label): json.loads(r.response_json) for r in frame.itertuples()}, []


def _feature_set(name, variants, matrix):
    from llm_variant_clustering.embeddings import embedding_feature_set

    return embedding_feature_set(name, variants, matrix)


def _ngram(variants):
    from llm_variant_clustering.features import FeatureSet, ngram_tfidf_features

    ngram = ngram_tfidf_features(variants)
    return FeatureSet(name=ngram.name, matrix=ngram.matrix.astype(np.float32),
                      variant_ids=ngram.variant_ids, feature_names=ngram.feature_names,
                      metric=ngram.metric)


def _fit(features, weights, k):
    from llm_variant_clustering.clustering import weighted_kmeans

    return weighted_kmeans(features.matrix, weights, n_clusters=k)


def _activity_maps(variants, codes: list[str], english: dict[str, str]):
    """Per-code activity vectors for every text-based source; None if a method abstains."""
    from llm_variant_clustering.features import activity_context_embeddings
    from llm_variant_clustering.semantic_compiler import canonical_activity_text
    from run_representation_source_benchmark import _compact_activity_space, _embed

    maps, info = {}, {}
    texts = sorted(set(english.values()))

    def by_text(name: str, strings: list[str]) -> None:
        compact, rank = _compact_activity_space(_embed(strings, HF_CACHE))
        vectors = dict(zip(texts, compact, strict=True))
        maps[name] = {code: vectors[english[code]] for code in codes}
        info[name] = {"status": "computed", "embedded_texts": len(strings),
                      "distinct_embedded_texts": len(set(strings)), "activity_rank": rank}

    by_text(RAW_EN, texts)
    p7, failed = _load_semantics(OUT / "p7/activity_semantics_english.parquet")
    if p7 is None:
        info[P7_EN] = {"status": "abstained_unvalidated_descriptors", "failed": failed}
    else:
        by_text(P7_EN, [canonical_activity_text(p7[t]) for t in texts])
    tag, failed = _load_semantics(OUT / "tagger/activity_semantics_english_tagger.parquet")
    if tag is None:
        info[TAGGER_EN] = {"status": "abstained_incomplete_coverage", "failed": failed}
    else:
        by_text(TAGGER_EN, [canonical_activity_text(tag[t]) for t in texts])
    compact, rank = _compact_activity_space(_embed(codes, HF_CACHE))
    maps[RAW_CODE] = dict(zip(codes, compact, strict=True))
    info[RAW_CODE] = {"status": "computed", "embedded_texts": len(codes), "activity_rank": rank}
    maps[A2V] = activity_context_embeddings(variants, context_window=2, embedding_dim=16)
    info[A2V] = {"status": "computed", "context_window": 2, "embedding_dim": 16}
    return maps, info


def stage_cluster() -> None:
    prespec_sha = require_prespec()
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    from llm_variant_clustering.dynamic_clustering import NoEligibleKError, select_dynamic_k
    from llm_variant_clustering.evaluation import evaluate_partition, weighted_medoid_indices
    from llm_variant_clustering.features import parse_sequences
    from llm_variant_clustering.semantic_compiler import ordered_semantic_pyramid
    from llm_variant_clustering.uncertainty import (
        BootstrapInterval,
        PairedDifference,
        bootstrap_seed_agreement,
        bootstrap_silhouette,
        paired_silhouette_difference,
    )
    from run_p7_development_comparison import _load_dynamic_config
    from run_representation_source_benchmark import MODEL_ID, MODEL_REVISION
    from run_reseeding_stability import SEEDS
    from run_seed_level_stability import _ari_matrix

    out = OUT / "cluster"
    assign_dir = OUT / "assignments"
    out.mkdir(parents=True, exist_ok=True)
    assign_dir.mkdir(parents=True, exist_ok=True)
    config, config_record = _load_dynamic_config(DYNAMIC_CONFIG)
    variants = _variants()
    sequences = parse_sequences(variants)
    weights = variants["frequency"].to_numpy(dtype=np.float64)
    ids = variants["variant_id"].astype(str).tolist()
    codes = sorted({a for s in sequences for a in s})
    english = _english_map()
    maps, source_info = _activity_maps(variants, codes, english)

    features = {}
    for name in METHODS:
        if name == NGRAM:
            features[name] = _ngram(variants)
        elif name in maps:
            features[name] = _feature_set(name, variants, ordered_semantic_pyramid(sequences, maps[name]))
    distances = {name: chunked_distances(f.matrix, f.metric) for name, f in features.items()}

    degenerate = {}
    for name, feats in features.items():
        dense = feats.matrix.toarray() if hasattr(feats.matrix, "toarray") else feats.matrix
        distinct_rows = int(np.unique(np.round(dense, 6), axis=0).shape[0])
        rank = source_info.get(name, {}).get("activity_rank")
        degenerate[name] = {"distinct_variant_rows": distinct_rows, "activity_rank": rank,
                            "degenerate": distinct_rows < MATCHED_K or (rank is not None and rank < 2)}

    def save(name, labels, suffix, k):
        medoids = weighted_medoid_indices(features[name].matrix, labels, weights,
                                          metric=features[name].metric)
        prefix = f"{name}_k{k}" if suffix.startswith("k") else f"{name}_{suffix}_k{k}"
        pd.DataFrame({
            "variant_id": ids, "cluster_label": labels,
            "cluster_id": [f"{prefix}_c{int(x):02d}" for x in labels],
            "is_medoid": [i == medoids[int(x)] for i, x in enumerate(labels)],
        }).to_parquet(assign_dir / f"assignments_{name}_{suffix}.parquet", index=False)

    rows, dynamic_k, partitions = [], {}, {}
    for name, feats in features.items():
        try:
            selection = select_dynamic_k(feats, weights, config, distances=distances[name])
        except NoEligibleKError as error:
            candidates, chosen = error.candidates, None
            candidates["selected"] = False
        else:
            candidates, chosen = selection.candidates, selection
            save(name, selection.selected_result.labels, "dynamic", selection.selected_k)
            dynamic_k[name] = selection.selected_k
        candidates.to_csv(out / f"dynamic_candidates_{name}.csv", index=False)
        selected = candidates.loc[candidates["selected"].astype(bool)]
        rows.append({
            "method": name, "feature_count": int(feats.matrix.shape[1]),
            "automatic_status": "selected" if chosen is not None else "no_eligible_k",
            "automatic_k": dynamic_k.get(name),
            "automatic_case_weighted_silhouette": (
                float(selected["case_weighted_silhouette"].iloc[0]) if chosen else None),
            "automatic_stability_ari": (
                float(selected["mean_seed_stability_ari"].iloc[0]) if chosen else None),
            "eligible_k": ",".join(str(int(k)) for k in candidates.loc[candidates["eligible"], "k"]),
        })
    p7_k = dynamic_k.get(P7_EN, MATCHED_K) if P7_EN in features else MATCHED_K
    p7_k_source = "p7_dynamic" if P7_EN in dynamic_k else "fallback_k8"
    ks = sorted({MATCHED_K, p7_k})

    fit_rows = []
    for k in ks:
        for name, feats in features.items():
            fit = _fit(feats, weights, k)
            metrics = evaluate_partition(feats.matrix, fit.labels, weights, metric=feats.metric,
                                         distances=distances[name])
            partitions[(k, name)] = fit.labels
            save(name, fit.labels, f"k{k}", k)
            fit_rows.append({"method": name, "k": k,
                             "mean_seed_stability_ari_5": fit.mean_seed_stability_ari,
                             **{key: metrics[key] for key in (
                                 "case_weighted_silhouette", "variant_silhouette",
                                 "case_cluster_entropy", "min_cluster_cases",
                                 "min_cluster_variants")}})
    bench = pd.DataFrame(rows).merge(pd.DataFrame(fit_rows), on="method", how="left")
    bench.to_csv(out / "method_benchmark.csv", index=False)

    interval_rows, paired_rows, ref_rows, ref_paired, stab_rows, stab_pairs, ari_rows = (
        [], [], [], [], [], [], [])
    samples = {}
    for k in ks:
        for name in features:
            interval = bootstrap_silhouette(distances[name], partitions[(k, name)], weights,
                                            resamples=RESAMPLES)
            interval_rows.append({"method": name, "k": k, "geometry": "own",
                                  **interval.as_dict("silhouette_")})
        pairs = [(role, a, b) for role, a, b in (*FAMILY, *DESCRIPTIVE_PAIRS)]
        for role, first, second in pairs:
            if first not in features or second not in features:
                paired_rows.append({"role": role, "first": first, "second": second, "k": k,
                                    "status": "not_estimable_method_abstained"})
                continue
            diff = paired_silhouette_difference(
                distances[first], partitions[(k, first)], distances[second],
                partitions[(k, second)], weights, resamples=RESAMPLES)
            samples[f"k{k}__{first}__minus__{second}"] = diff.samples
            row = {"role": role if k == MATCHED_K else f"{role}_secondary_k{k}",
                   "first": first, "second": second, "k": k, "status": "estimated",
                   "valid_resamples": int(diff.samples.size), **diff.as_dict("difference_")}
            if k == MATCHED_K and role in ("Q2", "Q3"):
                adjusted_confidence = 1.0 - (1.0 - CONFIDENCE) / FAMILY_SIZE
                adj = diff.adjusted_interval(adjusted_confidence)
                row.update(bonferroni_family_size=FAMILY_SIZE,
                           bonferroni_confidence=adjusted_confidence,
                           bonferroni_ci_lower=adj.lower, bonferroni_ci_upper=adj.upper,
                           support=support_label(row["difference_ci_lower"],
                                                 row["difference_ci_upper"], adj.lower, adj.upper))
                if degenerate[first]["degenerate"]:
                    row["support"] = "degenerate_not_interpretable"
            paired_rows.append(row)
        for geometry in REFERENCE_GEOMETRIES:
            for name in features:
                interval = bootstrap_silhouette(distances[geometry], partitions[(k, name)],
                                                weights, resamples=RESAMPLES)
                ref_rows.append({"method": name, "k": k, "geometry": geometry,
                                 **interval.as_dict("silhouette_")})
            for role, first, second in REFERENCE_PAIRS:
                if first not in features or second not in features:
                    ref_paired.append({"role": role, "geometry": geometry, "first": first,
                                       "second": second, "k": k,
                                       "status": "not_estimable_method_abstained"})
                    continue
                diff = paired_silhouette_difference(
                    distances[geometry], partitions[(k, first)], distances[geometry],
                    partitions[(k, second)], weights, resamples=RESAMPLES)
                ref_paired.append({"role": role if k == MATCHED_K else f"{role}_secondary_k{k}",
                                   "geometry": geometry, "first": first, "second": second, "k": k,
                                   "status": "estimated", **diff.as_dict("difference_")})
        names = list(features)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ari_rows.append({"k": k, "first": a, "second": b,
                                 "ari": adjusted_rand_score(partitions[(k, a)], partitions[(k, b)]),
                                 "nmi": normalized_mutual_info_score(partitions[(k, a)],
                                                                     partitions[(k, b)])})
        agreement = {}
        for name, feats in features.items():
            labelings = [KMeans(n_clusters=k, random_state=seed, n_init=10, algorithm="lloyd")
                         .fit_predict(feats.matrix, sample_weight=weights) for seed in SEEDS]
            agreement[name] = _ari_matrix(labelings)
            stab = bootstrap_seed_agreement(agreement[name], resamples=SEED_RESAMPLES)
            stab_rows.append({"method": name, "k": k, "seed_count": len(SEEDS),
                              **stab.as_dict("stability_")})
        for role, first, second in (*FAMILY, ("Q1", RAW_EN, RAW_CODE)):
            if first in agreement and second in agreement:
                diff = bootstrap_seed_agreement(agreement[first], second=agreement[second],
                                                resamples=SEED_RESAMPLES)
                stab_pairs.append({"role": role, "first": first, "second": second, "k": k,
                                   **diff.as_dict("difference_")})
        print(f"k={k} done", flush=True)

    pd.DataFrame(interval_rows).to_csv(out / "silhouette_intervals_own_space.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(out / "paired_differences_own_space.csv", index=False)
    pd.DataFrame(ref_rows).to_csv(out / "silhouette_reference_geometries.csv", index=False)
    pd.DataFrame(ref_paired).to_csv(out / "paired_differences_reference_geometries.csv", index=False)
    pd.DataFrame(stab_rows).to_csv(out / "seed_stability_12.csv", index=False)
    pd.DataFrame(stab_pairs).to_csv(out / "paired_seed_stability.csv", index=False)
    pd.DataFrame(ari_rows).to_csv(out / "ari_between_methods.csv", index=False)
    np.savez_compressed(out / "paired_difference_samples.npz", **samples)

    verification = []
    for supp_name, name in (("raw_activity_label_pyramid_bge", RAW_CODE), ("ngram_tfidf_1_3", NGRAM)):
        for suffix in ("k8", "dynamic"):
            path = SUPP_C / f"assignments_{supp_name}_{suffix}.parquet"
            frozen = pd.read_parquet(path).set_index("variant_id")["cluster_label"]
            ours_path = assign_dir / f"assignments_{name}_{suffix}.parquet"
            if not ours_path.exists():
                verification.append({"method": name, "partition": suffix,
                                     "status": "no_current_partition"})
                continue
            ours = pd.read_parquet(ours_path).set_index("variant_id")["cluster_label"]
            verification.append({"method": name, "partition": suffix, "status": "compared",
                                 "supp_c_k": int(frozen.nunique()), "current_k": int(ours.nunique()),
                                 "ari_vs_supp_c": adjusted_rand_score(frozen.loc[ids], ours.loc[ids])})
    old = pd.read_csv(SUPP_C / "baselines_descriptive/activity_source_benchmark.csv").set_index("method")
    for supp_name, name in (("raw_activity_label_bge", RAW_CODE), (A2V, A2V)):
        current = bench.loc[(bench["method"] == name) & (bench["k"] == MATCHED_K)].iloc[0]
        verification.append({
            "method": name, "partition": "k8_silhouette", "status": "compared",
            "supp_c_case_weighted_silhouette_prefix": old.loc[supp_name, "matched_case_weighted_silhouette"],
            "current_case_weighted_silhouette_fixed": current["case_weighted_silhouette"],
            "supp_c_automatic_k": old.loc[supp_name, "automatic_k"],
            "current_automatic_k": dynamic_k.get(name)})
    pd.DataFrame(verification).to_csv(out / "supp_c_verification.csv", index=False)
    _write_json(out / "cluster_manifest.json", {
        "prespec_sha256": prespec_sha, "matched_k": MATCHED_K, "p7_k": p7_k,
        "p7_k_source": p7_k_source, "ks": ks, "sources": source_info,
        "embedding_model": MODEL_ID, "embedding_revision": MODEL_REVISION,
        "resamples": RESAMPLES, "seed_resamples": SEED_RESAMPLES, "seeds": list(SEEDS),
        "bootstrap_seed": 20240601, "variants": len(ids), "dynamic_config": config_record,
        "dynamic_k": dynamic_k, "degeneracy": degenerate,
    })
    print(bench.to_string(index=False))


# --------------------------------------------------------------------------
# heldout (descriptive, transductive; per-cluster checkpoints; 4 h cap)
# --------------------------------------------------------------------------


def _heldout_dir() -> Path:
    return OUT / "heldout"


def _replay_partitions() -> tuple[dict[str, pd.DataFrame], int]:
    manifest = json.loads((OUT / "cluster/cluster_manifest.json").read_text(encoding="utf-8"))
    k = int(manifest["p7_k"])
    frames = {}
    for name in REPLAY_METHODS:
        path = OUT / "assignments" / f"assignments_{name}_k{k}.parquet"
        if path.exists():
            frames[name] = pd.read_parquet(path)
    return frames, k


def prepare_heldout_split() -> dict:
    import run_supp_semantic_tagger as tagger
    from llm_variant_clustering.process_evaluation import select_variant_holdout_split

    out = _heldout_dir()
    out.mkdir(parents=True, exist_ok=True)
    case_map = pd.read_parquet(CASE_MAP)
    frozen_inputs = {
        "supp_c_ngram_dynamic": pd.read_parquet(SUPP_C / "assignments_ngram_tfidf_1_3_dynamic.parquet"),
        "supp_c_raw_dynamic": pd.read_parquet(
            SUPP_C / "assignments_raw_activity_label_pyramid_bge_dynamic.parquet"),
    }
    split, seed = select_variant_holdout_split(case_map, frozen_inputs, test_fraction=0.2,
                                               max_seed=10_000, case_fraction_tolerance=0.05)
    frames, k = _replay_partitions()
    status: dict[str, object] = {"k": k, "frozen_split_seed": seed, "methods": {}}
    replay = {}
    for name, frame in frames.items():
        sizes = frame.groupby("cluster_id")["variant_id"].size()
        if (sizes < 2).any():
            status["methods"][name] = {"status": "structurally_infeasible_single_variant_cluster",
                                       "clusters": sizes[sizes < 2].index.tolist()}
        else:
            replay[name] = frame
    problems = {name: tagger.coverage_problems(frame, split) for name, frame in replay.items()}
    split_kind = "frozen_supp_c_split"
    if any(problems.values()):
        status["frozen_split_coverage_problems"] = problems
        split, seed = select_variant_holdout_split(case_map, replay, test_fraction=0.2,
                                                   max_seed=10_000, case_fraction_tolerance=0.05)
        split_kind = "fallback_shared_split_over_replayed_partitions"
    status.update(split_kind=split_kind, split_seed=seed,
                  test_variant_fraction=float(split["split"].eq("test").mean()),
                  test_case_fraction=float(split.loc[split["split"].eq("test"), "cases"].sum()
                                           / split["cases"].sum()))
    split.to_parquet(out / "variant_split.parquet", index=False)
    for name, frame in replay.items():
        frame.to_parquet(out / f"replay_assignments_{name}.parquet", index=False)
        status["methods"][name] = {"status": "scheduled"}
    status["methods"][WHOLE_LOG] = {"status": "scheduled"}
    _write_json(out / "split_status.json", status)
    return status


def _replay_worker(method: str) -> None:
    from llm_variant_clustering.process_evaluation import evaluate_holdout_model

    if os.environ.get("PYTHONHASHSEED") != HASH_SEED:
        raise SystemExit("Replay worker must run with PYTHONHASHSEED=0")
    out = _heldout_dir()
    split = pd.read_parquet(out / "variant_split.parquet")
    split["variant_id"] = split["variant_id"].astype(str)
    case_map = pd.read_parquet(CASE_MAP)
    case_map["variant_id"] = case_map["variant_id"].astype(str)
    case_map["case_id"] = case_map["case_id"].astype(str)
    if method == WHOLE_LOG:
        assignments = pd.DataFrame({"variant_id": split["variant_id"], "cluster_id": WHOLE_LOG})
    else:
        assignments = pd.read_parquet(out / f"replay_assignments_{method}.parquet")
    assignments["variant_id"] = assignments["variant_id"].astype(str)
    cases = case_map.merge(assignments[["variant_id", "cluster_id"]], on="variant_id",
                           validate="many_to_one").merge(split[["variant_id", "split"]],
                                                         on="variant_id", validate="many_to_one")
    events = _read_events()
    events = events[events[CASE_KEY].astype(str).isin(set(cases["case_id"]))].copy()
    cluster_by_case = dict(zip(cases["case_id"], cases["cluster_id"], strict=True))
    split_by_case = dict(zip(cases["case_id"], cases["split"], strict=True))
    event_case = events[CASE_KEY].astype(str)
    events["_cluster"] = event_case.map(cluster_by_case)
    events["_split"] = event_case.map(split_by_case)
    if events[["_cluster", "_split"]].isna().any().any():
        raise ValueError("Events without cluster or split")
    checkpoint = out / f"clusters_{method}.csv"
    done = set(pd.read_csv(checkpoint)["cluster_id"].astype(str)) if checkpoint.exists() else set()
    variant_counts = cases[["variant_id", "cluster_id", "split"]].drop_duplicates().groupby(
        ["cluster_id", "split"]).size()
    order = events.groupby("_cluster").size().sort_values(kind="stable")
    for cluster in order.index:
        if str(cluster) in done:
            continue
        group = events[events["_cluster"].eq(cluster)]
        train = group[group["_split"].eq("train")].drop(columns=["_cluster", "_split"])
        test = group[group["_split"].eq("test")].drop(columns=["_cluster", "_split"])
        started = time.time()
        metrics = evaluate_holdout_model(train, test, noise_threshold=NOISE_THRESHOLD)
        row = {"method": method, "scope": "whole_log" if method == WHOLE_LOG else "cluster",
               "cluster_id": cluster, "noise_threshold": NOISE_THRESHOLD,
               "train_variants": int(variant_counts.get((cluster, "train"), 0)),
               "test_variants": int(variant_counts.get((cluster, "test"), 0)),
               **metrics, "seconds": round(time.time() - started, 1)}
        pd.DataFrame([row]).to_csv(checkpoint, mode="a", header=not checkpoint.exists(), index=False)
        print(f"[{method}] {cluster} f1={row['fitness_precision_f1']:.4f} {row['seconds']}s", flush=True)
    _write_json(out / f"status_{method}.json", {"method": method, "status": "completed"})


def stage_heldout() -> None:
    require_prespec()
    out = _heldout_dir()
    status_path = out / "split_status.json"
    status = (json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists()
              else prepare_heldout_split())
    clock = out / "clock.json"
    if not clock.exists():
        _write_json(clock, {"first_start_epoch": time.time(), "cap_seconds": REPLAY_CAP_SECONDS})
    deadline = json.loads(clock.read_text(encoding="utf-8"))["first_start_epoch"] + REPLAY_CAP_SECONDS
    methods = [m for m, s in status["methods"].items() if s["status"] == "scheduled"]
    pending = [m for m in methods if not (out / f"status_{m}.json").exists()]
    procs = {}
    for method in pending:
        log = open(out / f"worker_{method}.log", "a", encoding="utf-8")
        procs[method] = (subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "_replay_worker", method],
            cwd=ROOT, env=_env(), stdout=log, stderr=subprocess.STDOUT), log)
    while procs:
        time.sleep(20)
        for method, (proc, log) in list(procs.items()):
            if proc.poll() is not None:
                log.close()
                del procs[method]
                if proc.returncode != 0:
                    _write_json(out / f"status_{method}.json",
                                {"method": method, "status": "failed", "returncode": proc.returncode})
        if time.time() > deadline:
            for method, (proc, log) in procs.items():
                proc.kill()
                proc.wait()
                log.close()
                _write_json(out / f"status_{method}.json",
                            {"method": method, "status": "not_completed_time_cap",
                             "cap_seconds": REPLAY_CAP_SECONDS})
            procs = {}
    print("heldout finished")


# --------------------------------------------------------------------------
# aggregate
# --------------------------------------------------------------------------


AGGREGATED = ("fitness", "precision", "fitness_precision_f1", "simplicity_arc_degree",
              "places", "transitions", "silent_transitions", "arcs")


def llm_cost() -> dict:
    files = list((OUT / "azure_activity_cache").rglob("*.json"))
    prompt = completion = attempts = 0
    for file in files:
        meta = json.loads(file.read_text(encoding="utf-8")).get("provider_metadata", {})
        usage = meta.get("usage") or {}
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
        attempts += int(meta.get("attempt_count") or 1)
    usd = prompt / 1e6 * PRICE_PER_M["prompt"] + completion / 1e6 * PRICE_PER_M["completion"]
    return {"dataset": "bpi2015_m1_english", "deployment": DEPLOYMENT,
            "model_version": MODEL_VERSION, "cache_entries": len(files),
            "provider_attempts": attempts, "prompt_tokens": prompt,
            "completion_tokens": completion, "approx_usd_list_price": round(usd, 4),
            "price_assumption_usd_per_1m": json.dumps(PRICE_PER_M)}


def _heldout_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    from llm_variant_clustering.uncertainty import bootstrap_cluster_metric

    out = _heldout_dir()
    status = json.loads((out / "split_status.json").read_text(encoding="utf-8"))
    rows, clusters_all = [], []
    for method, info in status["methods"].items():
        st_path = out / f"status_{method}.json"
        state = (json.loads(st_path.read_text(encoding="utf-8"))["status"] if st_path.exists()
                 else info["status"])
        checkpoint = out / f"clusters_{method}.csv"
        clusters = pd.read_csv(checkpoint) if checkpoint.exists() else pd.DataFrame()
        expected = (1 if method == WHOLE_LOG else
                    pd.read_parquet(out / f"replay_assignments_{method}.parquet")["cluster_id"].nunique()
                    if (out / f"replay_assignments_{method}.parquet").exists() else None)
        row = {"method": method, "status": state, "k": status["k"],
               "split_kind": status["split_kind"], "clusters_expected": expected,
               "clusters_completed": len(clusters)}
        if not clusters.empty:
            clusters_all.append(clusters)
        if state == "completed" and len(clusters) == expected:
            w = clusters["test_cases"].astype(float)
            row.update({m: float((clusters[m] * w).sum() / w.sum()) for m in AGGREGATED})
            if len(clusters) >= 2:
                interval = bootstrap_cluster_metric(
                    clusters["fitness_precision_f1"].to_numpy(dtype=float),
                    w.to_numpy(), resamples=F1_RESAMPLES)
                row.update(interval.as_dict("f1_"))
        rows.append(row)
    table = pd.DataFrame(rows)
    return table, (pd.concat(clusters_all, ignore_index=True) if clusters_all else pd.DataFrame())


def stage_aggregate() -> None:
    out = OUT
    english = _english_map()
    codes = sorted(english)
    code_weights = Counter(english.values())
    quality_rows, distinct_rows = [], []
    for method, path in ((P7_EN, OUT / "p7/activity_semantics_english.parquet"),
                         (TAGGER_EN, OUT / "tagger/activity_semantics_english_tagger.parquet")):
        frame = pd.read_parquet(path)
        valid = frame.loc[frame["status"].eq("validated")]
        semantics = {str(r.activity_label): json.loads(r.response_json) for r in valid.itertuples()}
        quality_rows.append({"method": method, "unit": "distinct_english_text",
                             "validated": len(valid), "attempted": len(frame),
                             **field_fill_rates(semantics)})
        if len(semantics) == len(set(english.values())):
            quality_rows.append({"method": method, "unit": "code",
                                 "validated": len(codes), "attempted": len(codes),
                                 **field_fill_rates(semantics, dict(code_weights))})
        from llm_variant_clustering.semantic_compiler import canonical_activity_text

        descriptors = {t: canonical_activity_text(s) for t, s in semantics.items()}
        distinct_rows.append({"method": method, "codes": len(codes),
                              "distinct_english_texts": len(set(english.values())),
                              "validated_texts": len(semantics),
                              "distinct_canonical_descriptors": len(set(descriptors.values()))})
    supp_c = pd.read_parquet(SUPP_C / "activity_embedding_index.parquet")
    distinct_rows.append({"method": "p7_code_supp_c", "codes": len(supp_c),
                          "distinct_english_texts": None, "validated_texts": len(supp_c),
                          "distinct_canonical_descriptors": supp_c["canonical_descriptor"].nunique()})
    pd.DataFrame(quality_rows).to_csv(out / "descriptor_fill_rates.csv", index=False)
    pd.DataFrame(distinct_rows).to_csv(out / "descriptor_distinctness.csv", index=False)

    pd.DataFrame([llm_cost()]).to_csv(out / "llm_cost.csv", index=False)

    summary: dict[str, object] = {}
    if (_heldout_dir() / "split_status.json").exists():
        table, clusters = _heldout_tables()
        table.to_csv(out / "heldout_f1.csv", index=False)
        if not clusters.empty:
            clusters.to_csv(out / "heldout_clusters.csv", index=False)
        f1 = table.set_index("method").get("fitness_precision_f1")
        diffs = []
        for role, first, second in (("Q1", RAW_EN, RAW_CODE), ("Q2", P7_EN, RAW_EN),
                                    ("Q3", TAGGER_EN, RAW_EN)):
            a = f1.get(first) if f1 is not None else None
            b = f1.get(second) if f1 is not None else None
            ok = a is not None and b is not None and pd.notna(a) and pd.notna(b)
            diffs.append({"role": role, "first": first, "second": second,
                          "f1_difference": float(a - b) if ok else None,
                          "status": "estimated" if ok else "not_completed"})
        pd.DataFrame(diffs).to_csv(out / "heldout_f1_differences.csv", index=False)
        summary["heldout_q1_f1_difference"] = diffs[0]["f1_difference"]
    ref = pd.read_csv(OUT / "cluster/paired_differences_reference_geometries.csv")
    q1 = ref.loc[ref["role"].eq("Q1") & ref["status"].eq("estimated")]
    summary["q1_reading"] = q1_reading(summary.get("heldout_q1_f1_difference"),
                                       q1["difference_ci_lower"].tolist(),
                                       q1["difference_ci_upper"].tolist())
    _write_json(out / "results_summary.json", summary)
    print(json.dumps(summary, indent=2))


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


def stage_manifest() -> None:
    inputs = [XES, VARIANTS, CASE_MAP, FROZEN / "prompt.txt", FROZEN / "schema.json",
              FROZEN / "manifest.json", DYNAMIC_CONFIG,
              SUPP_C / "embedding_index.parquet", SUPP_C / "activity_embedding_index.parquet",
              SUPP_C / "assignments_ngram_tfidf_1_3_dynamic.parquet",
              SUPP_C / "assignments_ngram_tfidf_1_3_k8.parquet",
              SUPP_C / "assignments_raw_activity_label_pyramid_bge_dynamic.parquet",
              SUPP_C / "assignments_raw_activity_label_pyramid_bge_k8.parquet",
              SUPP_C / "baselines_descriptive/activity_source_benchmark.csv",
              ROOT / "scripts/run_p7_activity_normalization.py",
              ROOT / "scripts/run_supp_semantic_tagger.py",
              ROOT / "scripts/run_representation_source_benchmark.py",
              Path(__file__).resolve()]
    cache = sorted((OUT / "azure_activity_cache").rglob("*.json"))
    cache_digest = hashlib.sha256("".join(sha256(p) for p in cache).encode()).hexdigest()
    outputs = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "manifest.json"
                     and "azure_activity_cache" not in p.parts)
    _write_json(OUT / "manifest.json", {
        "study": "supp_bpi2015_english_v1",
        "status": "new review-motivated follow-up; Supp C code-only result preserved",
        "prespec_sha256": sha256(OUT / "PRESPEC.md"),
        "inputs": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in inputs},
        "azure_cache": {"files": len(cache), "sha256_of_sorted_file_hashes": cache_digest},
        "outputs": {str(p.relative_to(OUT)).replace("\\", "/"): sha256(p) for p in outputs},
    })
    print(f"manifest: {len(outputs)} outputs")


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "_replay_worker":
        _replay_worker(sys.argv[2])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="+", choices=(
        "prepare", "normalize", "tag", "cluster", "heldout", "aggregate", "manifest"))
    args = parser.parse_args()
    stages = {"prepare": stage_prepare, "normalize": stage_normalize, "tag": stage_tag,
              "cluster": stage_cluster, "heldout": stage_heldout,
              "aggregate": stage_aggregate, "manifest": stage_manifest}
    for stage in args.stages:
        print(f"== {stage}", flush=True)
        stages[stage]()


if __name__ == "__main__":
    main()
