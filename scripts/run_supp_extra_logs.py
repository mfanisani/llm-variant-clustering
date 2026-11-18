"""Supplementary study C: frozen P7 pipeline on additional BPI Challenge logs.

Applies the frozen P7 configuration and the main-study baselines, unchanged, to
new out-of-calibration logs. Every computation reuses existing project
functions/scripts; this file only orchestrates them and writes into
``evaluation/supp_extra_logs_v1``. See that folder's PRESPEC.md.

Stages (run in order): normalize, benchmark, sources, heldout, aggregate.
``baselines`` (PRESPEC Amendment 1) replaces sources/heldout on a log where the
frozen P7 representation is degenerate and the benchmark wrote no metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from llm_variant_clustering.features import activity_context_embeddings, parse_sequences
from llm_variant_clustering.semantic_compiler import ordered_semantic_pyramid
from llm_variant_clustering.uncertainty import (
    BootstrapInterval,
    PairedDifference,
    bootstrap_cluster_metric,
    bootstrap_seed_agreement,
    bootstrap_silhouette,
    cluster_scope_mask,
    paired_silhouette_difference,
)
from run_p7_development_comparison import _load_dynamic_config
from run_representation_source_benchmark import (
    MODEL_ID,
    MODEL_REVISION,
    _compact_activity_space,
    _embed,
    _evaluate,
)
from run_reseeding_stability import SEEDS
from run_seed_level_stability import _ari_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("evaluation/supp_extra_logs_v1")
FROZEN = Path("configs/frozen/p7_activity_gpt41_v1")
DYNAMIC_CONFIG = Path("configs/clustering/dynamic_k_v1.json")
P7 = "p7_ordered_pyramid_1_2_4_bge"
RAW = "raw_activity_label_pyramid_bge"
NGRAM = "ngram_tfidf_1_3"
FIRST = "p7_llm_semantic_bge"
FALLBACK_K = 8
RESAMPLES = 400
MAX_BOOTSTRAP_VARIANTS = 2500
SEED_RESAMPLES = 4000
F1_RESAMPLES = 2000
CONFIDENCE = 0.95
# Prespecified Bonferroni family: P7 vs these comparators on every supplementary log.
FAMILY_COMPARATORS = ("raw_activity_label_bge", "activity2vec_log_context_d16")

LOGS = {
    "bpi2012": {
        "folder": "bpi2012",
        "xes": "datasets/raw/bpi2012/BPI_Challenge_2012.xes.gz",
    },
    "bpi2015_m1": {
        "folder": "bpi2015_municipality1",
        "xes": "datasets/raw/bpi2015_municipality1/BPIC15_1.xes",
    },
}


def _paths(log: str) -> dict[str, Path]:
    folder = Path("datasets/processed") / LOGS[log]["folder"]
    out = OUT / log
    return {
        "variants": folder / "variants.parquet",
        "case_map": folder / "case_variant_map.parquet",
        "xes": Path(LOGS[log]["xes"]),
        "out": out,
        "semantics": out / "activity_semantics_frozen.parquet",
        "cache": out / "azure_activity_cache",
        "sources": out / "sources",
    }


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "scripts")])
    env["PYTHONHASHSEED"] = "0"
    return env


def _run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT, env=_env())


def matched_k(benchmark_dir: Path) -> tuple[int, str]:
    """Matched K rule from the main study: P7 automatic K, else exploratory K=8."""
    dynamic = pd.read_csv(benchmark_dir / "metrics_dynamic_k.csv").set_index("method")
    row = dynamic.loc[P7]
    if row["status"] == "selected":
        return int(row["selected_k"]), "dynamic"
    return FALLBACK_K, "k8"


def stage_normalize(log: str, endpoint: str, deployment: str, model_version: str) -> None:
    p = _paths(log)
    _run([
        "scripts/run_p7_activity_normalization.py",
        str(p["variants"]),
        str(p["semantics"]),
        "--cache-dir", str(p["cache"]),
        "--endpoint", endpoint,
        "--deployment", deployment,
        "--model-version", model_version,
        "--prompt", str(FROZEN / "prompt.txt"),
        "--schema", str(FROZEN / "schema.json"),
    ])


def stage_benchmark(log: str) -> None:
    p = _paths(log)
    _run([
        "scripts/run_p7_scalable_benchmark.py",
        str(p["variants"]),
        str(p["semantics"]),
        str(p["out"]),
        "--dynamic-config", str(DYNAMIC_CONFIG),
        "--clusters", str(FALLBACK_K),
    ])


def _frozen_p7_labels(benchmark_dir: Path, partition: str, ids: list[str]) -> np.ndarray:
    suffix = "dynamic" if partition == "dynamic" else "k8"
    frame = pd.read_parquet(benchmark_dir / f"assignments_{P7}_{suffix}.parquet")
    return frame.set_index(frame["variant_id"].astype(str))["cluster_label"].loc[ids].to_numpy()


def stage_sources(log: str, cache_dir: Path) -> None:
    """Matched-K source comparison, paired bootstrap, and 12-seed stability.

    Mirrors run_representation_source_benchmark.py, run_representation_uncertainty.py
    and run_seed_level_stability.py; only the dataset loop is new.
    """
    p = _paths(log)
    out = p["sources"]
    out.mkdir(parents=True, exist_ok=True)
    k, partition = matched_k(p["out"])
    config, config_record = _load_dynamic_config(DYNAMIC_CONFIG)
    variants = pd.read_parquet(p["variants"])
    keep = frozenset(pd.read_parquet(p["out"] / "embedding_index.parquet")["variant_id"].astype(str))
    variants = variants.loc[variants["variant_id"].astype(str).isin(keep)].reset_index(drop=True)
    sequences = parse_sequences(variants)
    weights = variants["frequency"].to_numpy(dtype=np.float64)
    ids = variants["variant_id"].astype(str).tolist()
    reference = _frozen_p7_labels(p["out"], partition, ids)
    sample_size = MAX_BOOTSTRAP_VARIANTS if len(variants) > MAX_BOOTSTRAP_VARIANTS else None

    index = pd.read_parquet(p["out"] / "activity_embedding_index.parquet")
    labels = index["activity_label"].astype(str).tolist()
    descriptors = index["canonical_descriptor"].astype(str).tolist()
    semantic_compact, semantic_rank = _compact_activity_space(_embed(descriptors, cache_dir))
    raw_compact, raw_rank = _compact_activity_space(_embed(labels, cache_dir))
    sources = {
        FIRST: dict(zip(labels, semantic_compact, strict=True)),
        "raw_activity_label_bge": dict(zip(labels, raw_compact, strict=True)),
        "activity2vec_log_context_d16": activity_context_embeddings(
            variants, context_window=2, embedding_dim=16
        ),
        f"activity2vec_log_context_d{semantic_rank}": activity_context_embeddings(
            variants, context_window=2, embedding_dim=semantic_rank
        ),
    }

    bench_rows, interval_rows, paired_rows, stab_rows, stab_pairs = [], [], [], [], []
    computed, ari = {}, {}
    for name, activity_map in sources.items():
        matrix = ordered_semantic_pyramid(sequences, activity_map)
        row, labels_out = _evaluate(name, matrix, variants, weights, k, config, reference)
        row.update(dataset=log, matched_partition=partition)
        bench_rows.append(row)
        pd.DataFrame({
            "variant_id": ids,
            "cluster_label": labels_out,
            "cluster_id": [f"{name}_k{k}_c{int(x):02d}" for x in labels_out],
        }).to_parquet(out / f"assignments_{name}_k{k}.parquet", index=False)

        distances = pairwise_distances(matrix, metric="cosine")
        computed[name] = (distances, labels_out)
        interval = bootstrap_silhouette(
            distances, labels_out, weights, resamples=RESAMPLES, sample_size=sample_size
        )
        interval_rows.append({
            "dataset": log, "method": name, "k": k,
            "bootstrap_sample_size": sample_size or len(variants),
            **interval.as_dict("silhouette_"),
        })

        for stab_k in sorted({k, FALLBACK_K}):
            labelings = [
                KMeans(n_clusters=stab_k, random_state=seed, n_init=10, algorithm="lloyd")
                .fit_predict(matrix, sample_weight=weights)
                for seed in SEEDS
            ]
            agreement = _ari_matrix(labelings)
            ari[(stab_k, name)] = agreement
            seed_interval = bootstrap_seed_agreement(agreement, resamples=SEED_RESAMPLES)
            stab_rows.append({
                "dataset": log, "method": name, "k": stab_k, "seed_count": len(SEEDS),
                "bootstrap_unit": "seed", **seed_interval.as_dict("stability_"),
            })
        print(f"{log} {name} done", flush=True)

    paired_samples = {}
    for name in computed:
        if name == FIRST:
            continue
        difference = paired_silhouette_difference(
            *computed[FIRST], *computed[name], weights,
            resamples=RESAMPLES, sample_size=sample_size,
        )
        paired_samples[name] = difference.samples
        paired_rows.append({
            "dataset": log, "first": FIRST, "second": name, "k": k,
            "bootstrap_sample_size": sample_size or len(variants),
            "valid_resamples": int(difference.samples.size),
            **difference.as_dict("difference_"),
        })
    np.savez_compressed(out / "paired_difference_samples.npz", **paired_samples)
    for (stab_k, name), agreement in ari.items():
        if name == FIRST:
            continue
        difference = bootstrap_seed_agreement(
            ari[(stab_k, FIRST)], second=agreement, resamples=SEED_RESAMPLES
        )
        stab_pairs.append({
            "dataset": log, "first": FIRST, "second": name, "k": stab_k,
            **difference.as_dict("difference_"),
        })

    pd.DataFrame(bench_rows).to_csv(out / "activity_source_benchmark.csv", index=False)
    pd.DataFrame(interval_rows).to_csv(out / "source_silhouette_intervals.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(out / "source_paired_differences.csv", index=False)
    pd.DataFrame(stab_rows).to_csv(out / "reseeding_stability_seed_level.csv", index=False)
    pd.DataFrame(stab_pairs).to_csv(out / "paired_stability_differences_seed_level.csv", index=False)
    np.savez_compressed(out / "ari_matrices.npz", **{f"k{a}__{b}": m for (a, b), m in ari.items()})
    (out / "manifest.json").write_text(json.dumps({
        "dataset": log, "matched_k": k, "matched_partition": partition,
        "semantic_rank": semantic_rank, "raw_rank": raw_rank,
        "embedding_model": MODEL_ID, "embedding_revision": MODEL_REVISION,
        "context_window": 2, "resamples": RESAMPLES,
        "max_bootstrap_variants": MAX_BOOTSTRAP_VARIANTS,
        "seed_resamples": SEED_RESAMPLES, "seeds": list(SEEDS),
        "dynamic_config": config_record,
    }, indent=2) + "\n", encoding="utf-8")


def heldout_assignments(benchmark_dir: Path) -> list[Path]:
    """Raw, n-gram and P7 automatic-K assignments; K=8 for any method that abstains."""
    dynamic = pd.read_csv(benchmark_dir / "metrics_dynamic_k.csv").set_index("method")
    paths = []
    for method in (NGRAM, P7, RAW):
        suffix = "dynamic" if dynamic.loc[method, "status"] == "selected" else "k8"
        paths.append(benchmark_dir / f"assignments_{method}_{suffix}.parquet")
    return paths


def stage_heldout(log: str) -> None:
    p = _paths(log)
    _run([
        "scripts/run_heldout_process_evaluation.py",
        str(p["xes"]), str(p["case_map"]),
        str(p["out"] / "process_model_metrics_heldout.csv"),
        *[str(path) for path in heldout_assignments(p["out"])],
    ])


def p7_degenerate(benchmark_dir: Path) -> bool:
    """Amendment 1: P7 is non-computable when the frozen benchmark wrote no metrics."""
    return not (benchmark_dir / "metrics_dynamic_k.csv").exists()


def _candidate_summary(benchmark_dir: Path, method: str) -> dict:
    candidates = pd.read_csv(benchmark_dir / f"dynamic_candidates_{method}.csv")
    chosen = candidates.loc[candidates["selected"].astype(bool)]
    if chosen.empty:
        return {"method": method, "status": "no_eligible_k", "selected_k": None}
    row = chosen.iloc[0]
    return {"method": method, "status": "selected", "selected_k": int(row["k"]),
            **{key: row[key] for key in ("case_weighted_silhouette", "variant_silhouette",
                                         "mean_seed_stability_ari", "min_cluster_cases")}}


def stage_baselines(log: str, cache_dir: Path) -> None:
    """Amendment 1: descriptive baselines on a log where P7 is degenerate (no P7 contrast)."""
    p = _paths(log)
    out = p["out"] / "baselines_descriptive"
    out.mkdir(parents=True, exist_ok=True)
    config, _ = _load_dynamic_config(DYNAMIC_CONFIG)
    variants = pd.read_parquet(p["variants"])
    keep = frozenset(pd.read_parquet(p["out"] / "embedding_index.parquet")["variant_id"].astype(str))
    variants = variants.loc[variants["variant_id"].astype(str).isin(keep)].reset_index(drop=True)
    sequences = parse_sequences(variants)
    weights = variants["frequency"].to_numpy(dtype=np.float64)
    ids = variants["variant_id"].astype(str).tolist()
    sample_size = MAX_BOOTSTRAP_VARIANTS if len(variants) > MAX_BOOTSTRAP_VARIANTS else None
    raw_k8 = pd.read_parquet(p["out"] / f"assignments_{RAW}_k8.parquet")
    reference = raw_k8.set_index(raw_k8["variant_id"].astype(str))["cluster_label"].loc[ids].to_numpy()

    index = pd.read_parquet(p["out"] / "activity_embedding_index.parquet")
    labels = index["activity_label"].astype(str).tolist()
    raw_compact, raw_rank = _compact_activity_space(_embed(labels, cache_dir))
    sources = {
        "raw_activity_label_bge": dict(zip(labels, raw_compact, strict=True)),
        "activity2vec_log_context_d16": activity_context_embeddings(
            variants, context_window=2, embedding_dim=16
        ),
    }
    bench_rows, interval_rows, stab_rows = [], [], []
    for name, activity_map in sources.items():
        matrix = ordered_semantic_pyramid(sequences, activity_map)
        row, labels_out = _evaluate(name, matrix, variants, weights, FALLBACK_K, config, reference)
        row = {key: value for key, value in row.items() if "frozen_p7" not in key}
        bench_rows.append({"dataset": log, **row})
        distances = pairwise_distances(matrix, metric="cosine")
        interval = bootstrap_silhouette(
            distances, labels_out, weights, resamples=RESAMPLES, sample_size=sample_size
        )
        interval_rows.append({"dataset": log, "method": name, "k": FALLBACK_K,
                              **interval.as_dict("silhouette_")})
        labelings = [
            KMeans(n_clusters=FALLBACK_K, random_state=seed, n_init=10, algorithm="lloyd")
            .fit_predict(matrix, sample_weight=weights)
            for seed in SEEDS
        ]
        seed_interval = bootstrap_seed_agreement(_ari_matrix(labelings), resamples=SEED_RESAMPLES)
        stab_rows.append({"dataset": log, "method": name, "k": FALLBACK_K, "seed_count": len(SEEDS),
                          "bootstrap_unit": "seed", **seed_interval.as_dict("stability_")})
        print(f"{log} {name} done", flush=True)
    pd.DataFrame(bench_rows).to_csv(out / "activity_source_benchmark.csv", index=False)
    pd.DataFrame(interval_rows).to_csv(out / "source_silhouette_intervals.csv", index=False)
    pd.DataFrame(stab_rows).to_csv(out / "reseeding_stability_seed_level.csv", index=False)
    pd.DataFrame([_candidate_summary(p["out"], m) for m in (RAW, NGRAM)]).to_csv(
        out / "automatic_k_baselines.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "dataset": log, "p7_status": "degenerate_representation", "k": FALLBACK_K,
        "raw_rank": raw_rank, "embedding_model": MODEL_ID, "embedding_revision": MODEL_REVISION,
        "resamples": RESAMPLES, "seed_resamples": SEED_RESAMPLES, "seeds": list(SEEDS),
        "note": "PRESPEC Amendment 1; descriptive only, no P7 contrast exists.",
    }, indent=2) + "\n", encoding="utf-8")
    assignments = []
    for method in (NGRAM, RAW):
        suffix = "dynamic" if _candidate_summary(p["out"], method)["status"] == "selected" else "k8"
        assignments.append(str(p["out"] / f"assignments_{method}_{suffix}.parquet"))
    _run([
        "scripts/run_heldout_process_evaluation.py",
        str(p["xes"]), str(p["case_map"]),
        str(out / "process_model_metrics_heldout.csv"),
        *assignments,
    ])


def bonferroni(
    family: list[dict],
    samples: dict[tuple[str, str], np.ndarray],
    family_size: int | None = None,
) -> pd.DataFrame:
    """Re-read the stored paired replicates at the Bonferroni level (main-study method).

    ``family_size`` fixes the divisor to the prespecified family (Amendment 1)
    even when some contrasts are not estimable.
    """
    family = [record for record in family if record["second"] in FAMILY_COMPARATORS]
    size = family_size or len(family)
    if not family:
        return pd.DataFrame()
    adjusted_confidence = 1.0 - (1.0 - CONFIDENCE) / size
    rows = []
    for record in family:
        difference = PairedDifference(
            interval=BootstrapInterval(
                point_estimate=float(record["difference_point_estimate"]),
                lower=float(record["difference_ci_lower"]),
                upper=float(record["difference_ci_upper"]),
                confidence=float(record["difference_confidence"]),
                resamples=int(record["difference_resamples"]),
            ),
            excludes_zero=bool(record["difference_excludes_zero"]),
            proportion_favouring_first=float(record["difference_proportion_favouring_first"]),
            samples=samples[(record["dataset"], record["second"])],
        )
        adjusted = difference.adjusted_interval(adjusted_confidence)
        rows.append({
            **record,
            "bonferroni_family_size": size,
            "bonferroni_confidence": adjusted_confidence,
            "bonferroni_ci_lower": adjusted.lower,
            "bonferroni_ci_upper": adjusted.upper,
            "bonferroni_excludes_zero": bool(adjusted.lower > 0 or adjusted.upper < 0),
        })
    return pd.DataFrame(rows)


PRESPECIFIED_FAMILY_SIZE = len(FAMILY_COMPARATORS) * len(LOGS)


def _bonferroni(logs: list[str]) -> pd.DataFrame:
    family, samples = [], {}
    for log in logs:
        if p7_degenerate(_paths(log)["out"]):
            continue
        sources = _paths(log)["sources"]
        family.extend(pd.read_csv(sources / "source_paired_differences.csv").to_dict("records"))
        with np.load(sources / "paired_difference_samples.npz") as stored:
            samples.update({(log, name): stored[name] for name in stored.files})
    return bonferroni(family, samples, family_size=PRESPECIFIED_FAMILY_SIZE)


def heldout_f1_intervals(path: Path) -> list[dict]:
    """Descriptive cluster-level bootstrap of held-out F1, weighted by test cases."""
    frame = pd.read_csv(path)
    clusters = frame.loc[cluster_scope_mask(frame["scope"])]
    rows = []
    for method, group in clusters.groupby("method"):
        if len(group) < 2:
            continue
        interval = bootstrap_cluster_metric(
            group["fitness_precision_f1"].to_numpy(dtype=float),
            group["test_cases"].to_numpy(dtype=float),
            resamples=F1_RESAMPLES,
        )
        rows.append({"method": method, "cluster_count": len(group), **interval.as_dict("f1_")})
    return rows


def llm_cost(log: str) -> dict:
    files = list(_paths(log)["cache"].rglob("*.json"))
    prompt = completion = attempts = 0
    for file in files:
        meta = json.loads(file.read_text(encoding="utf-8")).get("provider_metadata", {})
        usage = meta.get("usage") or {}
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
        attempts += int(meta.get("attempt_count") or 1)
    return {"dataset": log, "cache_entries": len(files), "provider_attempts": attempts,
            "prompt_tokens": prompt, "completion_tokens": completion}


def stage_aggregate(logs: list[str]) -> None:
    table2, heldout, f1_rows, costs, stats = [], [], [], [], []
    for log in logs:
        p = _paths(log)
        summary = json.loads((Path("datasets/processed") / LOGS[log]["folder"] / "summary.json")
                             .read_text(encoding="utf-8"))
        stats.append({"dataset": log, **{key: summary.get(key) for key in (
            "lifecycle_policy_applied", "events_after_filter", "cases", "activities", "variants")}})
        costs.append(llm_cost(log))
        if p7_degenerate(p["out"]):
            descriptive = p["out"] / "baselines_descriptive"
            table2.append({"log": log, "k": FALLBACK_K, "p7_automatic": False,
                           f"{P7}_status": "degenerate_representation"})
            if (descriptive / "activity_source_benchmark.csv").exists():
                bench = pd.read_csv(descriptive / "activity_source_benchmark.csv").set_index("method")
                table2[-1].update(
                    raw_label=bench.loc["raw_activity_label_bge", "matched_case_weighted_silhouette"],
                    activity2vec_d16=bench.loc["activity2vec_log_context_d16",
                                               "matched_case_weighted_silhouette"],
                )
                auto = pd.read_csv(descriptive / "automatic_k_baselines.csv").set_index("method")
                for method in (RAW, NGRAM):
                    table2[-1][f"{method}_status"] = auto.loc[method, "status"]
                if auto.loc[NGRAM, "status"] == "selected":
                    table2[-1].update(ngram_own_k=auto.loc[NGRAM, "case_weighted_silhouette"],
                                      ngram_k=int(auto.loc[NGRAM, "selected_k"]))
            held_path = descriptive / "process_model_metrics_heldout.csv"
            manifest = descriptive / "process_model_metrics_heldout_manifest.json"
            if held_path.exists():
                h = pd.read_csv(held_path)
                whole = h[h["scope"] == "whole_log"].iloc[0]
                heldout.append({"log": log, "method": "whole_log",
                                "test_variants": int(whole["test_variants"]),
                                **{m: whole[m] for m in ("fitness", "precision", "fitness_precision_f1")}})
                for _, a in h[h["scope"] == "test_case_weighted_aggregate"].iterrows():
                    heldout.append({"log": log, "method": a["method"],
                                    "test_variants": int(a["test_variants"]),
                                    **{m: a[m] for m in ("fitness", "precision", "fitness_precision_f1")}})
                heldout.append({"log": log, "method": f"{P7}:abstained:degenerate_representation"})
                f1_rows.extend({"log": log, **r} for r in heldout_f1_intervals(held_path))
            else:
                status = (json.loads(manifest.read_text(encoding="utf-8"))["status"]
                          if manifest.exists() else "not_run")
                heldout.append({"log": log, "method": f"abstained:{status}"})
            continue
        dynamic = pd.read_csv(p["out"] / "metrics_dynamic_k.csv").set_index("method")
        fixed = pd.read_csv(p["out"] / "metrics_fixed_k8.csv").set_index("method")
        bench = pd.read_csv(p["sources"] / "activity_source_benchmark.csv").set_index("method")
        paired = pd.read_csv(p["sources"] / "source_paired_differences.csv").set_index("second")
        k, partition = matched_k(p["out"])
        ng_selected = dynamic.loc[NGRAM, "status"] == "selected"
        ng = dynamic.loc[NGRAM] if ng_selected else fixed.loc[NGRAM]
        row = {
            "log": log, "k": k, "p7_automatic": partition == "dynamic",
            **{f"{m}_status": dynamic.loc[m, "status"] for m in (P7, RAW, NGRAM)},
            "p7": bench.loc[FIRST, "matched_case_weighted_silhouette"],
            "raw_label": bench.loc["raw_activity_label_bge", "matched_case_weighted_silhouette"],
            "activity2vec_d16": bench.loc["activity2vec_log_context_d16", "matched_case_weighted_silhouette"],
            "ngram_own_k": ng["case_weighted_silhouette"],
            "ngram_k": int(ng["selected_k"]) if ng_selected else FALLBACK_K,
        }
        for second, tag in (("raw_activity_label_bge", "raw"), ("activity2vec_log_context_d16", "a2v16")):
            r = paired.loc[second]
            row[f"p7_minus_{tag}"] = r["difference_point_estimate"]
            row[f"p7_minus_{tag}_lo"] = r["difference_ci_lower"]
            row[f"p7_minus_{tag}_hi"] = r["difference_ci_upper"]
        rank_names = [n for n in bench.index if n.startswith("activity2vec_log_context_d") and n != "activity2vec_log_context_d16"]
        if rank_names:
            row["activity2vec_rank_method"] = rank_names[0]
            row["activity2vec_rank"] = bench.loc[rank_names[0], "matched_case_weighted_silhouette"]
        table2.append(row)

        held_path = p["out"] / "process_model_metrics_heldout.csv"
        if held_path.exists():
            h = pd.read_csv(held_path)
            whole = h[h["scope"] == "whole_log"].iloc[0]
            heldout.append({"log": log, "method": "whole_log", "test_variants": int(whole["test_variants"]),
                            **{m: whole[m] for m in ("fitness", "precision", "fitness_precision_f1")}})
            for _, a in h[h["scope"] == "test_case_weighted_aggregate"].iterrows():
                heldout.append({"log": log, "method": a["method"], "test_variants": int(a["test_variants"]),
                                **{m: a[m] for m in ("fitness", "precision", "fitness_precision_f1")}})
            f1_rows.extend({"log": log, **r} for r in heldout_f1_intervals(held_path))
        else:
            manifest = p["out"] / "process_model_metrics_heldout_manifest.json"
            status = json.loads(manifest.read_text(encoding="utf-8"))["status"] if manifest.exists() else "not_run"
            heldout.append({"log": log, "method": f"abstained:{status}"})

    pd.DataFrame(stats).to_csv(OUT / "dataset_stats.csv", index=False)
    pd.DataFrame(table2).to_csv(OUT / "table2_matched_k_silhouette.csv", index=False)
    _bonferroni(logs).to_csv(OUT / "bonferroni_paired_differences.csv", index=False)
    full = [log for log in logs if not p7_degenerate(_paths(log)["out"])]
    stab = [pd.read_csv(_paths(log)["sources"] / "paired_stability_differences_seed_level.csv")
            for log in full]
    pd.concat(stab, ignore_index=True).to_csv(OUT / "seed_stability_paired.csv", index=False)
    stability_files = [_paths(log)["sources"] / "reseeding_stability_seed_level.csv" for log in full]
    stability_files += [_paths(log)["out"] / "baselines_descriptive" / "reseeding_stability_seed_level.csv"
                        for log in logs if log not in full]
    pd.concat([pd.read_csv(path) for path in stability_files if path.exists()],
              ignore_index=True).to_csv(OUT / "seed_stability.csv", index=False)
    pd.DataFrame(heldout).to_csv(OUT / "table3_heldout.csv", index=False)
    pd.DataFrame(f1_rows).to_csv(OUT / "heldout_f1_cluster_bootstrap.csv", index=False)
    pd.DataFrame(costs).to_csv(OUT / "llm_cost.csv", index=False)
    print(OUT / "table2_matched_k_silhouette.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["normalize", "benchmark", "sources", "heldout",
                                          "baselines", "aggregate"])
    parser.add_argument("--logs", nargs="*", default=list(LOGS))
    parser.add_argument("--endpoint", default=os.environ.get("AZURE_OPENAI_ENDPOINT"))
    parser.add_argument("--deployment", default="gpt-4.1")
    parser.add_argument("--model-version", default="2025-04-14")
    parser.add_argument("--cache-dir", type=Path, default=Path("models/huggingface"))
    args = parser.parse_args()
    unknown = sorted(set(args.logs) - set(LOGS))
    if unknown:
        raise SystemExit(f"Unknown logs: {unknown}")
    os.chdir(ROOT)
    if args.stage == "aggregate":
        stage_aggregate(args.logs)
        return
    for log in args.logs:
        if args.stage == "normalize":
            if not args.endpoint:
                raise SystemExit("--endpoint or AZURE_OPENAI_ENDPOINT is required")
            stage_normalize(log, args.endpoint, args.deployment, args.model_version)
        elif args.stage == "benchmark":
            stage_benchmark(log)
        elif args.stage == "sources":
            stage_sources(log, args.cache_dir)
        elif args.stage == "heldout":
            stage_heldout(log)
        elif args.stage == "baselines":
            stage_baselines(log, args.cache_dir)


if __name__ == "__main__":
    main()
