"""Full-size (n-out-of-n) paired bootstrap for the capped Table 2 silhouette contrasts.

The Table 2 intervals and the 5,000-replicate Bonferroni intervals for logs with
more than 2,500 variants (BPI 2017, BPI 2019; BPI 2012 in the supplement) used an
unrescaled m-out-of-n bootstrap with m = 2,500. This script recomputes the same
paired contrasts with full-size resampling (n draws with replacement from the n
unique variants, case frequencies kept as weights, identical draws for every
representation within a replicate). Embeddings, K and partitions are those of
evaluation/review3_fixes_v1 (exact Table 2 partitions; corrected singleton
convention of llm_variant_clustering.evaluation.frequency_weighted_silhouette).

Writes only to evaluation/full_size_bootstrap_v1/. No LLM calls.

  python scripts/run_full_size_bootstrap.py run --logs bpi2017 bpi2019 bpi2012
  python scripts/run_full_size_bootstrap.py summarize
  python scripts/run_full_size_bootstrap.py manifest

Computation. A draw with replacement contains duplicated variants. A duplicated
variant is a set of cases at distance 0 from each other, so evaluating the
case-weighted silhouette on the expanded index list equals evaluating it on the
unique drawn variants with weight count x frequency (the library function only
uses weighted sums of distances, cluster case weights and a weighted mean).
That lets many replicates share one pass over the frozen distance matrix:
S = D[:, cluster k] @ (counts x weights)[cluster k] for a batch of replicates.
Row chunks of the float32 cosine distance matrix (computed exactly as in the
source scripts, with sklearn.pairwise_distances) are read from a memory map, so
at most one n x n float32 matrix is on disk and only row chunks are in memory.
The engine is checked against (i) the published point estimates, (ii) the
review3_fixes_v1 capped replicates (same draws, m = 2,500; every replicate),
(iii) the library function on full-size draws (BPI 2017, BPI 2012), (iv) the
BPI 2017 full-size replicates of review_bootstrap_tails_v1 and (v) the
published capped intervals, which it reproduces from its own capped replicates.
The A2V partition at d = rank of the P7 space (the paper's "either dimension")
is refit as in run_representation_uncertainty.py (BPI 2017/2019) and checked
against the published point estimate.
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
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from llm_variant_clustering.clustering import weighted_kmeans
from llm_variant_clustering.evaluation import frequency_weighted_silhouette
from llm_variant_clustering.features import activity_context_embeddings, parse_sequences
from llm_variant_clustering.semantic_compiler import ordered_semantic_pyramid
from llm_variant_clustering.uncertainty import _percentile_interval, _resample_indices

import run_review3_fixes as r3

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"
R3 = EVAL / "review3_fixes_v1"
TAILS = EVAL / "review_bootstrap_tails_v1"
OUT = EVAL / "full_size_bootstrap_v1"
REPS = OUT / "replicates"
SCRATCH = OUT / "_scratch"
SEED = 20240601
CAP = 2500
CONFIDENCE = 0.95
BONFERRONI = 1.0 - (1.0 - CONFIDENCE) / 4  # family: BPI 2017, BPI 2019, Sepsis, RTFM
BATCH = 500
ROW_CHUNK = 1024

# log -> {partition name (as in run_review3_fixes.partitions): replicates}. Every partition
# gets full-size replicates and, on the same engine, capped (m = 2,500) replicates.
# A2V d16 needs only 300 for Table 2; 5,000 are computed as a robustness check.
PLAN = {
    "bpi2017": {"p7_table2_k2": 5000, "raw_table2_k2": 5000, "a2v16_table2_k2": 5000,
                "a2v23_table2_k2": 5000, "raw_fullbench_dynamic": 5000, "tagger_k2": 5000},
    "bpi2019": {"p7_table2_k6": 5000, "raw_table2_k6": 5000, "a2v16_table2_k6": 5000,
                "a2v42_table2_k6": 5000, "raw_fullbench_dynamic": 5000, "tagger_k6": 5000},
    "bpi2012": {"p7_matched_k8": 400, "raw_matched_k8": 400, "a2v16_matched_k8": 400,
                "a2v23_matched_k8": 400},
}
# A2V at d = rank of the P7 activity space (the "other dimension" of the paper's
# P7-versus-A2V statement). BPI 2017/2019: dense refit exactly as
# run_representation_uncertainty.py; BPI 2012: the saved supplement assignment.
A2V_RANK = {"bpi2017": 23, "bpi2019": 42, "bpi2012": 23}
REFIT = OUT / "partitions"
# replicate count of the capped review3 run (shape of the capped draw matrix)
CAPPED_SHAPE = {"bpi2017": 5000, "bpi2019": 5000, "bpi2012": 400}

LOG_FILE = OUT / "run_log.txt"


def say(message: str) -> None:
    print(message, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------


def counts_from_draws(draws: np.ndarray, n: int) -> np.ndarray:
    out = np.empty((draws.shape[0], n), dtype=np.uint16)
    for row, indices in enumerate(draws):
        out[row] = np.bincount(indices, minlength=n)
    return out


def silhouette_batch(dist, labels: np.ndarray, weights: np.ndarray, counts: np.ndarray,
                     row_chunk: int = ROW_CHUNK) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Case-weighted silhouette (corrected singleton convention) for each count vector.

    dist: (n, n) distance matrix (memmap allowed); counts: (B, n) draw multiplicities.
    Returns (silhouette, min present cluster case weight, valid) per replicate.
    Equivalent to frequency_weighted_silhouette(D[ix], labels[ix], weights[ix]) on the
    expanded index list ix of each draw.
    """
    _, codes = np.unique(labels, return_inverse=True)
    n_codes = int(codes.max()) + 1
    n = codes.size
    batch = counts.shape[0]
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(n_codes + 1))
    cw = counts.T.astype(np.float64) * weights[:, None]  # n x B
    cw_sorted = np.ascontiguousarray(cw[order])
    wk = np.stack([cw_sorted[bounds[k]:bounds[k + 1]].sum(axis=0) for k in range(n_codes)])
    present = wk > 0
    num = np.zeros(batch)
    for start in range(0, n, row_chunk):
        stop = min(start + row_chunk, n)
        block_w = cw[start:stop]
        if not block_w.any():
            continue
        rows = np.asarray(dist[start:stop], dtype=np.float64)[:, order]
        size = stop - start
        sums = np.empty((n_codes, size, batch))
        for k in range(n_codes):
            lo, hi = bounds[k], bounds[k + 1]
            np.matmul(rows[:, lo:hi], cw_sorted[lo:hi], out=sums[k])
        del rows
        with np.errstate(divide="ignore", invalid="ignore"):
            cd = sums / wk[:, None, :]
            del sums
            own = codes[start:stop]
            at = np.arange(size)
            own_w = wk[own]
            within = cd[own, at] * own_w / np.maximum(own_w - 1.0, 1.0)
            single = own_w <= 1.0
            within[single] = 0.0
            cd = np.where(present[:, None, :], cd, np.inf)
            cd[own, at] = np.inf
            nearest = cd.min(axis=0)
            del cd
            denom = np.maximum(within, nearest)
            sil = np.divide(nearest - within, denom, out=np.zeros_like(denom),
                            where=denom > 0)
        sil[single] = 0.0
        num += np.where(block_w > 0, block_w * sil, 0.0).sum(axis=0)
    total = cw.sum(axis=0)
    valid = present.sum(axis=0) >= 2
    values = np.where(valid, num / total, np.nan)
    min_weight = np.where(present, wk, np.inf).min(axis=0)
    return values, min_weight, valid


def run_batches(dist, labels, weights, counts) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parts = [silhouette_batch(dist, labels, weights, counts[s:s + BATCH])
             for s in range(0, counts.shape[0], BATCH)]
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def _frozen_p7_files(log: str, k: int) -> list[Path]:
    if log == "bpi2012":
        return [EVAL / "supp_extra_logs_v1/bpi2012/sources/assignments_p7_llm_semantic_bge_k8.parquet"]
    return [
        EVAL / "full_benchmark" / log / f"assignments_{r3.P7_BENCH}_dynamic.parquet",
        EVAL / "representation_sources" / log / f"assignments_p7_llm_semantic_bge_k{k}.parquet",
    ]


def _published_points(log: str) -> dict[str, float]:
    """Point estimates of the published capped interval runs (run_representation_uncertainty.py)."""
    if log == "bpi2012":
        path = EVAL / "supp_extra_logs_v1/bpi2012/sources/source_silhouette_intervals.csv"
    else:
        path = EVAL / "representation_sources_extra/source_silhouette_intervals.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[frame["dataset"].eq(log)]
    return dict(zip(frame["method"], frame["silhouette_point_estimate"].astype(float)))


def rank_partition(log: str, variants: pd.DataFrame) -> dict:
    """A2V at d = P7 rank: saved assignment (BPI 2012) or dense refit (BPI 2017/2019)."""
    rank = A2V_RANK[log]
    order = variants["variant_id"].tolist()
    if log == "bpi2012":
        k = 8
        name = f"a2v{rank}_matched_k{k}"
        path = EVAL / f"supp_extra_logs_v1/bpi2012/sources/assignments_activity2vec_log_context_d{rank}_k{k}.parquet"
    else:
        k = r3.MATCHED_K[log]
        name = f"a2v{rank}_table2_k{k}"
        path = REFIT / f"{log}__{name}.parquet"
        if not path.exists():
            REFIT.mkdir(parents=True, exist_ok=True)
            weights = variants["frequency"].to_numpy(dtype=np.float64)
            matrix = ordered_semantic_pyramid(
                parse_sequences(variants), activity_context_embeddings(variants, embedding_dim=rank))
            labels = weighted_kmeans(matrix, weights, n_clusters=k).labels
            pd.DataFrame({"variant_id": variants["variant_id"], "cluster_label": labels.astype("int32")}
                         ).to_parquet(path, index=False)
            say(f"{log}: refit {name}")
    return {"partition": name, "distance": f"a2v{rank}", "path": path,
            "labels": r3.labels_from(path, order),
            "published_method": f"activity2vec_log_context_d{rank}"}


def run(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPS.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(R3 / "audit_point_estimates.csv").set_index(["log", "partition"])
    checks = []
    for log in args.logs:
        log_started = time.time()
        plan = PLAN[log]
        variants, _ = r3.log_frame(log)
        order = variants["variant_id"].tolist()
        weights = variants["frequency"].to_numpy(dtype=np.float64)
        n = len(weights)
        parts = {p["partition"]: p for p in r3.partitions(log) if p["partition"] in plan}
        extra = rank_partition(log, variants)
        parts[extra["partition"]] = extra
        if set(parts) != set(plan):
            raise SystemExit(f"{log}: missing partitions {set(plan) - set(parts)}")
        sources = r3.distance_sources(log)
        rank = A2V_RANK[log]
        sources[f"a2v{rank}"] = lambda: r3.cosine(ordered_semantic_pyramid(
            parse_sequences(variants), activity_context_embeddings(variants, embedding_dim=rank)))
        published = _published_points(log)

        # P7 partition identity against the frozen assignments
        p7_name = next(name for name in plan if name.startswith("p7_"))
        k = int(np.unique(parts[p7_name]["labels"]).size)
        for frozen in _frozen_p7_files(log, k):
            ari = float(adjusted_rand_score(r3.labels_from(frozen, order), parts[p7_name]["labels"]))
            checks.append({"log": log, "check": "p7_ari_vs_frozen", "partition": p7_name,
                           "reference": str(frozen.relative_to(ROOT)), "value": ari,
                           "tolerance": 0.0, "passed": ari == 1.0})
            say(f"{log}: ARI({p7_name}, {frozen.name}) = {ari:.6f}")
            if ari != 1.0:
                raise SystemExit("P7 partition differs from the frozen assignment")

        reps_full = max(plan.values())
        draws = _resample_indices(n, reps_full, SEED, None)
        counts_full = counts_from_draws(draws, n)
        del draws
        reps_cap = CAPPED_SHAPE[log]
        capped = _resample_indices(n, reps_cap, SEED, CAP)
        counts_cap = counts_from_draws(capped, n)
        del capped
        say(f"{log}: n={n} cases={int(weights.sum())} full-size draws {reps_full} x {n}, "
            f"capped draws {reps_cap} x {CAP}, seed {SEED}")

        for source in dict.fromkeys(parts[name]["distance"] for name in plan):
            group = [name for name in plan if parts[name]["distance"] == source]
            started = time.time()
            path = SCRATCH / f"{log}_{source}_distances.npy"
            distances = sources[source]()
            np.save(path, distances)
            del distances
            dist = np.load(path, mmap_mode="r")
            say(f"{log} {source}: distance matrix {dist.shape} {dist.dtype} in {time.time() - started:.0f}s")
            for name in group:
                labels = parts[name]["labels"]
                t0 = time.time()
                # (i) point estimate
                point = silhouette_batch(dist, labels, weights, np.ones((1, n), dtype=np.uint16))[0][0]
                if (log, name) in audit.index:
                    ref = float(audit.loc[(log, name), "new_silhouette"])
                    ref_name = "review3_fixes_v1/audit_point_estimates.csv"
                else:
                    ref = published[parts[name]["published_method"]]
                    ref_name = "published source_silhouette_intervals.csv point"
                checks.append({"log": log, "check": "point_vs_reference", "partition": name,
                               "reference": ref_name, "value": abs(point - ref), "tolerance": 1e-7,
                               "passed": abs(point - ref) <= 1e-7})
                # (ii) capped replicates (m = 2,500, same draws as review3) on the same engine
                reps = plan[name]
                t0 = time.time()
                cap_values, cap_min, cap_valid = run_batches(dist, labels, weights, counts_cap[:reps])
                cap_seconds = time.time() - t0
                ref_path = R3 / "replicates" / f"{log}__{name}.npz"
                gap_text = "no review3 replicates"
                if ref_path.exists():
                    ref_reps = np.load(ref_path)
                    common = min(reps, ref_reps["new"].size)
                    both = cap_valid[:common] & ref_reps["valid"][:common]
                    gap = float(np.max(np.abs(cap_values[:common][both] - ref_reps["new"][:common][both])))
                    same_valid = bool(np.array_equal(cap_valid[:common], ref_reps["valid"][:common]))
                    checks.append({"log": log, "check": f"capped_all{common}_vs_review3_replicates",
                                   "partition": name, "reference": f"review3_fixes_v1/replicates/{log}__{name}.npz",
                                   "value": gap, "tolerance": 1e-7, "passed": gap <= 1e-7 and same_valid})
                    gap_text = f"capped max gap vs review3 {gap:.2e} over {common}"
                # (iii) library on two full-size draws (not for BPI 2019: 11,928^2 float64 copies)
                if log != "bpi2019":
                    full_draws = _resample_indices(n, 2, SEED, None)
                    if not np.array_equal(counts_from_draws(full_draws, n), counts_full[:2]):
                        raise SystemExit("draw matrix is not filled row-major")
                    lib_gap = 0.0
                    for row, indices in enumerate(full_draws):
                        sub = np.asarray(dist[np.ix_(indices, indices)])
                        lib = frequency_weighted_silhouette(sub, labels[indices], weights[indices])
                        del sub
                        mine = silhouette_batch(dist, labels, weights, counts_full[row:row + 1])[0][0]
                        lib_gap = max(lib_gap, abs(lib - mine))
                    checks.append({"log": log, "check": "full_size_2draws_vs_library", "partition": name,
                                   "reference": "frequency_weighted_silhouette on expanded indices",
                                   "value": lib_gap, "tolerance": 1e-9, "passed": lib_gap <= 1e-9})
                say(f"  {name}: point {point:.9f} (ref {ref:.9f}); {gap_text}; capped {reps} in {cap_seconds:.0f}s")
                # full-size replicates
                t0 = time.time()
                values, min_weight, valid = run_batches(dist, labels, weights, counts_full[:reps])
                elapsed = time.time() - t0
                np.savez(REPS / f"{log}__{name}.npz", new=values, min_cluster_weight=min_weight,
                         valid=valid, sample_size=n, seed=SEED, point=point, seconds=elapsed,
                         capped=cap_values, capped_min_cluster_weight=cap_min, capped_valid=cap_valid,
                         capped_sample_size=CAP, capped_seconds=cap_seconds)
                say(f"  {name}: {reps} full-size replicates in {elapsed:.0f}s "
                    f"(valid {int(valid.sum())}, weight<=1 clusters {int((min_weight[valid] <= 1).sum())})")
            del dist
            path.unlink()
        say(f"{log}: done in {time.time() - log_started:.0f}s")
    frame = pd.DataFrame(checks)
    target = OUT / "engine_checks.csv"
    if target.exists():
        previous = pd.read_csv(target)
        frame = pd.concat([previous[~previous["log"].isin(args.logs)], frame], ignore_index=True)
    frame.to_csv(target, index=False)
    if not frame["passed"].astype(bool).all():
        raise SystemExit("engine check failed:\n" + frame.loc[~frame["passed"].astype(bool)].to_string())


# --------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------

# name, log, first, second, replicates, review3 analysis label of the capped counterpart
# (None: no 5,000-replicate capped run was published), role
CONTRASTS = []
for _log, _k in (("bpi2017", 2), ("bpi2019", 6)):
    _r = A2V_RANK[_log]
    CONTRASTS += [
        ("table2_p7_minus_raw", _log, f"p7_table2_k{_k}", f"raw_table2_k{_k}", 300, "task2_table2_nominal_300", "table2_nominal_95"),
        ("table2_p7_minus_a2v16", _log, f"p7_table2_k{_k}", f"a2v16_table2_k{_k}", 300, "task2_table2_nominal_300", "table2_nominal_95"),
        (f"text_p7_minus_a2v{_r}", _log, f"p7_table2_k{_k}", f"a2v{_r}_table2_k{_k}", 300, "published_300", "text_nominal_95"),
        ("bonferroni_p7_minus_raw", _log, f"p7_table2_k{_k}", f"raw_table2_k{_k}", 5000, "task2_exact_table2_5000", "bonferroni_9875"),
        ("bonferroni_tagger_minus_raw", _log, f"tagger_k{_k}", "raw_fullbench_dynamic", 5000, "task3_tagger_minus_raw_5000", "bonferroni_9875"),
        ("secondary_tagger_minus_p7", _log, f"tagger_k{_k}", f"p7_table2_k{_k}", 5000, "task3_tagger_minus_p7_5000", "secondary_bonferroni_9875"),
        ("secondary_p7_minus_raw_review_run_partitions", _log, f"p7_table2_k{_k}", "raw_fullbench_dynamic", 5000,
         "review_bootstrap_tails_v1_partitions_5000", "secondary"),
        ("robustness_p7_minus_a2v16_5000", _log, f"p7_table2_k{_k}", f"a2v16_table2_k{_k}", 5000, None, "robustness_5000"),
        (f"robustness_p7_minus_a2v{_r}_5000", _log, f"p7_table2_k{_k}", f"a2v{_r}_table2_k{_k}", 5000, None, "robustness_5000"),
    ]
CONTRASTS += [
    ("supp_bpi2012_p7_minus_raw", "bpi2012", "p7_matched_k8", "raw_matched_k8", 400, "supp_extra_logs_bpi2012_400", "supplement"),
    ("supp_bpi2012_p7_minus_a2v16", "bpi2012", "p7_matched_k8", "a2v16_matched_k8", 400, "supp_extra_logs_bpi2012_400", "supplement"),
    ("supp_bpi2012_p7_minus_a2v23", "bpi2012", "p7_matched_k8", "a2v23_matched_k8", 400, "published_400", "supplement"),
]

# review3 names the P7 partition of the tagger / review-run contrasts p7_fullbench_* (alias of p7_table2_*)
R3_FIRST_SECOND = {
    "secondary_tagger_minus_p7": lambda log, k: (f"tagger_k{k}", "p7_fullbench_dynamic"),
    "secondary_p7_minus_raw_review_run_partitions": lambda log, k: ("p7_fullbench_dynamic", "raw_fullbench_dynamic"),
}


def _excl(lo: float, hi: float) -> bool:
    return bool(lo > 0 or hi < 0)


def _published_pairs(log: str) -> pd.DataFrame:
    if log == "bpi2012":
        path = EVAL / "supp_extra_logs_v1/bpi2012/sources/source_paired_differences.csv"
    else:
        path = EVAL / "representation_sources_extra/source_paired_differences.csv"
    frame = pd.read_csv(path)
    return frame.loc[frame["dataset"].eq(log)].set_index("second")


def summarize(_: argparse.Namespace) -> None:
    old_all = pd.read_csv(R3 / "bootstrap_contrasts_all.csv")
    old_all = old_all[old_all["convention"].eq("new")]
    rows, ref_checks = [], []
    for name, log, first, second, reps, r3_analysis, role in CONTRASTS:
        a = np.load(REPS / f"{log}__{first}.npz")
        b = np.load(REPS / f"{log}__{second}.npz")
        point = float(a["point"]) - float(b["point"])
        valid = a["valid"][:reps] & b["valid"][:reps]
        diff = (a["new"][:reps] - b["new"][:reps])[valid]
        cvalid = a["capped_valid"][:reps] & b["capped_valid"][:reps]
        cdiff = (a["capped"][:reps] - b["capped"][:reps])[cvalid]
        k = int(first.rsplit("_k", 1)[1]) if "_k" in first else None
        nominal = _percentile_interval(point, diff, CONFIDENCE)
        adjusted = _percentile_interval(point, diff, BONFERRONI)
        c_nominal = _percentile_interval(point, cdiff, CONFIDENCE)
        c_adjusted = _percentile_interval(point, cdiff, BONFERRONI)
        # cross-check the capped side against the frozen published / review3 numbers
        if r3_analysis and r3_analysis.startswith("published_"):
            method = "activity2vec_log_context_d" + second.split("_")[0][3:]
            pub = _published_pairs(log).loc[method]
            refs = {"point": float(pub["difference_point_estimate"]),
                    "ci95_lower": float(pub["difference_ci_lower"]), "ci95_upper": float(pub["difference_ci_upper"])}
            ref_source = "representation_sources_extra or supp_extra_logs_v1 source_paired_differences.csv"
        elif r3_analysis:
            r3_first, r3_second = R3_FIRST_SECOND.get(name, lambda *_: (first, second))(log, k)
            old = old_all[(old_all["analysis"] == r3_analysis) & (old_all["log"] == log)
                          & (old_all["first"] == r3_first) & (old_all["second"] == r3_second)
                          & (old_all["replicates_requested"] == reps)]
            if len(old) != 1:
                raise SystemExit(f"{name} {log}: {len(old)} capped reference rows")
            old = old.iloc[0]
            refs = {"point": float(old["point_estimate"]),
                    "ci95_lower": float(old["ci95_lower"]), "ci95_upper": float(old["ci95_upper"]),
                    "bonf9875_lower": float(old["bonf9875_lower"]), "bonf9875_upper": float(old["bonf9875_upper"])}
            ref_source = f"review3_fixes_v1/bootstrap_contrasts_all.csv [{r3_analysis}]"
        else:
            refs, ref_source = {}, "none (capped 5,000-replicate run not previously published)"
        mine = {"point": point, "ci95_lower": c_nominal.lower, "ci95_upper": c_nominal.upper,
                "bonf9875_lower": c_adjusted.lower, "bonf9875_upper": c_adjusted.upper}
        tolerance = 1e-6 if r3_analysis and r3_analysis.startswith("published_") else 1e-7
        for key, value in refs.items():
            gap = abs(mine[key] - value)
            ref_checks.append({"log": log, "check": f"capped_{key}_vs_reference", "partition": f"{name}",
                               "reference": ref_source, "value": gap, "tolerance": tolerance,
                               "passed": gap <= tolerance})
        single = ((a["min_cluster_weight"][:reps] <= 1) | (b["min_cluster_weight"][:reps] <= 1))[valid]
        record = {
            "contrast": name, "role": role, "log": log, "first": first, "second": second,
            "n_variants": int(a["sample_size"]), "capped_m": CAP,
            "replicates": reps, "seed": SEED, "capped_reference": ref_source,
            "capped_valid_replicates": int(cdiff.size),
            "full_valid_replicates": int(diff.size),
            "full_replicates_with_cluster_weight_le_1": int(single.sum()),
            "point_estimate": point,
            "capped_ci95_lower": c_nominal.lower, "capped_ci95_upper": c_nominal.upper,
            "full_ci95_lower": nominal.lower, "full_ci95_upper": nominal.upper,
            "capped_bonf9875_lower": c_adjusted.lower, "capped_bonf9875_upper": c_adjusted.upper,
            "full_bonf9875_lower": adjusted.lower, "full_bonf9875_upper": adjusted.upper,
            "capped_share_gt_0": float(np.mean(cdiff > 0)), "full_share_gt_0": float(np.mean(diff > 0)),
        }
        record["capped_ci95_excludes_0"] = _excl(c_nominal.lower, c_nominal.upper)
        record["full_ci95_excludes_0"] = _excl(nominal.lower, nominal.upper)
        record["capped_bonf9875_excludes_0"] = _excl(c_adjusted.lower, c_adjusted.upper)
        record["full_bonf9875_excludes_0"] = _excl(adjusted.lower, adjusted.upper)
        primary = "bonf9875" if "bonferroni" in role else "ci95"
        record["reported_level"] = "98.75% (Bonferroni, family of 4)" if primary == "bonf9875" else "95% nominal"
        record["ci95_relation_to_zero_changes"] = record["capped_ci95_excludes_0"] != record["full_ci95_excludes_0"]
        record["bonf9875_relation_to_zero_changes"] = (
            record["capped_bonf9875_excludes_0"] != record["full_bonf9875_excludes_0"])
        record["relation_to_zero_changes"] = record[f"{primary}_relation_to_zero_changes"]
        record["width_ratio_full_over_capped"] = (
            (record[f"full_{primary}_upper"] - record[f"full_{primary}_lower"])
            / (record[f"capped_{primary}_upper"] - record[f"capped_{primary}_lower"]))
        rows.append(record)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "full_size_bootstrap_contrasts.csv", index=False)

    # BPI 2017: compare with the full-size run of review_bootstrap_tails_v1 (same draws, frozen raw)
    tails = np.load(TAILS / "replicates_bpi2017_full.npy")
    a = np.load(REPS / "bpi2017__p7_table2_k2.npz")
    b = np.load(REPS / "bpi2017__raw_fullbench_dynamic.npz")
    gap = float(np.max(np.abs((a["new"] - b["new"]) - tails)))
    checks = pd.read_csv(OUT / "engine_checks.csv")
    checks = checks[(checks["check"] != "bpi2017_full_size_vs_review_bootstrap_tails_v1")
                    & ~(checks["check"].str.startswith("capped_") & checks["check"].str.endswith("_vs_reference"))]
    checks = pd.concat([checks, pd.DataFrame(ref_checks), pd.DataFrame([{
        "log": "bpi2017", "check": "bpi2017_full_size_vs_review_bootstrap_tails_v1",
        "partition": "p7_table2_k2 - raw_fullbench_dynamic",
        "reference": "review_bootstrap_tails_v1/replicates_bpi2017_full.npy",
        "value": gap, "tolerance": 1e-7, "passed": gap <= 1e-7}])], ignore_index=True)
    checks.to_csv(OUT / "engine_checks.csv", index=False)
    pd.set_option("display.width", 250)
    show = frame[["contrast", "log", "replicates", "point_estimate", "capped_ci95_lower", "capped_ci95_upper",
                  "full_ci95_lower", "full_ci95_upper", "capped_bonf9875_lower", "capped_bonf9875_upper",
                  "full_bonf9875_lower", "full_bonf9875_upper", "relation_to_zero_changes"]]
    print(show.round(4).to_string(index=False))
    print(checks.to_string(index=False))
    if not checks["passed"].astype(bool).all():
        raise SystemExit("engine check failed")


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def manifest(_: argparse.Namespace) -> None:
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    inputs: set[Path] = set()
    for log, plan in PLAN.items():
        variants_dir = (ROOT / "datasets/processed/bpi2012" if log == "bpi2012"
                        else ROOT / "datasets/processed" / r3.DATASET_FOLDERS[log])
        inputs.add(variants_dir / "variants.parquet")
        _, bench = r3.log_frame(log)
        for name in ("embedding_index.parquet", "p7_ordered_embeddings.npy", "raw_ordered_embeddings.npy",
                     "activity_embedding_index.parquet"):
            if (bench / name).exists():
                inputs.add(bench / name)
        if log != "bpi2012":
            inputs.add(R3 / "cache" / f"activity_vectors_{log}.npz")
        for part in r3.partitions(log):
            if part["partition"] in plan:
                inputs.add(Path(part["path"]))
                inputs.add(R3 / "replicates" / f"{log}__{part['partition']}.npz")
        k = 8 if log == "bpi2012" else r3.MATCHED_K[log]
        inputs.update(_frozen_p7_files(log, k))
        rank_file = (EVAL / f"supp_extra_logs_v1/bpi2012/sources/assignments_activity2vec_log_context_d{A2V_RANK[log]}_k8.parquet"
                     if log == "bpi2012" else None)
        if rank_file is not None:
            inputs.add(rank_file)
    inputs.update({EVAL / "representation_sources_extra/source_paired_differences.csv",
                   EVAL / "representation_sources_extra/source_silhouette_intervals.csv",
                   EVAL / "supp_extra_logs_v1/bpi2012/sources/source_paired_differences.csv",
                   EVAL / "supp_extra_logs_v1/bpi2012/sources/source_silhouette_intervals.csv"})
    inputs.update({R3 / "audit_point_estimates.csv", R3 / "bootstrap_contrasts_all.csv",
                   TAILS / "replicates_bpi2017_full.npy"})
    code = [ROOT / "scripts/run_full_size_bootstrap.py", ROOT / "scripts/run_review3_fixes.py",
            ROOT / "scripts/run_representation_source_benchmark.py", ROOT / "scripts/run_statistical_uncertainty.py",
            ROOT / "src/llm_variant_clustering/evaluation.py", ROOT / "src/llm_variant_clustering/uncertainty.py",
            ROOT / "src/llm_variant_clustering/features.py", ROOT / "src/llm_variant_clustering/semantic_compiler.py",
            ROOT / "src/llm_variant_clustering/clustering.py", ROOT / "scripts/run_representation_uncertainty.py"]
    outputs = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "manifest.json")
    import sklearn
    record = {
        "schema_version": 1,
        "created_by": "scripts/run_full_size_bootstrap.py",
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "bootstrap": {"seed": SEED, "resampling": "n-out-of-n over unique variants, case weights kept",
                      "capped_reference_m": CAP, "plan": PLAN, "confidence": CONFIDENCE,
                      "bonferroni_confidence": BONFERRONI, "batch": BATCH, "row_chunk": ROW_CHUNK},
        "threads": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS")},
        "inputs": {_rel(p): _sha(p) for p in sorted(inputs)},
        "code": {_rel(p): _sha(p) for p in code},
        "outputs": {str(p.relative_to(OUT)).replace("\\", "/"): {"sha256": _sha(p), "bytes": p.stat().st_size}
                    for p in outputs},
    }
    (OUT / "manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {len(record['inputs'])} inputs, {len(outputs)} outputs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rp = sub.add_parser("run")
    rp.add_argument("--logs", nargs="+", default=list(PLAN))
    sub.add_parser("summarize")
    sub.add_parser("manifest")
    args = parser.parse_args()
    {"run": run, "summarize": summarize, "manifest": manifest}[args.command](args)


if __name__ == "__main__":
    main()
