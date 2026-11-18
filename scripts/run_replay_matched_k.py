"""Controlled held-out replay at P7's K (Table 3 matched-K check) and repeated splits.

Writes only to evaluation/replay_matched_k_v1/. Frozen artifacts under evaluation/
are read, never written. No LLM, no encoder: every partition is a saved assignment
(Table 2, Supp A, Supp E), cross-checked against refits from cached BGE vectors.

  python scripts/run_replay_matched_k.py partitions      # load + ARI checks
  python scripts/run_replay_matched_k.py splits          # Part B seeds + feasibility
  python scripts/run_replay_matched_k.py run --logs rtfm bpi2020 bpi2017 bpi2019
  python scripts/run_replay_matched_k.py summarize       # Part A/B/C tables
  python scripts/run_replay_matched_k.py manifest

The replay engine is llm_variant_clustering.process_evaluation
(evaluate_process_partition_holdout / evaluate_holdout_model, IM noise 0.2,
token-based replay). Workers run one log at a time in a fresh interpreter with
PYTHONHASHSEED=0 and cache one CSV per (log, split seed, method), so a native
crash only loses the method in progress.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import pandas as pd

from llm_variant_clustering.process_evaluation import (
    _clusters_cover_both_splits,
    _holdout_score,
    select_variant_holdout_split,
)

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"
OUT = EVAL / "replay_matched_k_v1"
PARTS = OUT / "partitions"
RUNS = OUT / "runs"
LOGDIR = OUT / "worker_logs"
BENCH = EVAL / "full_benchmark"
SOURCES = EVAL / "representation_sources"
REVIEW3 = EVAL / "review3_fixes_v1" / "partitions"
TAGGER = EVAL / "supp_semantic_tagger_v1" / "assignments"
SUPP_E = EVAL / "supp_llm_direct_clustering_v1"
FROZEN = "process_model_metrics_dynamic_heldout"

LOGS = {"bpi2020": 9, "bpi2017": 2, "bpi2019": 6, "rtfm": 2}
RUN_ORDER = ("rtfm", "bpi2020", "bpi2017", "bpi2019")
N_SPLITS = {"bpi2020": 20, "bpi2017": 20, "bpi2019": 10, "rtfm": 20}
TEST_FRACTION = 0.2
TOLERANCE = 0.05
NOISE = 0.2
METHODS = ("p7", "raw_label", "a2v_d16", "tagger", "ngram")
ALL_METHODS = METHODS + ("whole_log",)
COMPARATORS = ("raw_label", "a2v_d16", "tagger", "ngram", "whole_log")
# Part A only: the BPI 2017 raw dense refit differs from the saved partition by 2 variants.
EXTRA_FROZEN = {"bpi2017": ("raw_label_refit",)}
METRICS = ("fitness", "precision", "fitness_precision_f1")

# Values quoted in the task / paper (3 decimals) and exact Supp E / Supp A values (hash seed 0).
KNOWN = {
    ("bpi2020", "p7"): 0.606, ("bpi2017", "p7"): 0.431, ("bpi2019", "p7"): 0.381, ("rtfm", "p7"): 0.716,
    ("bpi2020", "raw_label"): 0.624, ("bpi2017", "raw_label"): 0.343,
    ("bpi2019", "raw_label"): 0.469, ("rtfm", "raw_label"): 0.686,
    ("bpi2020", "a2v_d16"): 0.608, ("bpi2017", "a2v_d16"): 0.410,
    ("bpi2019", "a2v_d16"): 0.397, ("rtfm", "a2v_d16"): 0.712,
    ("bpi2020", "tagger"): 0.547, ("bpi2017", "tagger"): 0.413, ("rtfm", "tagger"): 0.716,
    ("bpi2020", "whole_log"): 0.451, ("bpi2017", "whole_log"): 0.311,
    ("bpi2019", "whole_log"): 0.328, ("rtfm", "whole_log"): 0.662,
    ("bpi2020", "ngram"): 0.602, ("bpi2017", "ngram"): 0.413, ("rtfm", "ngram"): 0.701,
}
# Supp E method ids on the frozen split (heldout/<log>_heldout.csv).
SUPP_E_ID = {
    "p7": "p7_llm_semantic_bge_k{k}", "raw_label": "raw_activity_label_bge_k{k}",
    "a2v_d16": "activity2vec_log_context_d16_k{k}", "ngram": "ngram_tfidf_1_3_k{k}",
    "whole_log": "whole_log", "raw_label_refit": "raw_activity_label_pyramid_bge_dynamic",
}


# Published Table 3 (main_draft.tex, tab:heldout): method id, K, F1 per column.
ORIGINAL_TABLE3 = {
    "bpi2020": {"whole_log": ("whole_log", None, 0.451), "raw_label": ("raw_sequence_bge_dynamic", 6, 0.468),
                "ngram": ("ngram_tfidf_1_3_dynamic", 6, 0.573), "a2v_d16": ("activity2vec_log_context_d16", 9, 0.608),
                "p7": ("p7_ordered_pyramid_1_2_4_bge_dynamic", 9, 0.606)},
    "bpi2017": {"whole_log": ("whole_log", None, 0.311),
                "raw_label": ("raw_activity_label_pyramid_bge_dynamic", 2, 0.343),
                "ngram": ("ngram_tfidf_1_3_dynamic", 4, 0.521), "a2v_d16": ("activity2vec_log_context_d16", 2, 0.410),
                "p7": ("p7_ordered_pyramid_1_2_4_bge_dynamic", 2, 0.431)},
    "bpi2019": {"whole_log": ("whole_log", None, 0.328),
                "raw_label": ("raw_activity_label_pyramid_bge_dynamic", 5, 0.486),
                "ngram": ("ngram_tfidf_1_3_dynamic", 2, 0.291), "a2v_d16": ("activity2vec_log_context_d16", 6, 0.397),
                "p7": ("p7_ordered_pyramid_1_2_4_bge_dynamic", 6, 0.381)},
    "rtfm": {"whole_log": ("whole_log", None, 0.662),
             "raw_label": ("raw_activity_label_pyramid_bge_dynamic", 2, 0.686),
             "ngram": ("ngram_tfidf_1_3_dynamic", 3, 0.639), "a2v_d16": ("activity2vec_log_context_d16", 2, 0.712),
             "p7": ("p7_ordered_pyramid_1_2_4_bge_dynamic", 2, 0.716)},
}


def variants_moved(first, second) -> int:
    """Variants outside the best one-to-one label matching (0 means identical partitions)."""
    from scipy.optimize import linear_sum_assignment

    table = pd.crosstab(pd.Series(np.asarray(first)), pd.Series(np.asarray(second))).to_numpy()
    rows, cols = linear_sum_assignment(-table)
    return int(table.sum() - table[rows, cols].sum())


def method_id(method: str, k: int) -> str:
    return {
        "p7": f"p7_llm_semantic_bge_k{k}",
        "raw_label": f"raw_activity_label_bge_k{k}",
        "raw_label_refit": f"raw_activity_label_bge_dense_refit_k{k}",
        "a2v_d16": f"activity2vec_log_context_d16_k{k}",
        "tagger": f"p7_tagger_k{k}",
        "ngram": f"ngram_tfidf_1_3_k{k}",
        "whole_log": "whole_log",
    }[method]


def source_path(log: str, method: str) -> Path:
    k = LOGS[log]
    return {
        "p7": SOURCES / log / f"assignments_p7_llm_semantic_bge_k{k}.parquet",
        "raw_label": SOURCES / log / f"assignments_raw_activity_label_bge_k{k}.parquet",
        "raw_label_refit": REVIEW3 / f"{log}__raw_table2_k{k}.parquet",
        "a2v_d16": SOURCES / log / f"assignments_activity2vec_log_context_d16_k{k}.parquet",
        "tagger": TAGGER / f"assignments_p7_tagger_{log}_k{k}.parquet",
        "ngram": SUPP_E / "assignments" / log / f"assignments_ngram_tfidf_1_3_k{k}.parquet",
    }[method]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# pure helpers (tested)
# --------------------------------------------------------------------------


def cluster_frame(variant_ids, labels, mid: str) -> pd.DataFrame:
    return pd.DataFrame({
        "variant_id": [str(v) for v in variant_ids],
        "cluster_id": [f"{mid}_c{int(label):02d}" for label in labels],
    })


def variant_cases(case_map: pd.DataFrame) -> pd.Series:
    """Cases per variant, sorted by variant id, as in select_variant_holdout_split."""
    frame = case_map[["case_id", "variant_id"]].copy()
    frame["variant_id"] = frame["variant_id"].astype(str)
    return frame.groupby("variant_id", sort=True).size()


def split_for_seed(counts: pd.Series, seed: int, test_fraction: float = TEST_FRACTION) -> pd.DataFrame:
    """Split frame exactly as select_variant_holdout_split builds it for one seed."""
    is_test = [_holdout_score(v, seed) < test_fraction for v in counts.index]
    return pd.DataFrame({
        "variant_id": list(counts.index),
        "split": ["test" if t else "train" for t in is_test],
        "cases": [int(c) for c in counts.to_numpy()],
    })


def test_case_share(split: pd.DataFrame) -> float:
    return float(split.loc[split["split"].eq("test"), "cases"].sum() / split["cases"].sum())


def share_valid_seeds(counts: pd.Series, n: int, *, test_fraction: float = TEST_FRACTION,
                      tolerance: float = TOLERANCE, max_seed: int = 10_000) -> list[tuple[int, pd.DataFrame]]:
    """First n seeds (from 0) whose test case share lies in the paper's window.

    The bounds are computed with the same expressions as select_variant_holdout_split;
    cluster coverage is deliberately not required here.
    """
    low = max(0.0, test_fraction - tolerance)
    high = min(1.0, test_fraction + tolerance)
    found = []
    for seed in range(max_seed):
        split = split_for_seed(counts, seed, test_fraction)
        if low <= test_case_share(split) <= high:
            found.append((seed, split))
            if len(found) == n:
                return found
    raise ValueError(f"only {len(found)} share-valid seeds below {max_seed}")


def coverage(frame: pd.DataFrame, split: pd.DataFrame) -> dict:
    """Feasibility = every cluster has >=1 train and >=1 test variant."""
    merged = frame[["variant_id", "cluster_id"]].merge(
        split[["variant_id", "split"]], on="variant_id", how="left", validate="one_to_one")
    counts = merged.groupby(["cluster_id", "split"]).size().unstack(fill_value=0)
    for column in ("train", "test"):
        if column not in counts:
            counts[column] = 0
    no_test = sorted(counts.index[counts["test"].eq(0)].astype(str))
    no_train = sorted(counts.index[counts["train"].eq(0)].astype(str))
    feasible = not no_test and not no_train
    assert feasible == bool(_clusters_cover_both_splits(frame[["variant_id", "cluster_id"]],
                                                        split[["variant_id", "split"]]))
    return {"feasible": feasible, "clusters_without_test": ";".join(no_test),
            "clusters_without_train": ";".join(no_train),
            "min_test_variants": int(counts["test"].min()),
            "min_train_variants": int(counts["train"].min())}


def aggregate(rows: pd.DataFrame) -> dict:
    """Test-case-weighted mean over clusters (whole log: its single row)."""
    clusters = rows[rows["scope"].eq("cluster")]
    if clusters.empty:
        whole = rows[rows["scope"].eq("whole_log")].iloc[0]
        return {m: float(whole[m]) for m in METRICS}
    weights = clusters["test_cases"].to_numpy(dtype=float)
    return {m: float((clusters[m].to_numpy(dtype=float) * weights).sum() / weights.sum()) for m in METRICS}


def paired_summary(per_split: pd.DataFrame, first: str = "p7",
                   others=COMPARATORS) -> pd.DataFrame:
    """P7 minus X over splits where both are feasible (descriptive, no test)."""
    rows = []
    for log, group in per_split.groupby("log", sort=False):
        wide = group[group["feasible"]].pivot(index="hash_seed", columns="method",
                                              values="fitness_precision_f1")
        for other in others:
            if first not in wide or other not in wide:
                both = pd.Series(dtype=float)
            else:
                both = (wide[first] - wide[other]).dropna()
            n = int(both.size)
            rows.append({
                "log": log, "first": first, "second": other, "n_paired_splits": n,
                "median_diff": float(both.median()) if n else np.nan,
                "min_diff": float(both.min()) if n else np.nan,
                "max_diff": float(both.max()) if n else np.nan,
                "n_first_greater": int((both > 0).sum()),
                "n_ties": int((both == 0).sum()),
                "share_first_greater": float((both > 0).mean()) if n else np.nan,
            })
    return pd.DataFrame(rows)


def method_summary(per_split: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (log, method), group in per_split.groupby(["log", "method"], sort=False):
        ok = group[group["feasible"]]
        f1 = ok["fitness_precision_f1"]
        rows.append({
            "log": log, "method": method, "n_splits": int(group["hash_seed"].nunique()),
            "n_feasible": int(ok["hash_seed"].nunique()),
            "infeasible_seeds": ";".join(str(s) for s in group.loc[~group["feasible"], "hash_seed"]),
            "median_f1": float(f1.median()) if len(f1) else np.nan,
            "min_f1": float(f1.min()) if len(f1) else np.nan,
            "max_f1": float(f1.max()) if len(f1) else np.nan,
            "median_fitness": float(ok["fitness"].median()) if len(ok) else np.nan,
            "median_precision": float(ok["precision"].median()) if len(ok) else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def case_map_for(log: str, variant_ids: frozenset) -> pd.DataFrame:
    from run_heldout_hashseed_sensitivity import _case_map

    case_map = pd.read_parquet(_case_map(log))
    return case_map[case_map["variant_id"].astype(str).isin(variant_ids)].copy()


def frozen_split(log: str) -> pd.DataFrame:
    split = pd.read_parquet(BENCH / log / f"{FROZEN}_variant_split.parquet")
    split["variant_id"] = split["variant_id"].astype(str)
    return split


def frozen_manifest(log: str) -> dict:
    return json.loads((BENCH / log / f"{FROZEN}_manifest.json").read_text(encoding="utf-8"))


def load_frames(log: str, extras: bool = True) -> dict[str, pd.DataFrame]:
    names = METHODS + (EXTRA_FROZEN.get(log, ()) if extras else ())
    frames = {}
    for method in names:
        frame = pd.read_parquet(PARTS / f"{log}__{method}.parquet")
        frame["variant_id"] = frame["variant_id"].astype(str)
        frames[method] = frame[["variant_id", "cluster_id"]]
    return frames


# --------------------------------------------------------------------------
# partitions
# --------------------------------------------------------------------------


def _labels(path: Path, order: list[str]) -> np.ndarray:
    frame = pd.read_parquet(path)
    return frame.set_index(frame["variant_id"].astype(str))["cluster_label"].loc[order].to_numpy()


def stage_partitions(args: argparse.Namespace) -> None:
    from scipy.sparse import csr_matrix
    from sklearn.metrics import adjusted_rand_score

    from llm_variant_clustering.clustering import weighted_kmeans
    from llm_variant_clustering.features import ngram_tfidf_features, parse_sequences
    from run_review3_fixes import log_frame, rebuilt_matrix

    PARTS.mkdir(parents=True, exist_ok=True)
    checks, sizes = [], []
    for log in args.logs:
        k = LOGS[log]
        variants, _ = log_frame(log)
        order = variants["variant_id"].tolist()
        weights = variants["frequency"].to_numpy(dtype=np.float64)
        sequences = parse_sequences(variants)
        labels = {m: _labels(source_path(log, m), order) for m in METHODS + EXTRA_FROZEN.get(log, ())}

        def check(method, reference, ref_labels, note=""):
            checks.append({"log": log, "method": method, "k": k, "reference": reference,
                           "ari": float(adjusted_rand_score(labels[method], ref_labels)),
                           "variants_moved": variants_moved(labels[method], ref_labels), "note": note})

        check("p7", "full_benchmark dynamic P7 (Table 3 P7)",
              _labels(BENCH / log / "assignments_p7_ordered_pyramid_1_2_4_bge_dynamic.parquet", order))
        check("p7", "review3 dense refit (Table 2 interval partition)",
              _labels(REVIEW3 / f"{log}__p7_table2_k{k}.parquet", order))
        check("raw_label", "review3 dense refit (Table 2 interval partition)",
              _labels(REVIEW3 / f"{log}__raw_table2_k{k}.parquet", order))
        check("a2v_d16", "review3 dense refit (Table 2 interval partition)",
              _labels(REVIEW3 / f"{log}__a2v16_table2_k{k}.parquet", order))
        if log == "bpi2020":
            check("tagger", "review3 dense refit (BPI 2020 secondary contrast)",
                  _labels(REVIEW3 / f"{log}__tagger_k{k}.parquet", order))
        started = time.time()
        tagger_refit = weighted_kmeans(rebuilt_matrix(log, "tagger", variants, sequences), weights,
                                       n_clusters=k).labels
        check("tagger", "dense refit from cached tagger-descriptor BGE vectors (this script)",
              tagger_refit, f"{time.time() - started:.0f} s")
        started = time.time()
        ngram_refit = weighted_kmeans(csr_matrix(ngram_tfidf_features(variants).matrix), weights,
                                      n_clusters=k).labels
        check("ngram", "recomputed TF-IDF 1-3 + weighted k-means, seeds (17,29,43,59,71), n_init 10",
              ngram_refit, f"{time.time() - started:.0f} s")
        check("p7", "tagger partition (overlap check, not expected to be 1)", labels["tagger"])
        if log in EXTRA_FROZEN:
            check("raw_label", "raw_label_refit (secondary Part A partition)", labels["raw_label_refit"])
        for method, values in labels.items():
            mid = method_id(method, k)
            frame = cluster_frame(order, values, mid)
            frame["cluster_label"] = values.astype("int64")
            frame.to_parquet(PARTS / f"{log}__{method}.parquet", index=False)
            weight = pd.Series(weights).groupby(values).sum()
            count = pd.Series(values).value_counts()
            sizes.append({"log": log, "method": method, "method_id": mid, "k_requested": k,
                          "k_observed": int(np.unique(values).size),
                          "min_cluster_variants": int(count.min()),
                          "min_cluster_cases": int(weight.min()),
                          "source": str(source_path(log, method).relative_to(ROOT)),
                          "source_sha256": _sha256(source_path(log, method))})
        print(f"{log}: partitions written", flush=True)
    frame = pd.DataFrame(checks)
    for name, new in (("partition_checks.csv", frame), ("partition_sources.csv", pd.DataFrame(sizes))):
        path = OUT / name
        if path.exists():
            old = pd.read_csv(path)
            new = pd.concat([old[~old["log"].isin(args.logs)], new], ignore_index=True)
        new.to_csv(path, index=False)
    print(frame.to_string(), flush=True)


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------


def plan_for(log: str, n: int) -> dict:
    """Seeds to evaluate: Part B seeds, the frozen seed, the all-method paper-rule seed."""
    frames = load_frames(log, extras=False)
    variant_ids = frozenset(frames["p7"]["variant_id"])
    case_map = case_map_for(log, variant_ids)
    counts = variant_cases(case_map)
    seeds = share_valid_seeds(counts, n)
    frozen = frozen_split(log)
    frozen_seed = int(frozen_manifest(log)["split_seed"])
    structural = {m: f for m, f in frames.items()
                  if int(f.groupby("cluster_id")["variant_id"].size().min()) >= 2}
    try:
        all_rule, all_seed = select_variant_holdout_split(case_map, structural, test_fraction=TEST_FRACTION,
                                                          case_fraction_tolerance=TOLERANCE)
        all_seed = int(all_seed)
    except ValueError:
        all_rule, all_seed = None, None
    return {"case_map": case_map, "counts": counts, "seeds": seeds, "frozen": frozen,
            "frozen_seed": frozen_seed, "all_rule_seed": all_seed, "all_rule_split": all_rule,
            "all_rule_methods": sorted(structural),
            "structurally_infeasible": sorted(set(frames) - set(structural)), "frames": frames}


def stage_splits(args: argparse.Namespace) -> None:
    rows, recovery = [], {}
    for log in args.logs:
        n = N_SPLITS[log]
        plan = plan_for(log, n)
        frames = load_frames(log)
        counts, frozen = plan["counts"], plan["frozen"]
        if frozenset(frozen["variant_id"]) != frozenset(counts.index):
            raise SystemExit(f"{log}: frozen split variants differ from the partition variants")
        manifest = frozen_manifest(log)
        frozen_frames = {m: pd.read_parquet(BENCH / log / f"assignments_{m}.parquet")
                         for m in manifest["methods"]}
        recovered, recovered_seed = select_variant_holdout_split(
            plan["case_map"], frozen_frames, test_fraction=TEST_FRACTION, case_fraction_tolerance=TOLERANCE)
        own = split_for_seed(counts, plan["frozen_seed"])
        part_b = [s for s, _ in plan["seeds"]]
        recovery[log] = {
            "frozen_split_seed_manifest": plan["frozen_seed"],
            "recovered_seed_with_frozen_methods": int(recovered_seed),
            "recovered_split_equals_frozen_parquet": bool(recovered.reset_index(drop=True).equals(frozen.reset_index(drop=True))),
            "split_for_seed_equals_frozen_parquet": bool(own.equals(frozen.reset_index(drop=True))),
            "frozen_test_variants": int(frozen["split"].eq("test").sum()),
            "frozen_test_case_share": test_case_share(frozen),
            "frozen_seed_in_part_b": plan["frozen_seed"] in part_b,
            "part_b_seeds": part_b,
            "part_b_n_splits": n,
            "all_methods_paper_rule_seed": plan["all_rule_seed"],
            "all_methods_paper_rule_methods": plan["all_rule_methods"],
            "structurally_infeasible_methods": plan["structurally_infeasible"],
            "all_methods_paper_rule_seed_in_part_b": plan["all_rule_seed"] in part_b,
        }
        if not (recovery[log]["recovered_split_equals_frozen_parquet"]
                and recovery[log]["split_for_seed_equals_frozen_parquet"]
                and recovered_seed == plan["frozen_seed"]):
            raise SystemExit(f"{log}: frozen split not reproduced: {recovery[log]}")
        evaluated = dict(plan["seeds"])
        if plan["all_rule_seed"] is not None:
            evaluated.setdefault(plan["all_rule_seed"], plan["all_rule_split"])
        if plan["frozen_seed"] not in evaluated:
            raise SystemExit(f"{log}: frozen seed {plan['frozen_seed']} is not among the Part B seeds")
        for seed in sorted(evaluated):
            split = evaluated[seed]
            base = {"log": log, "hash_seed": seed,
                    "part_b_rank": part_b.index(seed) if seed in part_b else None,
                    "in_part_b": seed in part_b, "is_frozen_seed": seed == plan["frozen_seed"],
                    "is_all_methods_paper_rule_seed": seed == plan["all_rule_seed"],
                    "test_variants": int(split["split"].eq("test").sum()),
                    "train_variants": int(split["split"].eq("train").sum()),
                    "test_case_share": test_case_share(split)}
            for method, frame in frames.items():
                if method in EXTRA_FROZEN.get(log, ()) and seed != plan["frozen_seed"]:
                    continue
                rows.append({**base, "method": method, **coverage(frame, split)})
            rows.append({**base, "method": "whole_log", "feasible": True, "clusters_without_test": "",
                         "clusters_without_train": "", "min_test_variants": base["test_variants"],
                         "min_train_variants": base["train_variants"]})
        print(log, json.dumps({k: v for k, v in recovery[log].items() if k != "part_b_seeds"}), flush=True)
    previous = OUT / "splits_partB.csv"
    frame = pd.DataFrame(rows)
    if previous.exists():
        old = pd.read_csv(previous)
        frame = pd.concat([old[~old["log"].isin(args.logs)], frame], ignore_index=True)
    frame.to_csv(previous, index=False)
    path = OUT / "split_recovery.json"
    merged = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    merged.update(recovery)
    path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# worker (one log, fresh interpreter, PYTHONHASHSEED fixed by the launcher)
# --------------------------------------------------------------------------


def load_events(log: str, case_map: pd.DataFrame) -> pd.DataFrame:
    import pm4py

    from run_heldout_hashseed_sensitivity import _xes

    events = pm4py.read_xes(str(_xes(log)), return_legacy_log_object=False, show_progress_bar=False)
    cases = set(case_map["case_id"].astype(str))
    # The replay uses only case, activity and timestamp; dropping the other attributes
    # keeps row order and saves memory (validated by the frozen-split reproduction).
    events = events.loc[events["case:concept:name"].astype(str).isin(cases),
                        ["case:concept:name", "concept:name", "time:timestamp"]].copy()
    if events["case:concept:name"].astype(str).nunique() != len(cases):
        raise ValueError(f"{log}: events do not cover every selected case")
    return events


def evaluate(events, case_map, split, frame, method) -> pd.DataFrame:
    from llm_variant_clustering.process_evaluation import (
        evaluate_holdout_model,
        evaluate_process_partition_holdout,
    )

    if frame is None:
        by_variant = dict(zip(split["variant_id"], split["split"], strict=True))
        by_case = dict(zip(case_map["case_id"].astype(str),
                           case_map["variant_id"].astype(str).map(by_variant), strict=True))
        labels = events["case:concept:name"].astype(str).map(by_case)
        return pd.DataFrame([{
            "method": "whole_log", "scope": "whole_log", "cluster_id": "whole_log",
            "noise_threshold": NOISE,
            "train_variants": int(split["split"].eq("train").sum()),
            "test_variants": int(split["split"].eq("test").sum()),
            **evaluate_holdout_model(events[labels.eq("train")].copy(), events[labels.eq("test")].copy(),
                                     noise_threshold=NOISE)}])
    return evaluate_process_partition_holdout(events, case_map, frame, split, method=method,
                                              noise_threshold=NOISE)


def stage_worker(args: argparse.Namespace) -> None:
    log = args.log
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("worker requires PYTHONHASHSEED=0 (IM discovery is hash-seed dependent)")
    plan_rows = pd.read_csv(OUT / "splits_partB.csv")
    plan_rows = plan_rows[plan_rows["log"].eq(log)]
    frames = load_frames(log)
    variant_ids = frozenset(frames["p7"]["variant_id"])
    case_map = case_map_for(log, variant_ids)
    counts = variant_cases(case_map)
    frozen_seed = int(frozen_manifest(log)["split_seed"])
    seeds = sorted(plan_rows["hash_seed"].unique())
    seeds = sorted(seeds, key=lambda s: (s != frozen_seed,
                                         not bool(plan_rows.loc[plan_rows["hash_seed"].eq(s),
                                                                "is_all_methods_paper_rule_seed"].any()), s))
    todo = []
    for seed in seeds:
        rows = plan_rows[plan_rows["hash_seed"].eq(seed)]
        for _, row in rows.iterrows():
            target = RUNS / log / f"seed{seed}" / f"{row['method']}.csv"
            if bool(row["feasible"]) and not target.exists():
                todo.append((seed, row["method"], target))
    print(f"[worker] {log}: {len(todo)} evaluations pending", flush=True)
    if not todo:
        return
    started = time.time()
    events = load_events(log, case_map)
    print(f"[worker] {log}: {len(events)} events loaded in {time.time() - started:.0f} s", flush=True)
    splits: dict[int, pd.DataFrame] = {}
    for seed, method, target in todo:
        if seed not in splits:
            split = split_for_seed(counts, seed)
            expected = plan_rows.loc[plan_rows["hash_seed"].eq(seed), "test_variants"].iloc[0]
            if int(split["split"].eq("test").sum()) != int(expected):
                raise SystemExit(f"{log} seed {seed}: split differs from the plan")
            if seed == frozen_seed and not split.equals(frozen_split(log).reset_index(drop=True)):
                raise SystemExit(f"{log}: frozen split not reproduced")
            splits = {seed: split}
        tick = time.time()
        rows = evaluate(events, case_map, splits[seed], None if method == "whole_log" else frames[method],
                        method_id(method, LOGS[log]))
        rows.insert(0, "method_key", method)
        rows.insert(0, "hash_seed", seed)
        rows.insert(0, "log", log)
        rows["pythonhashseed"] = os.environ.get("PYTHONHASHSEED")
        rows["runtime_seconds"] = round(time.time() - tick, 1)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(target.with_suffix(".tmp"), index=False)
        os.replace(target.with_suffix(".tmp"), target)
        print(f"[worker] {log} seed {seed} {method}: F1={aggregate(rows)['fitness_precision_f1']:.4f} "
              f"({time.time() - tick:.0f} s, total {time.time() - started:.0f} s)", flush=True)


def stage_run(args: argparse.Namespace) -> None:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "2", "PYTHONPATH": os.pathsep.join(
               [str(ROOT / "scripts"), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    for log in [x for x in RUN_ORDER if x in args.logs]:
        for attempt in range(1, args.retries + 1):
            tick = time.time()
            with (LOGDIR / f"{log}.log").open("a", encoding="utf-8") as handle:
                handle.write(f"=== attempt {attempt} {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                handle.flush()
                code = subprocess.call([sys.executable, str(Path(__file__).resolve()), "worker", "--log", log],
                                       env=env, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT)
                handle.write(f"=== exit {code} after {time.time() - tick:.0f} s ===\n")
            print(f"{log}: attempt {attempt} exit {code} ({time.time() - tick:.0f} s)", flush=True)
            if code == 0:
                break
        else:
            raise SystemExit(f"{log}: worker failed {args.retries} times")


# --------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------


def _reference(log: str) -> dict:
    """Exact frozen-split values at PYTHONHASHSEED=0 from Supp E and Supp A."""
    k = LOGS[log]
    ref = {}
    supp_path = SUPP_E / "heldout" / f"{log}_heldout.csv"
    # Cross-check only; written by a GPT-5 supplementary study that is not released.
    supp = (pd.read_csv(supp_path) if supp_path.exists()
            else pd.DataFrame(columns=["split", "scope", "method"]))
    supp = supp[supp["split"].eq("frozen") & supp["scope"].isin(["test_case_weighted_aggregate", "whole_log"])]
    supp = supp.set_index("method")
    for method, template in SUPP_E_ID.items():
        name = template.format(k=k)
        if method == "raw_label_refit" and log != "bpi2017":
            continue
        if name in supp.index:
            ref[method] = {m: float(supp.loc[name, m]) for m in METRICS}
            ref[method]["source"] = f"supp_llm_direct_clustering_v1/heldout/{log}_heldout.csv:{name}"
    tagger = pd.read_csv(EVAL / "supp_semantic_tagger_v1" / "heldout_f1.csv").set_index("dataset")
    if log in tagger.index and tagger.loc[log, "status"] == "completed":
        ref["tagger"] = {"fitness": float(tagger.loc[log, "tagger_fitness"]),
                         "precision": float(tagger.loc[log, "tagger_precision"]),
                         "fitness_precision_f1": float(tagger.loc[log, "tagger_f1"]),
                         "source": "supp_semantic_tagger_v1/heldout_f1.csv"}
    return ref


def collect() -> pd.DataFrame:
    plan = pd.read_csv(OUT / "splits_partB.csv")
    rows = []
    for _, row in plan.iterrows():
        entry = row.to_dict()
        path = RUNS / row["log"] / f"seed{row['hash_seed']}" / f"{row['method']}.csv"
        entry["evaluated"] = path.exists()
        if path.exists():
            result = pd.read_csv(path)
            entry.update(aggregate(result))
            entry["runtime_seconds"] = float(result["runtime_seconds"].iloc[0])
            clusters = result[result["scope"].eq("cluster")]
            entry["n_clusters_evaluated"] = int(len(clusters)) if len(clusters) else None
            entry["test_cases"] = int(result["test_cases"].sum())
        rows.append(entry)
    return pd.DataFrame(rows)


def _infeasible_reason(row: pd.Series) -> str:
    parts = []
    for key, label in (("clusters_without_test", "without test variant"),
                       ("clusters_without_train", "without train variant")):
        value = row.get(key)
        if isinstance(value, str) and value:
            parts.append(f"cluster(s) {label}: {value}")
    return "infeasible: " + ("; ".join(parts) if parts else "coverage rule failed")


def stage_summarize(_: argparse.Namespace) -> None:
    allrows = collect()
    allrows["method_id"] = [method_id(m, LOGS[log]) for log, m in zip(allrows["log"], allrows["method"])]
    allrows["k"] = [None if m == "whole_log" else LOGS[log] for log, m in zip(allrows["log"], allrows["method"])]
    pending = allrows[allrows["feasible"] & ~allrows["evaluated"]]
    if len(pending):
        print(f"WARNING: {len(pending)} feasible evaluations missing", flush=True)

    # Part A: frozen split
    frozen = allrows[allrows["is_frozen_seed"]].copy()
    part_a = []
    for _, row in frozen.iterrows():
        ref = _reference(row["log"]).get(row["method"], {})
        known = KNOWN.get((row["log"], row["method"]))
        f1 = row.get("fitness_precision_f1")
        part_a.append({
            "log": row["log"], "split": "frozen", "hash_seed": row["hash_seed"], "method": row["method"],
            "method_id": row["method_id"], "k": row["k"], "feasible": bool(row["feasible"]),
            "clusters_without_test": row["clusters_without_test"],
            "clusters_without_train": row["clusters_without_train"],
            "fitness_precision_f1": f1, "fitness": row.get("fitness"), "precision": row.get("precision"),
            "known_value_3dp": known,
            "matches_known_3dp": None if known is None or pd.isna(f1) else bool(round(f1, 3) == known),
            "reference_exact_f1": ref.get("fitness_precision_f1"),
            "abs_diff_vs_reference": None if not ref or pd.isna(f1) else abs(f1 - ref["fitness_precision_f1"]),
            "reference_source": ref.get("source"), "runtime_seconds": row.get("runtime_seconds"),
        })
    part_a = pd.DataFrame(part_a)
    rule = allrows[allrows["is_all_methods_paper_rule_seed"] & ~allrows["is_frozen_seed"]].copy()
    secondary = rule.assign(split="all_methods_paper_rule")[
        ["log", "split", "hash_seed", "method", "method_id", "k", "feasible", "clusters_without_test",
         "clusters_without_train", "fitness_precision_f1", "fitness", "precision", "runtime_seconds"]]
    pd.concat([part_a, secondary], ignore_index=True).to_csv(OUT / "partA_frozen.csv", index=False)

    # Part B: Part B seeds only, Part B methods only
    part_b = allrows[allrows["in_part_b"] & allrows["method"].isin(ALL_METHODS)].copy()
    cols = ["log", "hash_seed", "part_b_rank", "is_frozen_seed", "test_variants", "test_case_share", "method",
            "method_id", "k", "feasible", "clusters_without_test", "clusters_without_train", "evaluated",
            "fitness_precision_f1", "fitness", "precision", "runtime_seconds"]
    part_b[cols].to_csv(OUT / "partB_per_split.csv", index=False)
    usable = part_b[part_b["evaluated"] | ~part_b["feasible"]]
    method_summary(usable).to_csv(OUT / "partB_summary.csv", index=False)
    paired_summary(usable[usable["evaluated"] | ~usable["feasible"]]).to_csv(OUT / "partB_paired.csv", index=False)

    # Part C: controlled Table 3 + column map
    table, colmap = [], []
    order = ("whole_log", "raw_label", "ngram", "a2v_d16", "tagger", "p7")
    original = ORIGINAL_TABLE3
    sources = pd.read_csv(OUT / "partition_sources.csv").set_index(["log", "method"])
    for split_name, block in (("frozen", part_a), ("all_methods_paper_rule", secondary)):
        for log in LOGS:
            sub = block[block["log"].eq(log)].set_index("method")
            if sub.empty:
                continue
            entry = {"log": log, "split": split_name, "hash_seed": int(sub["hash_seed"].iloc[0]), "p7_k": LOGS[log]}
            for method in order:
                feasible = method in sub.index and bool(sub.loc[method, "feasible"])
                value = sub.loc[method, "fitness_precision_f1"] if feasible else np.nan
                entry[f"{method}_f1"] = value
                entry[f"{method}_status"] = "ok" if feasible and pd.notna(value) else (
                    _infeasible_reason(sub.loc[method]) if method in sub.index and not feasible else "missing")
                if split_name != "frozen":
                    continue
                orig = original[log].get(method)
                colmap.append({
                    "log": log, "column": f"{method}_f1", "method_key": method,
                    "method_id": method_id(method, LOGS[log]),
                    "k": None if method == "whole_log" else LOGS[log],
                    "partition_source": "" if method == "whole_log" else sources.loc[(log, method), "source"],
                    "partition_sha256": "" if method == "whole_log" else sources.loc[(log, method), "source_sha256"],
                    "split": f"frozen (hash seed {entry['hash_seed']})",
                    "features": {"p7": "ordered pyramid 1,2,4 of BGE(P7 descriptor)",
                                 "raw_label": "ordered pyramid 1,2,4 of BGE(raw activity label)",
                                 "a2v_d16": "ordered pyramid 1,2,4 of PPMI-SVD activity context d16",
                                 "tagger": "ordered pyramid 1,2,4 of BGE(Supp A tagger descriptor)",
                                 "ngram": "TF-IDF activity n-grams 1-3 (no pyramid)",
                                 "whole_log": "single model"}[method],
                    "clustering": "" if method == "whole_log" else "case-weighted k-means, seeds (17,29,43,59,71), n_init 10",
                    "original_table3_method_id": orig[0] if orig else "not in Table 3",
                    "original_table3_k": orig[1] if orig else None,
                    "original_table3_f1": orig[2] if orig else None,
                    "same_partition_as_original": bool(orig) and orig[0] in (
                        "p7_ordered_pyramid_1_2_4_bge_dynamic", "activity2vec_log_context_d16", "whole_log"),
                })
            table.append(entry)
    pd.DataFrame(table).to_csv(OUT / "table3_matched_k.csv", index=False)
    pd.DataFrame(colmap).to_csv(OUT / "column_map.csv", index=False)

    timing = allrows[allrows["evaluated"]].groupby(["log", "method"])["runtime_seconds"].agg(
        ["count", "median", "sum"]).reset_index()
    timing.to_csv(OUT / "timings.csv", index=False)
    print(pd.read_csv(OUT / "partA_frozen.csv").to_string(), flush=True)
    print(pd.read_csv(OUT / "partB_summary.csv").to_string(), flush=True)
    print(pd.read_csv(OUT / "partB_paired.csv").to_string(), flush=True)


def stage_manifest(_: argparse.Namespace) -> None:
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "manifest.json"
                   and "_scratch" not in p.parts)
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "scripts/run_replay_matched_k.py",
        "script_sha256": _sha256(Path(__file__).resolve()),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "pythonhashseed_for_discovery": "0", "noise_threshold": NOISE,
        "llm_calls": 0,
        "files": {str(p.relative_to(OUT)).replace("\\", "/"): _sha256(p) for p in files},
    }
    try:
        import pm4py
        manifest["pm4py"] = pm4py.__version__
    except Exception:  # pragma: no cover
        pass
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {len(files)} files", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("partitions", "splits", "run"):
        stage = sub.add_parser(name)
        stage.add_argument("--logs", nargs="+", default=list(RUN_ORDER), choices=list(LOGS))
        if name == "run":
            stage.add_argument("--retries", type=int, default=4)
    worker = sub.add_parser("worker")
    worker.add_argument("--log", required=True, choices=list(LOGS))
    sub.add_parser("summarize")
    sub.add_parser("manifest")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"partitions": stage_partitions, "splits": stage_splits, "worker": stage_worker, "run": stage_run,
     "summarize": stage_summarize, "manifest": stage_manifest}[args.stage](args)


if __name__ == "__main__":
    main()
