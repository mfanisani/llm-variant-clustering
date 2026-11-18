"""Build the final paper tables (main_draft.tex) with cell-level provenance.

Successor of scripts/build_paper_tables.py (v1, kept unchanged). Differences:

* Table 2 adds the non-LLM tagger column (Supp A) and records, per row and
  contrast, which bootstrap produced the interval. It is emitted twice:
  ``table2_matched_k_silhouette.csv`` with the published intervals (capped
  m-out-of-n, m = 2,500, for BPI 2017 and BPI 2019) and
  ``table2_matched_k_silhouette_full_size.csv`` with full-size n-out-of-n
  intervals from evaluation/full_size_bootstrap_v1 for those two logs (the
  other logs have fewer than 2,500 variants and were never capped).
* Table 3 is the controlled matched-K replay (every method at the K of P7) from
  evaluation/replay_matched_k_v1/table3_matched_k.csv (frozen split). v1 used
  ``raw_sequence_bge_dynamic`` (whole-sequence embedding, own K) for BPI 2020 and
  the raw-label pyramid at its own K elsewhere; v2 uses the raw-label pyramid at
  the K of P7 on every log.
* ``provenance.csv`` maps every emitted value to its source file, column/method
  key and the SHA-256 of the source; every value is compared with the value
  printed in (or planned for) the paper with tolerance 0.0015, and the script
  exits non-zero on any mismatch.

Reads only existing outputs and writes only to --output-dir. No LLM calls.

  python scripts/build_paper_tables_v2.py [--output-dir evaluation/paper_tables_v2]
                                          [--wait-for-replay-minutes 0] [--require-complete]

Replay cells whose run has not finished (status "missing") are reported as pending,
not as mismatches; --require-complete turns them into a failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"
LOGS = ["bpi2020", "bpi2017", "bpi2019", "sepsis", "rtfm"]
CASE_MAPS = {
    "bpi2020": "datasets/processed/bpi2020_international_declarations/case_variant_map.parquet",
    "bpi2017": "datasets/processed/bpi2017/case_variant_map.parquet",
    "bpi2019": "datasets/processed/bpi2019/case_variant_map.parquet",
    "sepsis": "datasets/processed/sepsis/case_variant_map.parquet",
    "rtfm": "datasets/processed/rtfm/case_variant_map.parquet",
}
P7_BENCH = "p7_ordered_pyramid_1_2_4_bge"
NGRAM = "ngram_tfidf_1_3"
METHOD = {"p7": "p7_llm_semantic_bge", "raw": "raw_activity_label_bge",
          "a2v16": "activity2vec_log_context_d16"}
TOLERANCE = 0.0015
BOOTSTRAP_CAP = 2500
SEED = 20240601

BENCH_CSV = EVAL / "representation_sources/activity_source_benchmark.csv"
PAIRED_CSV = {log: EVAL / ("representation_sources" if log in ("bpi2020", "sepsis", "rtfm")
                           else "representation_sources_extra") / "source_paired_differences.csv"
              for log in LOGS}
INTERVAL_CSV = {log: path.with_name("source_silhouette_intervals.csv") for log, path in PAIRED_CSV.items()}
TAGGER_CSV = EVAL / "supp_semantic_tagger_v1/partition_metrics.csv"
FULL_SIZE_CSV = EVAL / "full_size_bootstrap_v1/full_size_bootstrap_contrasts.csv"
REPLAY_DIR = EVAL / "replay_matched_k_v1"
REPLAY_TABLE = REPLAY_DIR / "table3_matched_k.csv"
REPLAY_SPLITS = REPLAY_DIR / "splits_partB.csv"
REPLAY_COLUMNS = REPLAY_DIR / "column_map.csv"

LOG_NAMES = {"bpi2020": "BPI 2020", "bpi2017": "BPI 2017", "bpi2019": "BPI 2019",
             "sepsis": "Sepsis", "rtfm": "RTFM"}

# Values printed in GenAI4PM-version/Paper/main_draft.tex (Table 1 / tab:logs).
EXPECTED_TABLE1 = {
    # variants, cases, activities, automatic K of P7
    "bpi2020": (753, 6449, 34, "9"),
    "bpi2017": (5578, 31428, 24, "2"),
    "bpi2019": (11928, 251381, 42, "6"),
    "sepsis": (846, 1050, 16, "abstains"),
    "rtfm": (231, 150370, 11, "2"),
}
# Table 2 / tab:silhouette, published (capped) intervals.
EXPECTED_TABLE2 = {
    # k, p7, raw, a2v16, tagger, ngram, ngram_k, d_raw (pt, lo, hi), d_a2v16 (pt, lo, hi)
    "bpi2020": (9, 0.571, 0.539, 0.526, 0.567, 0.435, 6, (0.033, -0.028, 0.077), (0.045, -0.032, 0.097)),
    "bpi2017": (2, 0.628, 0.519, 0.652, 0.585, 0.323, 4, (0.109, 0.086, 0.118), (-0.023, -0.046, 0.005)),
    "bpi2019": (6, 0.469, 0.445, 0.389, 0.432, 0.319, 2, (0.024, -0.079, 0.059), (0.080, -0.239, 0.188)),
    "sepsis": (8, 0.316, 0.323, 0.329, 0.275, 0.146, 8, (-0.006, -0.021, 0.011), (-0.012, -0.032, 0.003)),
    "rtfm": (2, 0.882, 0.799, 0.840, 0.881, 0.771, 3, (0.083, 0.012, 0.132), (0.042, -0.020, 0.134)),
}
# Controlled matched-K Table 3 (every method at the K of P7, frozen split). None: no
# paper value to check (n-gram at the K of P7 is new); "infeasible": a cluster lacks
# test variants on the frozen split.
EXPECTED_TABLE3 = {
    # test variants, whole log, raw, a2v16, tagger, ngram, p7
    "bpi2020": (170, 0.451, 0.624, 0.608, 0.547, None, 0.606),
    "bpi2017": (1162, 0.311, 0.343, 0.410, 0.413, None, 0.431),
    "bpi2019": (2371, 0.328, 0.469, 0.397, "infeasible", None, 0.381),
    "rtfm": (46, 0.662, 0.686, 0.712, 0.716, None, 0.716),
}
TABLE3_METHODS = ("whole_log", "raw_label", "a2v_d16", "tagger", "ngram", "p7")
TABLE3_COLUMNS = {"whole_log": "whole_log", "raw_label": "raw", "a2v_d16": "activity2vec_d16",
                  "tagger": "tagger", "ngram": "ngram_at_p7_k", "p7": "p7"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


@dataclass
class Builder:
    provenance: list[dict] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)
    hashes: dict[str, str] = field(default_factory=dict)

    def _hash(self, path: Path) -> str:
        key = rel(path)
        if key not in self.hashes:
            self.hashes[key] = sha256(path)
        return self.hashes[key]

    def cell(self, table: str, row: str, column: str, value, source: Path, key: str,
             expected=None, exact: bool = False, note: str = ""):
        """Record provenance for one emitted value and check it against the paper value."""
        self.provenance.append({"table": table, "row": row, "column": column, "value": value,
                                "source_file": rel(source), "source_key": key,
                                "source_sha256": self._hash(source), "note": note})
        if expected is None:
            return value
        if exact or isinstance(expected, str):
            passed = value == expected
            gap = None
        else:
            gap = abs(float(value) - float(expected))
            passed = gap <= TOLERANCE
        self.checks.append({"table": table, "row": row, "column": column, "value": value,
                            "expected": expected, "abs_diff": gap,
                            "tolerance": "exact" if gap is None else TOLERANCE, "passed": bool(passed)})
        return value

    def pending_cell(self, table: str, row: str, column: str, status: str, expected, source: Path, key: str,
                     note: str = ""):
        """A replay cell whose run has not finished yet: recorded, neither passed nor a mismatch."""
        self.provenance.append({"table": table, "row": row, "column": f"{column}_status", "value": status,
                                "source_file": rel(source), "source_key": key,
                                "source_sha256": self._hash(source), "note": note})
        self.checks.append({"table": table, "row": row, "column": column, "value": status,
                            "expected": expected, "abs_diff": None, "tolerance": "exact",
                            "passed": False, "pending": True})

    @property
    def errors(self) -> list[str]:
        return [f"{c['table']} {c['row']} {c['column']}: {c['value']} != expected {c['expected']}"
                for c in self.checks if not c["passed"] and not c.get("pending", False)]

    @property
    def pending(self) -> list[str]:
        return [f"{c['table']} {c['row']} {c['column']} (expected {c['expected']})"
                for c in self.checks if c.get("pending", False)]


# --------------------------------------------------------------------------
# Table 1
# --------------------------------------------------------------------------


def table1(b: Builder) -> pd.DataFrame:
    rows = []
    for log in LOGS:
        exp = EXPECTED_TABLE1[log]
        cmap_path = ROOT / CASE_MAPS[log]
        dyn_path = EVAL / "full_benchmark" / log / "metrics_dynamic_k.csv"
        assign_path = EVAL / "full_benchmark" / log / f"assignments_{P7_BENCH}_k8.parquet"
        act_path = EVAL / "full_benchmark" / log / "activity_embedding_index.parquet"
        # Evaluated variants exclude the 45-variant calibration subsets of BPI 2017/2019.
        evaluated = set(pd.read_parquet(assign_path)["variant_id"].astype(str))
        cmap = pd.read_parquet(cmap_path, columns=["variant_id"])
        cmap = cmap[cmap["variant_id"].astype(str).isin(evaluated)]
        dyn = pd.read_csv(dyn_path)
        p7 = dyn[dyn["method"] == P7_BENCH].iloc[0]
        k = str(int(p7["selected_k"])) if p7["status"] == "selected" else "abstains"
        activities = pd.read_parquet(act_path)["activity_label"].nunique()
        row = {"log": log, "log_name": LOG_NAMES[log], "used_in_design": log in ("bpi2020", "bpi2017", "bpi2019")}
        t = "table1"
        row["unique_variants"] = b.cell(t, log, "unique_variants", int(cmap["variant_id"].nunique()), cmap_path,
                                        f"variant_id restricted to {rel(assign_path)}", exp[0], exact=True)
        row["cases"] = b.cell(t, log, "cases", int(len(cmap)), cmap_path,
                              f"rows restricted to {rel(assign_path)}", exp[1], exact=True)
        row["activities"] = b.cell(t, log, "activities", int(activities), act_path,
                                   "activity_label (distinct)", exp[2], exact=True)
        row["automatic_p7_k"] = b.cell(t, log, "automatic_p7_k", k, dyn_path,
                                       f"method={P7_BENCH}: status, selected_k", exp[3], exact=True)
        row["printed_k"] = "8 (exploratory, no automatic K)" if k == "abstains" else k
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Table 2
# --------------------------------------------------------------------------


def _bootstrap_label(log: str, n: int, sample_size: int, resamples: int, kind: str, source: str) -> str:
    if kind == "capped":
        what = f"capped m-out-of-n, m={sample_size} of n={n}, not rescaled"
    else:
        what = f"full-size n-out-of-n (n={n})"
    return f"{what}; {resamples} paired replicates; percentile 95%; {source}"


def table2(b: Builder) -> tuple[pd.DataFrame, pd.DataFrame]:
    bench = pd.read_csv(BENCH_CSV)
    tagger = pd.read_csv(TAGGER_CSV).set_index("dataset")
    full = pd.read_csv(FULL_SIZE_CSV) if FULL_SIZE_CSV.exists() else None
    published_rows, full_rows = [], []
    for log in LOGS:
        exp = EXPECTED_TABLE2[log]
        t = "table2"
        sil = bench[bench["dataset"] == log].set_index("method")
        k = int(sil.loc[METHOD["p7"], "matched_k"])
        row = {"log": log, "log_name": LOG_NAMES[log]}
        row["k"] = b.cell(t, log, "k", k, BENCH_CSV, f"dataset={log}, method={METHOD['p7']}: matched_k",
                          exp[0], exact=True)
        for column, method, value_exp in (("p7", "p7", exp[1]), ("raw", "raw", exp[2]),
                                          ("activity2vec_d16", "a2v16", exp[3])):
            name = METHOD[method]
            if int(sil.loc[name, "matched_k"]) != k:
                b.checks.append({"table": t, "row": log, "column": f"{column}_k", "value": int(sil.loc[name, "matched_k"]),
                                 "expected": k, "abs_diff": None, "tolerance": "exact", "passed": False})
            row[column] = b.cell(t, log, column, float(sil.loc[name, "matched_case_weighted_silhouette"]), BENCH_CSV,
                                 f"dataset={log}, method={name}: matched_case_weighted_silhouette", value_exp)
        if int(tagger.loc[log, "matched_k"]) != k:
            b.checks.append({"table": t, "row": log, "column": "tagger_k", "value": int(tagger.loc[log, "matched_k"]),
                             "expected": k, "abs_diff": None, "tolerance": "exact", "passed": False})
        row["tagger"] = b.cell(t, log, "tagger", float(tagger.loc[log, "tagger_silhouette"]), TAGGER_CSV,
                               f"dataset={log}: tagger_silhouette (K={k})", exp[4])
        ngram_path = EVAL / "full_benchmark" / log / ("metrics_fixed_k8.csv" if log == "sepsis" else "metrics_dynamic_k.csv")
        ng = pd.read_csv(ngram_path).set_index("method").loc[NGRAM]
        row["ngram_own_k"] = b.cell(t, log, "ngram_own_k", float(ng["case_weighted_silhouette"]), ngram_path,
                                    f"method={NGRAM}: case_weighted_silhouette", exp[5])
        ngram_k = 8 if log == "sepsis" else int(ng["selected_k"])
        row["ngram_k"] = b.cell(t, log, "ngram_k", ngram_k, ngram_path,
                                f"method={NGRAM}: selected_k" if log != "sepsis" else "fixed K=8 (metrics_fixed_k8.csv)",
                                exp[6], exact=True)

        intervals = pd.read_csv(INTERVAL_CSV[log])
        n_boot = intervals.loc[intervals["dataset"].eq(log), "bootstrap_sample_size"].astype(int)
        paired = pd.read_csv(PAIRED_CSV[log])
        paired = paired[(paired["dataset"] == log) & (paired["first"] == METHOD["p7"])].set_index("second")
        published = dict(row)
        full_size = dict(row)
        n_variants = EXPECTED_TABLE1[log][0]
        capped = bool((n_boot < n_variants).any())
        for label, method, pexp in (("p7_minus_raw", "raw", exp[7]), ("p7_minus_a2v16", "a2v16", exp[8])):
            r = paired.loc[METHOD[method]]
            key = f"dataset={log}, first={METHOD['p7']}, second={METHOD[method]}"
            values = []
            for part, column, e in zip(("point", "lower", "upper"),
                                       ("difference_point_estimate", "difference_ci_lower", "difference_ci_upper"), pexp):
                values.append(b.cell(t, log, f"{label}_{part}", float(r[column]), PAIRED_CSV[log],
                                     f"{key}: {column}", e))
            resamples = int(r["difference_resamples"])
            published[f"{label}_point"], published[f"{label}_ci95_lower"], published[f"{label}_ci95_upper"] = values
            published[f"{label}_excludes_0"] = bool(values[1] > 0 or values[2] < 0)
            published[f"{label}_bootstrap"] = _bootstrap_label(
                log, n_variants, int(n_boot.min()), resamples, "capped" if capped else "full", rel(PAIRED_CSV[log]))
            b.provenance.append({"table": "table2", "row": log, "column": f"{label}_bootstrap",
                                 "value": published[f"{label}_bootstrap"], "source_file": rel(INTERVAL_CSV[log]),
                                 "source_key": f"dataset={log}: bootstrap_sample_size; {rel(PAIRED_CSV[log])}: "
                                               "difference_resamples",
                                 "source_sha256": b._hash(INTERVAL_CSV[log]), "note": ""})

            t_full = "table2_full_size"
            if capped:
                if full is None:
                    raise SystemExit(f"{rel(FULL_SIZE_CSV)} is missing; run scripts/run_full_size_bootstrap.py")
                f = full[(full["log"] == log) & (full["contrast"] == f"table2_{label}")]
                if len(f) != 1:
                    raise SystemExit(f"{log} {label}: {len(f)} rows in {rel(FULL_SIZE_CSV)}")
                f = f.iloc[0]
                fkey = f"contrast=table2_{label}, log={log}"
                # same point estimate and the same capped band as the published table
                fp = b.cell(t_full, log, f"{label}_point", float(f["point_estimate"]), FULL_SIZE_CSV,
                            f"{fkey}: point_estimate", values[0])
                for side, value in (("lower", values[1]), ("upper", values[2])):
                    b.cell(t_full, log, f"{label}_capped_ci95_{side}", float(f[f"capped_ci95_{side}"]),
                           FULL_SIZE_CSV, f"{fkey}: capped_ci95_{side} (cross-check)", value)
                lo = b.cell(t_full, log, f"{label}_ci95_lower", float(f["full_ci95_lower"]), FULL_SIZE_CSV,
                            f"{fkey}: full_ci95_lower", note="new full-size interval; no paper value yet")
                hi = b.cell(t_full, log, f"{label}_ci95_upper", float(f["full_ci95_upper"]), FULL_SIZE_CSV,
                            f"{fkey}: full_ci95_upper", note="new full-size interval; no paper value yet")
                full_size[f"{label}_point"], full_size[f"{label}_ci95_lower"], full_size[f"{label}_ci95_upper"] = fp, lo, hi
                full_size[f"{label}_excludes_0"] = bool(lo > 0 or hi < 0)
                full_size[f"{label}_bootstrap"] = _bootstrap_label(
                    log, int(f["n_variants"]), int(f["n_variants"]), int(f["replicates"]), "full",
                    f"{rel(FULL_SIZE_CSV)} (seed {int(f['seed'])})")
            else:
                for part, value in zip(("point", "ci95_lower", "ci95_upper"), values):
                    full_size[f"{label}_{part}"] = value
                full_size[f"{label}_excludes_0"] = published[f"{label}_excludes_0"]
                full_size[f"{label}_bootstrap"] = published[f"{label}_bootstrap"] + " (never capped: n <= 2,500)"
            full_size[f"{label}_relation_to_zero_changes"] = (
                full_size[f"{label}_excludes_0"] != published[f"{label}_excludes_0"])
        published_rows.append(published)
        full_rows.append(full_size)
    return pd.DataFrame(published_rows), pd.DataFrame(full_rows)


# --------------------------------------------------------------------------
# Table 3
# --------------------------------------------------------------------------


def wait_for(path: Path, minutes: float, poll_minutes: float = 10.0) -> bool:
    deadline = time.time() + minutes * 60
    while not path.exists() and time.time() < deadline:
        print(f"waiting for {rel(path)} (poll every {poll_minutes:g} min)", flush=True)
        time.sleep(min(poll_minutes * 60, max(0.0, deadline - time.time())))
    return path.exists()


def table3(b: Builder) -> pd.DataFrame | None:
    if not REPLAY_TABLE.exists():
        return None
    replay = pd.read_csv(REPLAY_TABLE)
    replay = replay[replay["split"].eq("frozen")].set_index("log")
    splits = pd.read_csv(REPLAY_SPLITS)
    splits = splits[splits["is_frozen_seed"].astype(str).eq("True")]
    colmap = pd.read_csv(REPLAY_COLUMNS).set_index(["log", "method_key"]) if REPLAY_COLUMNS.exists() else None
    rows = []
    for log, exp in EXPECTED_TABLE3.items():
        t = "table3"
        if log not in replay.index:
            b.checks.append({"table": t, "row": log, "column": "row", "value": "missing", "expected": "present",
                             "abs_diff": None, "tolerance": "exact", "passed": False})
            continue
        r = replay.loc[log]
        test_variants = splits.loc[splits["log"].eq(log), "test_variants"].astype(int).unique()
        if len(test_variants) != 1:
            raise SystemExit(f"table3 {log}: test_variants not unique on the frozen split: {test_variants}")
        row = {"log": log, "log_name": LOG_NAMES[log]}
        row["hash_seed"] = b.cell(t, log, "hash_seed", int(r["hash_seed"]), REPLAY_TABLE,
                                  f"log={log}, split=frozen: hash_seed")
        row["p7_k"] = b.cell(t, log, "p7_k", int(r["p7_k"]), REPLAY_TABLE, f"log={log}, split=frozen: p7_k",
                             int(EXPECTED_TABLE2[log][0]), exact=True)
        row["test_variants"] = b.cell(t, log, "test_variants", int(test_variants[0]), REPLAY_SPLITS,
                                      f"log={log}, is_frozen_seed=True: test_variants", exp[0], exact=True)
        for method, e in zip(TABLE3_METHODS, exp[1:]):
            column = TABLE3_COLUMNS[method]
            status = str(r[f"{method}_status"])
            note = ""
            if colmap is not None and (log, method) in colmap.index:
                c = colmap.loc[(log, method)]
                note = f"method_id={c['method_id']}; partition={c['partition_source']}"
            if status == "missing":
                b.pending_cell(t, log, column, status, e if e is not None else "ok (no paper value)",
                               REPLAY_TABLE, f"log={log}, split=frozen: {method}_status", note=note)
                row[column] = None
            elif e == "infeasible":
                b.cell(t, log, f"{column}_status", status.split(":")[0], REPLAY_TABLE,
                       f"log={log}, split=frozen: {method}_status", "infeasible", exact=True, note=status)
                row[column] = None
            elif status != "ok":
                b.checks.append({"table": t, "row": log, "column": column, "value": status,
                                 "expected": "ok" if e is not None else "ok (no paper value)",
                                 "abs_diff": None, "tolerance": "exact", "passed": e is None})
                b.provenance.append({"table": t, "row": log, "column": f"{column}_status", "value": status,
                                     "source_file": rel(REPLAY_TABLE), "source_key": f"{method}_status",
                                     "source_sha256": b._hash(REPLAY_TABLE), "note": note})
                row[column] = None
            else:
                row[column] = b.cell(t, log, column, float(r[f"{method}_f1"]), REPLAY_TABLE,
                                     f"log={log}, split=frozen: {method}_f1", e, note=note)
            row[f"{column}_status"] = status
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def build(output_dir: Path | None, wait_minutes: float = 0.0) -> dict:
    """Build all tables. With output_dir=None nothing is written (used by the tests)."""
    b = Builder()
    t1 = table1(b)
    t2, t2_full = table2(b)
    if wait_minutes > 0:
        wait_for(REPLAY_TABLE, wait_minutes)
    t3 = table3(b)
    if t3 is None:
        t3_status = f"pending: {rel(REPLAY_TABLE)} not found"
    elif b.pending:
        t3_status = f"partial: {len(b.pending)} cell(s) pending (replay run not finished)"
    else:
        t3_status = "built"
    checks = pd.DataFrame(b.checks)
    checks["pending"] = checks["pending"].eq(True) if "pending" in checks else False
    result = {"table1": t1, "table2": t2, "table2_full_size": t2_full, "table3": t3,
              "provenance": pd.DataFrame(b.provenance), "checks": checks, "errors": b.errors,
              "pending": b.pending, "table3_status": t3_status}
    if output_dir is None:
        return result
    output_dir.mkdir(parents=True, exist_ok=True)
    t1.to_csv(output_dir / "table1_logs.csv", index=False)
    t2.to_csv(output_dir / "table2_matched_k_silhouette.csv", index=False)
    t2_full.to_csv(output_dir / "table2_matched_k_silhouette_full_size.csv", index=False)
    if t3 is not None:
        t3.to_csv(output_dir / "table3_matched_k_replay.csv", index=False)
    elif (output_dir / "table3_matched_k_replay.csv").exists():
        (output_dir / "table3_matched_k_replay.csv").unlink()
    result["provenance"].to_csv(output_dir / "provenance.csv", index=False)
    result["checks"].to_csv(output_dir / "value_checks.csv", index=False)
    outputs = sorted(p for p in output_dir.glob("*.csv"))
    manifest = {
        "schema_version": 2,
        "created_by": "scripts/build_paper_tables_v2.py",
        "paper": "GenAI4PM-version/Paper/main_draft.tex",
        "tolerance": TOLERANCE,
        "verified_against_paper": not b.errors,
        "errors": b.errors,
        "checks": {"total": len(b.checks), "passed": int(sum(c["passed"] for c in b.checks)),
                   "pending": len(b.pending), "failed": len(b.errors)},
        "pending": b.pending,
        "table3_status": result["table3_status"],
        "code": {rel(Path(__file__)): sha256(Path(__file__))},
        "inputs": dict(sorted(b.hashes.items())),
        "outputs": {p.name: sha256(p) for p in outputs}
                   | ({"README.md": sha256(output_dir / "README.md")} if (output_dir / "README.md").exists() else {}),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=EVAL / "paper_tables_v2")
    parser.add_argument("--wait-for-replay-minutes", type=float, default=0.0,
                        help="poll every 10 minutes for the matched-K replay table up to this long")
    parser.add_argument("--require-complete", action="store_true",
                        help="also fail when replay cells are still pending (use for the submission build)")
    args = parser.parse_args()
    result = build(args.output_dir, args.wait_for_replay_minutes)
    checks = result["checks"]
    print(f"table3: {result['table3_status']}")
    print(f"{int(checks['passed'].sum())}/{len(checks)} value checks passed, {len(result['pending'])} pending, "
          f"{len(result['errors'])} failed; {len(result['provenance'])} provenance rows")
    for p in result["pending"]:
        print(f"  pending: {p}")
    if result["errors"]:
        raise SystemExit("MISMATCH with paper values:\n" + "\n".join(result["errors"]))
    if args.require_complete and (result["pending"] or result["table3"] is None):
        raise SystemExit("Table 3 incomplete: " + result["table3_status"])


if __name__ == "__main__":
    main()
