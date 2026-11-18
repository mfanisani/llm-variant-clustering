"""Build the canonical paper-table bundle for the workshop paper (paper 1).

Reads only frozen or versioned v2 outputs, writes nothing outside
``--output-dir``, and fails if any number differs from the value printed in
the paper (rounded to three decimals).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOGS = ["bpi2020", "bpi2017", "bpi2019", "sepsis", "rtfm"]
CASE_MAPS = {
    "bpi2020": "datasets/processed/bpi2020_international_declarations/case_variant_map.parquet",
    "bpi2017": "datasets/processed/bpi2017/case_variant_map.parquet",
    "bpi2019": "datasets/processed/bpi2019/case_variant_map.parquet",
    "sepsis": "datasets/processed/sepsis/case_variant_map.parquet",
    "rtfm": "datasets/processed/rtfm/case_variant_map.parquet",
}
P7 = "p7_ordered_pyramid_1_2_4_bge"
NGRAM = "ngram_tfidf_1_3"

# Values printed in the paper; the script fails if a source disagrees.
EXPECTED_TABLE1 = {
    "bpi2020": (753, 6449, "9"),
    "bpi2017": (5578, 31428, "2"),
    "bpi2019": (11928, 251381, "6"),
    "sepsis": (846, 1050, "abstains"),
    "rtfm": (231, 150370, "2"),
}
EXPECTED_TABLE2 = {
    # k, p7, raw, a2v16, ngram, ngram_k, d_raw (pt, lo, hi), d_a2v16 (pt, lo, hi)
    "bpi2020": (9, 0.571, 0.539, 0.526, 0.435, 6, (0.033, -0.028, 0.077), (0.045, -0.032, 0.097)),
    "bpi2017": (2, 0.628, 0.519, 0.652, 0.323, 4, (0.109, 0.086, 0.118), (-0.023, -0.046, 0.005)),
    "bpi2019": (6, 0.469, 0.445, 0.389, 0.319, 2, (0.024, -0.079, 0.059), (0.080, -0.239, 0.188)),
    "sepsis": (8, 0.316, 0.323, 0.329, 0.146, 8, (-0.006, -0.021, 0.011), (-0.012, -0.032, 0.003)),
    "rtfm": (2, 0.882, 0.799, 0.840, 0.771, 3, (0.083, 0.012, 0.132), (0.042, -0.020, 0.134)),
}
EXPECTED_TABLE3 = {
    # test variants, whole, raw, ngram, p7
    "bpi2020": (170, 0.451, 0.468, 0.573, 0.606),
    "bpi2017": (1162, 0.311, 0.343, 0.521, 0.431),
    "bpi2019": (2371, 0.328, 0.486, 0.291, 0.381),
    "rtfm": (46, 0.662, 0.686, 0.639, 0.716),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(a: float, b: float) -> bool:
    return abs(round(float(a), 3) - b) < 1e-9


def _source_table() -> pd.DataFrame:
    frames = []
    for folder in ("representation_sources", "representation_sources_extra"):
        path = ROOT / "evaluation" / folder / "source_paired_differences.csv"
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def build(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs: set[Path] = set()
    errors: list[str] = []

    # Table 1
    rows = []
    for log in LOGS:
        cmap_path = ROOT / CASE_MAPS[log]
        dyn_path = ROOT / "evaluation/full_benchmark" / log / "metrics_dynamic_k.csv"
        inputs.update({cmap_path, dyn_path})
        # Evaluated variants exclude the 45-variant calibration subsets of BPI 2017/2019.
        assign_path = ROOT / "evaluation/full_benchmark" / log / f"assignments_{P7}_k8.parquet"
        inputs.add(assign_path)
        evaluated = set(pd.read_parquet(assign_path)["variant_id"].astype(str))
        cmap = pd.read_parquet(cmap_path)
        cmap = cmap[cmap["variant_id"].astype(str).isin(evaluated)]
        dyn = pd.read_csv(dyn_path)
        p7 = dyn[dyn["method"] == P7].iloc[0]
        k = str(int(p7["selected_k"])) if p7["status"] == "selected" else "abstains"
        row = {"log": log, "unique_variants": cmap["variant_id"].nunique(),
               "cases": len(cmap), "automatic_p7_k": k}
        rows.append(row)
        if (row["unique_variants"], row["cases"], k) != EXPECTED_TABLE1[log]:
            errors.append(f"table1 {log}: {row} != {EXPECTED_TABLE1[log]}")
    pd.DataFrame(rows).to_csv(output_dir / "table1_logs.csv", index=False)

    # Table 2
    bench_paths = [ROOT / "evaluation/representation_sources/activity_source_benchmark.csv"]
    inputs.update(bench_paths)
    inputs.update(ROOT / "evaluation" / f / "source_paired_differences.csv"
                  for f in ("representation_sources", "representation_sources_extra"))
    bench = pd.read_csv(bench_paths[0])
    paired = _source_table()
    rows = []
    for log in LOGS:
        b = bench[bench["dataset"] == log].set_index("method")
        sil = b["matched_case_weighted_silhouette"]
        k = int(b.loc["p7_llm_semantic_bge", "matched_k"])
        ngram_file = "metrics_fixed_k8.csv" if log == "sepsis" else "metrics_dynamic_k.csv"
        ngram_path = ROOT / "evaluation/full_benchmark" / log / ngram_file
        inputs.add(ngram_path)
        ng = pd.read_csv(ngram_path).set_index("method").loc[NGRAM]
        ngram_k = 8 if log == "sepsis" else int(ng["selected_k"])
        p = paired[(paired["dataset"] == log) & (paired["first"] == "p7_llm_semantic_bge")].set_index("second")

        def diff(second: str) -> tuple[float, float, float]:
            r = p.loc[second]
            return (round(r["difference_point_estimate"], 3), round(r["difference_ci_lower"], 3),
                    round(r["difference_ci_upper"], 3))

        row = {
            "log": log, "k": k,
            "p7": round(sil["p7_llm_semantic_bge"], 3),
            "raw_label": round(sil["raw_activity_label_bge"], 3),
            "activity2vec_d16": round(sil["activity2vec_log_context_d16"], 3),
            "ngram_own_k": round(ng["case_weighted_silhouette"], 3), "ngram_k": ngram_k,
            "p7_minus_raw": diff("raw_activity_label_bge"),
            "p7_minus_a2v16": diff("activity2vec_log_context_d16"),
            "resamples": int(p["difference_resamples"].iloc[0]),
        }
        exp = EXPECTED_TABLE2[log]
        got = (row["k"], row["p7"], row["raw_label"], row["activity2vec_d16"], row["ngram_own_k"],
               row["ngram_k"], row["p7_minus_raw"], row["p7_minus_a2v16"])
        flat_got = [*got[:6], *got[6], *got[7]]
        flat_exp = [*exp[:6], *exp[6], *exp[7]]
        if not all(_close(g, e) for g, e in zip(flat_got, flat_exp)):
            errors.append(f"table2 {log}: {got} != {exp}")
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "table2_matched_k_silhouette.csv", index=False)

    # Table 3
    rows = []
    for log, exp in EXPECTED_TABLE3.items():
        path = ROOT / "evaluation/full_benchmark" / log / "process_model_metrics_dynamic_heldout.csv"
        inputs.add(path)
        h = pd.read_csv(path)
        whole = h[h["scope"] == "whole_log"].iloc[0]
        agg = h[h["scope"] == "test_case_weighted_aggregate"].set_index("method")["fitness_precision_f1"]

        def pick(prefix_options: tuple[str, ...]) -> float:
            for name in agg.index:
                if name.startswith(prefix_options):
                    return round(float(agg[name]), 3)
            raise KeyError(prefix_options)

        row = {
            "log": log, "test_variants": int(whole["test_variants"]),
            "whole_log": round(float(whole["fitness_precision_f1"]), 3),
            "raw_bge": pick(("raw_activity_label", "raw_sequence")),
            "ngram": pick((NGRAM,)), "p7": pick((P7,)),
        }
        got = (row["test_variants"], row["whole_log"], row["raw_bge"], row["ngram"], row["p7"])
        if not all(_close(g, e) for g, e in zip(got, exp)):
            errors.append(f"table3 {log}: {got} != {exp}")
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "table3_heldout_f1.csv", index=False)

    manifest = {
        "purpose": "canonical tables for the paper",
        "verified_against_paper": not errors,
        "errors": errors,
        "inputs": {str(p.relative_to(ROOT)).replace("\\", "/"): _sha256(p) for p in sorted(inputs)},
        "outputs": sorted(p.name for p in output_dir.glob("table*.csv")),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation/paper_tables_v1")
    args = parser.parse_args()
    manifest = build(args.output_dir)
    if manifest["errors"]:
        raise SystemExit("Mismatch with paper:\n" + "\n".join(manifest["errors"]))
    print(f"OK: {len(manifest['outputs'])} tables match the paper; {len(manifest['inputs'])} inputs hashed.")


if __name__ == "__main__":
    main()
