"""Sensitivity of held-out process-model F1 (Table 3) to Python hash randomization.

Re-runs the unchanged ``run_heldout_process_evaluation.py`` in separate
interpreter processes with fixed ``PYTHONHASHSEED`` values. The variant split is
SHA-256 based and therefore identical across hash seeds; only Inductive Miner
tie-breaking can change. Frozen held-out outputs are read, never written.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN = "process_model_metrics_dynamic_heldout"
XES = {"bpi2020": "datasets/raw/bpi2020_international_declarations/InternationalDeclarations.xes.gz"}
CASE_MAP = {"bpi2020": "datasets/processed/bpi2020_international_declarations/case_variant_map.parquet"}


def _xes(dataset: str) -> Path:
    if dataset in XES:
        return ROOT / XES[dataset]
    matches = sorted((ROOT / "datasets/raw" / dataset).glob("*.xes*"))
    if not matches:
        raise FileNotFoundError(dataset)
    return matches[0]


def _case_map(dataset: str) -> Path:
    return ROOT / CASE_MAP.get(dataset, f"datasets/processed/{dataset}/case_variant_map.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["bpi2020", "rtfm"])
    parser.add_argument("--hash-seeds", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "evaluation/statistical_uncertainty_v2/discovery_hashseed_heldout",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        bench = ROOT / "evaluation/full_benchmark" / dataset
        manifest = json.loads((bench / f"{FROZEN}_manifest.json").read_text(encoding="utf-8"))
        assignments = [str(bench / f"assignments_{method}.parquet") for method in manifest["methods"]]
        frames = []
        for seed in range(args.hash_seeds):
            output = args.output_dir / f"{dataset}_seed{seed}.csv"
            if not output.exists():
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/run_heldout_process_evaluation.py"),
                        str(_xes(dataset)),
                        str(_case_map(dataset)),
                        str(output),
                        *assignments,
                    ],
                    check=True,
                    env={**os.environ, "PYTHONHASHSEED": str(seed)},
                )
            frame = pd.read_csv(output)
            frame["hash_seed"] = seed
            frames.append(frame)
            print(dataset, "hash seed", seed, "done", flush=True)
        runs = pd.concat(frames, ignore_index=True)
        keep = runs["scope"].isin(["whole_log", "test_case_weighted_aggregate"])
        aggregate = runs.loc[keep, ["method", "hash_seed", "fitness_precision_f1"]]
        aggregate.to_csv(args.output_dir / f"{dataset}_aggregate_by_seed.csv", index=False)
        frozen = pd.read_csv(bench / f"{FROZEN}.csv")
        frozen = frozen.loc[
            frozen["scope"].isin(["whole_log", "test_case_weighted_aggregate"]),
            ["method", "fitness_precision_f1"],
        ].rename(columns={"fitness_precision_f1": "frozen_f1"})
        summary = (
            aggregate.groupby("method")["fitness_precision_f1"]
            .agg(["min", "median", "max", "std", "count"])
            .reset_index()
            .merge(frozen, on="method")
            .assign(dataset=dataset)
        )
        summary.to_csv(args.output_dir / f"{dataset}_summary.csv", index=False)
        print(summary.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
