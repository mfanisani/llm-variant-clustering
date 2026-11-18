"""Sensitivity of same-log process-model F1 to Python hash randomization.

pm4py's inductive miner breaks ties through set iteration order, which depends
on PYTHONHASHSEED. The same sublog can therefore yield different Petri nets in
different interpreter processes. This script re-runs the unchanged discovery
and token-replay evaluation in separate worker processes with fixed hash seeds
and reports the spread of per-cluster and case-weighted F1.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "bpi2020": (
        "datasets/raw/bpi2020_international_declarations/InternationalDeclarations.xes.gz",
        "datasets/processed/bpi2020_international_declarations/case_variant_map.parquet",
        "process_model_metrics_dynamic.csv",
    ),
    "bpi2017": (None, "datasets/processed/bpi2017/case_variant_map.parquet", "process_model_metrics_dynamic.csv"),
    "bpi2019": (None, "datasets/processed/bpi2019/case_variant_map.parquet", "process_model_metrics_dynamic.csv"),
    "sepsis": (None, "datasets/processed/sepsis/case_variant_map.parquet", "process_model_metrics_k8_fixed_noise.csv"),
    "rtfm": (None, "datasets/processed/rtfm/case_variant_map.parquet", "process_model_metrics_dynamic.csv"),
}


def _xes_path(dataset: str) -> Path:
    configured = DATASETS[dataset][0]
    if configured:
        return ROOT / configured
    folder = ROOT / "datasets" / "raw" / dataset
    matches = sorted(folder.glob("*.xes*"))
    if not matches:
        raise FileNotFoundError(f"No XES log under {folder}")
    return matches[0]


def _assignment_files(dataset: str) -> list[tuple[str, Path]]:
    metrics = pd.read_csv(ROOT / "evaluation/full_benchmark" / dataset / DATASETS[dataset][2])
    methods = sorted(metrics.loc[metrics["scope"] == "cluster", "method"].unique())
    return [
        (method, ROOT / "evaluation/full_benchmark" / dataset / f"assignments_{method}.parquet")
        for method in methods
    ]


def _cache_events(dataset: str, cache_dir: Path) -> Path:
    import pm4py

    target = cache_dir / f"{dataset}_events.parquet"
    if target.exists():
        return target
    events = pm4py.read_xes(
        str(_xes_path(dataset)), return_legacy_log_object=False, show_progress_bar=False
    )
    keep = ["case:concept:name", "concept:name", "time:timestamp"]
    events[keep].to_parquet(target, index=False)
    return target


def _worker(dataset: str, events_path: Path, output: Path) -> None:
    from llm_variant_clustering.process_evaluation import evaluate_process_partition

    events = pd.read_parquet(events_path)
    case_map = pd.read_parquet(ROOT / DATASETS[dataset][1])
    case_map["variant_id"] = case_map["variant_id"].astype(str)
    frames = []
    for method, path in _assignment_files(dataset):
        assignments = pd.read_parquet(path)
        assignments["variant_id"] = assignments["variant_id"].astype(str)
        selected = case_map[case_map["variant_id"].isin(set(assignments["variant_id"]))]
        subset = events[
            events["case:concept:name"].astype(str).isin(set(selected["case_id"].astype(str)))
        ]
        frames.append(
            evaluate_process_partition(subset, selected, assignments, method=method)
        )
    result = pd.concat(frames, ignore_index=True)
    result["hash_seed"] = int(os.environ["PYTHONHASHSEED"])
    result.to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--hash-seeds", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "evaluation/statistical_uncertainty_v2/discovery_hashseed",
    )
    parser.add_argument("--worker", nargs=3, metavar=("DATASET", "EVENTS", "OUTPUT"))
    args = parser.parse_args()
    if args.worker:
        dataset, events_path, output = args.worker
        _worker(dataset, Path(events_path), Path(output))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "_cache"
    cache_dir.mkdir(exist_ok=True)
    for dataset in args.datasets:
        events_path = _cache_events(dataset, cache_dir)
        per_seed = []
        for seed in range(args.hash_seeds):
            output = args.output_dir / f"{dataset}_seed{seed}.csv"
            if not output.exists():
                env = {**os.environ, "PYTHONHASHSEED": str(seed)}
                subprocess.run(
                    [sys.executable, __file__, "--worker", dataset, str(events_path), str(output)],
                    check=True,
                    env=env,
                )
            per_seed.append(pd.read_csv(output))
            print(dataset, "hash seed", seed, "done", flush=True)
        clusters = pd.concat(per_seed, ignore_index=True)
        clusters["weighted"] = clusters["fitness_precision_f1"] * clusters["cases"]
        aggregate = (
            clusters.groupby(["method", "hash_seed"])
            .agg(cases=("cases", "sum"), weighted=("weighted", "sum"))
            .assign(f1=lambda frame: frame["weighted"] / frame["cases"])
            .reset_index()
        )
        frozen = pd.read_csv(ROOT / "evaluation/full_benchmark" / dataset / DATASETS[dataset][2])
        frozen = frozen.loc[frozen["scope"] == "case_weighted_aggregate", ["method", "fitness_precision_f1"]]
        summary = (
            aggregate.groupby("method")["f1"]
            .agg(["min", "median", "max", "std", "count"])
            .reset_index()
            .merge(frozen.rename(columns={"fitness_precision_f1": "frozen_f1"}), on="method")
            .assign(dataset=dataset)
        )
        summary.to_csv(args.output_dir / f"{dataset}_summary.csv", index=False)
        aggregate.to_csv(args.output_dir / f"{dataset}_aggregate_by_seed.csv", index=False)
        print(summary.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
