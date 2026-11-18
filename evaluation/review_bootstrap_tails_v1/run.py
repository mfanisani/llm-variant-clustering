"""C4: 5,000-replicate paired bootstrap of the P7 - raw case-weighted silhouette.

Re-runs the bootstrap of scripts/run_statistical_uncertainty.py (published
family: BPI 2017, BPI 2019, Sepsis, RTFM) with identical inputs and RNG seed,
parallelising replicate chunks only. Writes only inside this folder.

  python run.py compute --setting cap  --datasets bpi2017 bpi2019 sepsis rtfm
  python run.py compute --setting full --datasets bpi2017
  python run.py pilot   --dataset bpi2019 --replicates 6
  python run.py verify-serial --datasets rtfm sepsis
  python run.py summarize
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from llm_variant_clustering.evaluation import frequency_weighted_silhouette
from llm_variant_clustering.uncertainty import (
    _percentile_interval,
    _resample_indices,
    _silhouette_on_subset,
    paired_silhouette_difference,
)
from run_statistical_uncertainty import DATASETS, _labels, _load_variants

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SCRATCH = HERE / "_scratch"
FAMILY = ["bpi2017", "bpi2019", "sepsis", "rtfm"]
P7 = "p7_ordered_pyramid_1_2_4_bge"
SEED = 20240601
RESAMPLES = 5000
CAP = 2500
PUBLISHED_RESAMPLES = 400
CONFIDENCE = 0.95
BONFERRONI = 1.0 - (1.0 - CONFIDENCE) / len(FAMILY)

_STATE: dict = {}


def load_inputs(dataset: str) -> dict:
    p7_file, raw_file, raw_method, suffix = DATASETS[dataset]
    benchmark_dir = ROOT / "evaluation/full_benchmark" / dataset
    cwd = os.getcwd()
    os.chdir(ROOT)  # _load_variants uses a repository-relative path
    try:
        variants = _load_variants(dataset, Path("evaluation/full_benchmark") / dataset)
    finally:
        os.chdir(cwd)
    order = variants["variant_id"].astype(str).tolist()
    weights = variants["frequency"].to_numpy(dtype=np.float64)
    p7_labels = _labels(benchmark_dir, P7, suffix, order)
    raw_labels = _labels(benchmark_dir, raw_method, suffix, order)
    p7_distances = pairwise_distances(np.load(benchmark_dir / p7_file), metric="cosine")
    raw_distances = pairwise_distances(np.load(benchmark_dir / raw_file), metric="cosine")
    return {
        "raw_method": raw_method,
        "suffix": suffix,
        "order": order,
        "weights": weights,
        "p7_labels": p7_labels,
        "raw_labels": raw_labels,
        "p7_distances": p7_distances,
        "raw_distances": raw_distances,
    }


def _init(paths: dict) -> None:
    _STATE["p7_distances"] = np.load(paths["p7_distances"], mmap_mode="r")
    _STATE["raw_distances"] = np.load(paths["raw_distances"], mmap_mode="r")
    _STATE["p7_labels"] = np.load(paths["p7_labels"], allow_pickle=True)
    _STATE["raw_labels"] = np.load(paths["raw_labels"], allow_pickle=True)
    _STATE["weights"] = np.load(paths["weights"])


def _chunk(task: tuple[int, np.ndarray]) -> list[tuple[int, float | None]]:
    start, draws = task
    out = []
    for offset, indices in enumerate(draws):
        first = _silhouette_on_subset(_STATE["p7_distances"], _STATE["p7_labels"], _STATE["weights"], indices)
        second = _silhouette_on_subset(_STATE["raw_distances"], _STATE["raw_labels"], _STATE["weights"], indices)
        value = None if first is None or second is None else first - second
        out.append((start + offset, value))
    return out


def _stage(dataset: str, inputs: dict) -> dict:
    SCRATCH.mkdir(exist_ok=True)
    paths = {}
    for key in ["p7_distances", "raw_distances", "p7_labels", "raw_labels", "weights"]:
        path = SCRATCH / f"{dataset}_{key}.npy"
        np.save(path, np.asarray(inputs[key]))
        paths[key] = str(path)
    return paths


def run_bootstrap(dataset: str, inputs: dict, sample_size: int | None, workers: int, resamples: int) -> tuple[np.ndarray, np.ndarray, int]:
    n = len(inputs["order"])
    draws = _resample_indices(n, resamples, SEED, sample_size)
    paths = _stage(dataset, inputs)
    chunk = max(1, min(50, resamples // (workers * 4) or 1))
    tasks = [(s, draws[s : s + chunk]) for s in range(0, resamples, chunk)]
    results: list[tuple[int, float | None]] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(paths,)) as pool:
        for done, part in enumerate(pool.map(_chunk, tasks), start=1):
            results.extend(part)
            if done % max(1, len(tasks) // 10) == 0:
                print(f"  {dataset}: {done}/{len(tasks)} chunks", flush=True)
    results.sort(key=lambda item: item[0])
    values = [v for _, v in results if v is not None]
    valid_index = [i for i, v in results if v is not None]
    skipped = sum(v is None for _, v in results)
    return np.asarray(values, dtype=np.float64), np.asarray(valid_index), skipped


def interval_record(point: float, samples: np.ndarray, prefix: str) -> dict:
    nominal = _percentile_interval(point, samples, CONFIDENCE)
    adjusted = _percentile_interval(point, samples, BONFERRONI)
    return {
        f"{prefix}valid_replicates": int(samples.size),
        f"{prefix}ci95_lower": nominal.lower,
        f"{prefix}ci95_upper": nominal.upper,
        f"{prefix}ci95_excludes_zero": bool(nominal.lower > 0 or nominal.upper < 0),
        f"{prefix}bonf9875_lower": adjusted.lower,
        f"{prefix}bonf9875_upper": adjusted.upper,
        f"{prefix}bonf9875_excludes_zero": bool(adjusted.lower > 0 or adjusted.upper < 0),
        f"{prefix}share_gt_0": float(np.mean(samples > 0)),
        f"{prefix}n_below_bonf_lower": int(np.sum(samples < adjusted.lower)),
        f"{prefix}n_above_bonf_upper": int(np.sum(samples > adjusted.upper)),
        f"{prefix}n_le_0": int(np.sum(samples <= 0)),
    }


def compute(args: argparse.Namespace) -> None:
    for dataset in args.datasets:
        started = time.time()
        inputs = load_inputs(dataset)
        n = len(inputs["order"])
        if args.setting == "cap":
            sample_size = CAP if n > CAP else None
        else:
            sample_size = None
        point = frequency_weighted_silhouette(
            inputs["p7_distances"], inputs["p7_labels"], inputs["weights"]
        ) - frequency_weighted_silhouette(inputs["raw_distances"], inputs["raw_labels"], inputs["weights"])
        print(f"{dataset} setting={args.setting} n={n} m={sample_size or n}", flush=True)
        samples, valid_index, skipped = run_bootstrap(dataset, inputs, sample_size, args.workers, RESAMPLES)
        tag = f"{dataset}_{args.setting}"
        np.save(HERE / f"replicates_{tag}.npy", samples)
        record = {
            "dataset": dataset,
            "setting": args.setting,
            "first": P7,
            "second": inputs["raw_method"],
            "partition": inputs["suffix"],
            "variant_count": n,
            "bootstrap_sample_size": sample_size or n,
            "rng_seed": SEED,
            "resamples_requested": RESAMPLES,
            "skipped_replicates": skipped,
            "point_estimate": float(point),
            **interval_record(point, samples, ""),
        }
        first400 = samples[valid_index < PUBLISHED_RESAMPLES]
        record["first400_draws_skipped"] = int(PUBLISHED_RESAMPLES - first400.size)
        record.update(interval_record(point, first400, "first400_"))
        record["runtime_seconds"] = round(time.time() - started, 1)
        (HERE / f"result_{tag}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=2), flush=True)
        for path in SCRATCH.glob(f"{dataset}_*.npy"):
            path.unlink()


def pilot(args: argparse.Namespace) -> None:
    inputs = load_inputs(args.dataset)
    n = len(inputs["order"])
    draws = _resample_indices(n, args.replicates, SEED, None)
    started = time.time()
    for indices in draws:
        _silhouette_on_subset(inputs["p7_distances"], inputs["p7_labels"], inputs["weights"], indices)
        _silhouette_on_subset(inputs["raw_distances"], inputs["raw_labels"], inputs["weights"], indices)
    per = (time.time() - started) / args.replicates
    record = {
        "dataset": args.dataset,
        "n": n,
        "pilot_replicates": args.replicates,
        "seconds_per_replicate_single_process": per,
        "projected_minutes_5000_at_workers": {w: per * RESAMPLES / w / 60 for w in (1, 4, 6, 8, 12)},
    }
    (HERE / f"pilot_{args.dataset}_full.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


def verify_serial(args: argparse.Namespace) -> None:
    out = {}
    for dataset in args.datasets:
        inputs = load_inputs(dataset)
        serial = paired_silhouette_difference(
            inputs["p7_distances"], inputs["p7_labels"], inputs["raw_distances"], inputs["raw_labels"],
            inputs["weights"], resamples=RESAMPLES, sample_size=None,
        )
        parallel = np.load(HERE / f"replicates_{dataset}_cap.npy")
        out[dataset] = {
            "serial_size": int(serial.samples.size),
            "parallel_size": int(parallel.size),
            "identical": bool(serial.samples.size == parallel.size and np.array_equal(serial.samples, parallel)),
            "serial_ci95": [serial.interval.lower, serial.interval.upper],
        }
    (HERE / "serial_equivalence_check.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


def summarize(_: argparse.Namespace) -> None:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(HERE.glob("result_*.json"))]
    frame = pd.DataFrame(rows)
    published = pd.read_csv(ROOT / "evaluation/statistical_uncertainty_v2/paired_silhouette_differences.csv")
    published = published.set_index("dataset")
    for column, source in [
        ("published_ci95_lower", "difference_ci_lower"),
        ("published_ci95_upper", "difference_ci_upper"),
        ("published_point_estimate", "difference_point_estimate"),
    ]:
        frame[column] = frame["dataset"].map(published[source])
    frame.to_csv(HERE / "paired_bootstrap_5000.csv", index=False)
    pd.set_option("display.width", 250)
    print(frame.round(4).T.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--setting", choices=["cap", "full"], required=True)
    c.add_argument("--datasets", nargs="+", default=FAMILY)
    c.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("pilot")
    p.add_argument("--dataset", default="bpi2019")
    p.add_argument("--replicates", type=int, default=6)
    v = sub.add_parser("verify-serial")
    v.add_argument("--datasets", nargs="+", default=["rtfm", "sepsis"])
    sub.add_parser("summarize")
    args = parser.parse_args()
    {"compute": compute, "pilot": pilot, "verify-serial": verify_serial, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
