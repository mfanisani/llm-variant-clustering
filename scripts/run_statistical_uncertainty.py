"""Bootstrap confidence intervals for the frozen benchmark comparisons.

The primary study reports point estimates only. Several cross-method gaps are
smaller than 0.02 silhouette, which cannot support a directional claim without
an uncertainty estimate. This script resamples variants from the frozen
embedding artifacts and reports percentile intervals for each method plus a
paired interval for the P7 minus baseline difference on identical variants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from llm_variant_clustering.uncertainty import (
    bootstrap_cluster_metric,
    bootstrap_silhouette,
    cluster_scope_mask,
    paired_silhouette_difference,
)

CONFIDENCE = 0.95

# dataset -> (p7 embedding file, baseline embedding file, baseline method name,
#             frozen partition suffix)
DATASETS = {
    "bpi2020": (
        "ordered_pyramid_embeddings.npy",
        None,
        "raw_sequence_bge",
        "dynamic",
    ),
    "bpi2017": (
        "p7_ordered_embeddings.npy",
        "raw_ordered_embeddings.npy",
        "raw_activity_label_pyramid_bge",
        "dynamic",
    ),
    "bpi2019": (
        "p7_ordered_embeddings.npy",
        "raw_ordered_embeddings.npy",
        "raw_activity_label_pyramid_bge",
        "dynamic",
    ),
    "sepsis": (
        "p7_ordered_embeddings.npy",
        "raw_ordered_embeddings.npy",
        "raw_activity_label_pyramid_bge",
        "k8",
    ),
    "rtfm": (
        "p7_ordered_embeddings.npy",
        "raw_ordered_embeddings.npy",
        "raw_activity_label_pyramid_bge",
        "dynamic",
    ),
}

DATASET_FOLDERS = {
    "bpi2020": "bpi2020_international_declarations",
    "bpi2017": "bpi2017",
    "bpi2019": "bpi2019",
    "sepsis": "sepsis",
    "rtfm": "rtfm",
}


def _load_variants(dataset: str, benchmark_dir: Path) -> pd.DataFrame:
    variants = pd.read_parquet(
        Path("datasets/processed") / DATASET_FOLDERS[dataset] / "variants.parquet"
    )
    index_path = benchmark_dir / "embedding_index.parquet"
    if index_path.exists():
        order = pd.read_parquet(index_path)["variant_id"].astype(str)
        variants = (
            variants.assign(variant_id=variants["variant_id"].astype(str))
            .set_index("variant_id")
            .loc[order]
            .reset_index()
        )
    return variants.reset_index(drop=True)


def _labels(benchmark_dir: Path, method: str, suffix: str, order: list[str]):
    path = benchmark_dir / f"assignments_{method}_{suffix}.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    lookup = frame.set_index(frame["variant_id"].astype(str))["cluster_label"]
    return lookup.loc[order].to_numpy()


def _process_f1_intervals(benchmark_dir: Path, resamples: int) -> list[dict]:
    rows = []
    for path in sorted(benchmark_dir.glob("process_model_metrics_*.csv")):
        if "heldout" in path.name or "manifest" in path.name:
            continue
        frame = pd.read_csv(path)
        if "scope" not in frame.columns:
            raise ValueError(f"{path} has no scope column; cannot isolate clusters")
        clusters = frame.loc[cluster_scope_mask(frame["scope"])]
        if clusters.empty or "fitness_precision_f1" not in clusters.columns:
            continue
        weight_column = "cases" if "cases" in clusters.columns else None
        aggregates = frame.loc[frame["scope"] == "case_weighted_aggregate"]
        for method, group in clusters.groupby("method"):
            if len(group) < 2:
                continue
            weights = (
                group[weight_column].to_numpy(dtype=float)
                if weight_column
                else None
            )
            aggregate = aggregates.loc[aggregates["method"] == method]
            aggregate_f1 = None
            if weight_column and len(aggregate) == 1:
                if int(aggregate[weight_column].iloc[0]) != int(weights.sum()):
                    raise ValueError(
                        f"{path.name} {method}: cluster cases do not sum to the "
                        "case-weighted aggregate row"
                    )
                aggregate_f1 = float(aggregate["fitness_precision_f1"].iloc[0])
            interval = bootstrap_cluster_metric(
                group["fitness_precision_f1"].to_numpy(dtype=float),
                weights,
                resamples=resamples,
            )
            rows.append(
                {
                    "source_file": path.name,
                    "method": method,
                    "cluster_count": len(group),
                    "weighting": "cases" if weight_column else "uniform",
                    "reported_aggregate_f1": aggregate_f1,
                    **interval.as_dict("f1_"),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resamples", type=int, default=400)
    parser.add_argument(
        "--max-bootstrap-variants",
        type=int,
        default=2500,
        help="m for the m-out-of-n bootstrap; 0 disables the cap",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/statistical_uncertainty_v2"),
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument(
        "--skip-process-f1",
        action="store_true",
        help="Only compute silhouette intervals (for cap-sensitivity runs)",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cap = args.max_bootstrap_variants if args.max_bootstrap_variants > 0 else None

    selected = {
        name: spec
        for name, spec in DATASETS.items()
        if args.datasets is None or name in set(args.datasets)
    }
    if not selected:
        raise SystemExit("No matching datasets")

    silhouette_rows = []
    paired_rows = []
    paired_results = []
    f1_rows = []

    for dataset, (p7_file, raw_file, raw_method, suffix) in selected.items():
        benchmark_dir = Path("evaluation/full_benchmark") / dataset
        variants = _load_variants(dataset, benchmark_dir)
        order = variants["variant_id"].astype(str).tolist()
        weights = variants["frequency"].to_numpy(dtype=np.float64)
        sample_size = cap if cap is not None and len(order) > cap else None

        p7_embeddings = np.load(benchmark_dir / p7_file)
        p7_distances = pairwise_distances(p7_embeddings, metric="cosine")
        p7_labels = _labels(
            benchmark_dir,
            "p7_ordered_pyramid_1_2_4_bge",
            suffix,
            order,
        )

        method_labels = {"p7_ordered_pyramid_1_2_4_bge": (p7_distances, p7_labels)}
        if raw_file is not None:
            raw_embeddings = np.load(benchmark_dir / raw_file)
            raw_distances = pairwise_distances(raw_embeddings, metric="cosine")
            raw_labels = _labels(benchmark_dir, raw_method, suffix, order)
            if raw_labels is not None:
                method_labels[raw_method] = (raw_distances, raw_labels)

        for method, (distances, labels) in method_labels.items():
            if labels is None:
                continue
            interval = bootstrap_silhouette(
                distances,
                labels,
                weights,
                resamples=args.resamples,
                sample_size=sample_size,
            )
            silhouette_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "partition": suffix,
                    "variant_count": len(order),
                    "bootstrap_sample_size": sample_size or len(order),
                    "cluster_count": int(np.unique(labels).size),
                    **interval.as_dict("silhouette_"),
                }
            )
            print(f"{dataset} {method} silhouette CI done", flush=True)

        if raw_method in method_labels and p7_labels is not None:
            raw_distances, raw_labels = method_labels[raw_method]
            difference = paired_silhouette_difference(
                p7_distances,
                p7_labels,
                raw_distances,
                raw_labels,
                weights,
                resamples=args.resamples,
                sample_size=sample_size,
            )
            paired_rows.append(
                {
                    "dataset": dataset,
                    "first": "p7_ordered_pyramid_1_2_4_bge",
                    "second": raw_method,
                    "partition": suffix,
                    "bootstrap_sample_size": sample_size or len(order),
                    **difference.as_dict("difference_"),
                }
            )
            paired_results.append(difference)
            print(f"{dataset} paired difference done", flush=True)

        if not args.skip_process_f1:
            for row in _process_f1_intervals(benchmark_dir, args.resamples):
                row["dataset"] = dataset
                f1_rows.append(row)

    family_size = len(paired_results)
    if family_size:
        adjusted_confidence = 1.0 - (1.0 - CONFIDENCE) / family_size
        for row, difference in zip(paired_rows, paired_results):
            adjusted = difference.adjusted_interval(adjusted_confidence)
            row["bonferroni_family_size"] = family_size
            row["bonferroni_confidence"] = adjusted_confidence
            row["bonferroni_ci_lower"] = adjusted.lower
            row["bonferroni_ci_upper"] = adjusted.upper
            row["bonferroni_excludes_zero"] = bool(
                adjusted.lower > 0 or adjusted.upper < 0
            )

    pd.DataFrame(silhouette_rows).to_csv(
        args.output_dir / "silhouette_confidence_intervals.csv", index=False
    )
    pd.DataFrame(paired_rows).to_csv(
        args.output_dir / "paired_silhouette_differences.csv", index=False
    )
    if not args.skip_process_f1:
        pd.DataFrame(f1_rows).to_csv(
            args.output_dir / "process_f1_confidence_intervals.csv", index=False
        )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "ancillary_after_primary_artifact_lock",
                "supersedes": "evaluation/statistical_uncertainty (schema 1)",
                "purpose": (
                    "Attach percentile confidence intervals to the frozen "
                    "point estimates without altering any frozen partition."
                ),
                "datasets": list(selected),
                "resamples": args.resamples,
                "confidence": CONFIDENCE,
                "bootstrap_unit_silhouette": "variant, case frequency as weight",
                "bootstrap_unit_process_f1": (
                    "discovered cluster (scope == 'cluster' rows only), "
                    "case count as weight"
                ),
                "max_bootstrap_variants": cap,
                "paired_family_wise_adjustment": (
                    f"Bonferroni over {family_size} paired silhouette "
                    "comparisons, read from the same bootstrap samples"
                ),
                "corrections_from_schema_1": [
                    "Process-F1 intervals previously resampled the "
                    "case_weighted_aggregate summary rows as if they were "
                    "clusters; only scope == 'cluster' rows are now used.",
                    "The m-out-of-n interval is no longer described as "
                    "conservative; see cap-sensitivity outputs.",
                ],
                "note": (
                    "Logs above the variant cap use an unrescaled m-out-of-n "
                    "bootstrap. Its width reflects m, not n, and it is not "
                    "guaranteed to be conservative."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output_dir / "silhouette_confidence_intervals.csv")


if __name__ == "__main__":
    main()
