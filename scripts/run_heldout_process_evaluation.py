from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pm4py

from llm_variant_clustering.process_evaluation import (
    evaluate_holdout_model,
    evaluate_process_partition_holdout,
    select_variant_holdout_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Discover process models on training variants and evaluate them on "
            "held-out variants using one shared deterministic split."
        )
    )
    parser.add_argument("xes_log", type=Path)
    parser.add_argument("case_variant_map", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("assignments", type=Path, nargs="+")
    parser.add_argument("--noise-threshold", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--case-fraction-tolerance", type=float, default=0.05)
    parser.add_argument("--max-split-seeds", type=int, default=10_000)
    args = parser.parse_args()

    case_map = pd.read_parquet(args.case_variant_map)
    assignment_frames = {
        path.stem.removeprefix("assignments_"): pd.read_parquet(path)
        for path in args.assignments
    }
    variant_sets = [
        frozenset(frame["variant_id"].astype(str))
        for frame in assignment_frames.values()
    ]
    if any(variant_ids != variant_sets[0] for variant_ids in variant_sets[1:]):
        raise ValueError("All compared assignments must cover the same variants")
    selected_variants = variant_sets[0]
    if not selected_variants:
        raise ValueError("Assignments must cover at least one variant")

    case_map = case_map[
        case_map["variant_id"].astype(str).isin(selected_variants)
    ].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.with_name(f"{args.output.stem}_manifest.json")
    try:
        variant_split, split_seed = select_variant_holdout_split(
            case_map,
            assignment_frames,
            test_fraction=args.test_fraction,
            max_seed=args.max_split_seeds,
            case_fraction_tolerance=args.case_fraction_tolerance,
        )
    except ValueError as error:
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "infeasible",
                    "split_unit": "unique_variant",
                    "reason": str(error),
                    "requested_test_fraction": args.test_fraction,
                    "case_fraction_tolerance": args.case_fraction_tolerance,
                    "methods": sorted(assignment_frames),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Held-out evaluation abstained: {error}")
        print(manifest_path)
        return

    events = pm4py.read_xes(
        str(args.xes_log),
        return_legacy_log_object=False,
        show_progress_bar=False,
    )
    selected_cases = set(case_map["case_id"].astype(str))
    events = events[
        events["case:concept:name"].astype(str).isin(selected_cases)
    ].copy()
    if events["case:concept:name"].astype(str).nunique() != len(selected_cases):
        raise ValueError("Evaluation events do not cover every selected case")
    split_by_variant = dict(
        zip(variant_split["variant_id"], variant_split["split"], strict=True)
    )
    case_splits = case_map["variant_id"].astype(str).map(split_by_variant)
    split_by_case = dict(
        zip(case_map["case_id"].astype(str), case_splits, strict=True)
    )
    event_splits = events["case:concept:name"].astype(str).map(split_by_case)
    train_events = events[event_splits.eq("train")].copy()
    test_events = events[event_splits.eq("test")].copy()

    rows = [
        {
            "method": "whole_log",
            "scope": "whole_log",
            "cluster_id": "whole_log",
            "noise_threshold": args.noise_threshold,
            "train_variants": int(variant_split["split"].eq("train").sum()),
            "test_variants": int(variant_split["split"].eq("test").sum()),
            **evaluate_holdout_model(
                train_events,
                test_events,
                noise_threshold=args.noise_threshold,
            ),
        }
    ]
    for method, assignments in assignment_frames.items():
        partition = evaluate_process_partition_holdout(
            events,
            case_map,
            assignments,
            variant_split,
            method=method,
            noise_threshold=args.noise_threshold,
        )
        rows.extend(partition.to_dict(orient="records"))

    metrics = pd.DataFrame(rows)
    aggregate_rows = []
    for method, group in metrics[metrics["scope"].eq("cluster")].groupby(
        "method", sort=True
    ):
        weights = group["test_cases"]
        aggregate_rows.append(
            {
                "method": method,
                "scope": "test_case_weighted_aggregate",
                "cluster_id": "aggregate",
                "noise_threshold": args.noise_threshold,
                "train_variants": int(group["train_variants"].sum()),
                "test_variants": int(group["test_variants"].sum()),
                "train_cases": int(group["train_cases"].sum()),
                "train_events": int(group["train_events"].sum()),
                "test_cases": int(group["test_cases"].sum()),
                "test_events": int(group["test_events"].sum()),
                **{
                    metric: float((group[metric] * weights).sum() / weights.sum())
                    for metric in [
                        "fitness",
                        "precision",
                        "fitness_precision_f1",
                        "simplicity_arc_degree",
                        "places",
                        "transitions",
                        "silent_transitions",
                        "arcs",
                    ]
                },
            }
        )
    metrics = pd.concat(
        [metrics, pd.DataFrame(aggregate_rows)],
        ignore_index=True,
    )

    metrics.to_csv(args.output, index=False)
    split_path = args.output.with_name(f"{args.output.stem}_variant_split.parquet")
    variant_split.to_parquet(split_path, index=False)
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "split_unit": "unique_variant",
        "split_algorithm": "sha256_threshold_with_cross_method_cluster_coverage",
        "split_seed": split_seed,
        "requested_test_fraction": args.test_fraction,
        "observed_test_variant_fraction": float(
            variant_split["split"].eq("test").mean()
        ),
        "observed_test_case_fraction": float(
            variant_split.loc[
                variant_split["split"].eq("test"), "cases"
            ].sum()
            / variant_split["cases"].sum()
        ),
        "noise_threshold": args.noise_threshold,
        "methods": sorted(assignment_frames),
        "variant_split": str(split_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
