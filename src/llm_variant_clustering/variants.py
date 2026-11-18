from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import pm4py

LifecyclePolicy = Literal["all", "complete", "auto"]


@dataclass(frozen=True)
class ExtractionSummary:
    source: str
    lifecycle_policy_requested: str
    lifecycle_policy_applied: str
    events_before_filter: int
    events_after_filter: int
    cases: int
    activities: int
    variants: int
    timestamp_tie_events: int
    min_variant_length: int
    max_variant_length: int


def variant_id(sequence: tuple[str, ...]) -> str:
    canonical = json.dumps(
        sequence, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return f"v_{hashlib.sha256(canonical).hexdigest()[:20]}"


def _apply_lifecycle_policy(
    events: pd.DataFrame,
    policy: LifecyclePolicy,
    lifecycle_key: str,
) -> tuple[pd.DataFrame, str]:
    if policy == "all":
        return events, "all"

    if lifecycle_key not in events.columns:
        if policy == "complete":
            raise ValueError(
                f"Lifecycle policy 'complete' requires column {lifecycle_key!r}"
            )
        return events, "all_column_absent"

    transitions = events[lifecycle_key].astype("string").str.casefold()
    complete_mask = transitions.eq("complete")
    if not complete_mask.any():
        if policy == "complete":
            raise ValueError(
                f"No 'complete' events found in column {lifecycle_key!r}"
            )
        return events, "all_no_complete_value"

    return events.loc[complete_mask].copy(), "complete"


def extract_variants(
    events: pd.DataFrame,
    *,
    source: str,
    lifecycle_policy: LifecyclePolicy = "auto",
    case_key: str = "case:concept:name",
    activity_key: str = "concept:name",
    timestamp_key: str = "time:timestamp",
    lifecycle_key: str = "lifecycle:transition",
) -> tuple[pd.DataFrame, pd.DataFrame, ExtractionSummary]:
    required = {case_key, activity_key, timestamp_key}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"Missing required event columns: {missing}")

    event_count_before = len(events)
    filtered, applied_policy = _apply_lifecycle_policy(
        events, lifecycle_policy, lifecycle_key
    )
    if filtered.empty:
        raise ValueError("Lifecycle filtering removed every event")

    null_counts = filtered[[case_key, activity_key, timestamp_key]].isna().sum()
    if null_counts.any():
        invalid = {key: int(value) for key, value in null_counts.items() if value}
        raise ValueError(f"Required event fields contain nulls: {invalid}")

    ordered = filtered.copy()
    ordered["_source_order"] = range(len(ordered))
    ordered = ordered.sort_values(
        [case_key, timestamp_key, "_source_order"], kind="mergesort"
    )

    timestamp_tie_events = int(
        ordered.duplicated([case_key, timestamp_key], keep=False).sum()
    )

    case_sequences = (
        ordered.groupby(case_key, sort=False, observed=True)[activity_key]
        .agg(lambda values: tuple(str(value) for value in values))
        .rename("sequence")
    )
    if case_sequences.empty:
        raise ValueError("No cases remained after preprocessing")

    pm4py_counts = pm4py.get_variants_as_tuples(
        ordered,
        activity_key=activity_key,
        timestamp_key=timestamp_key,
        case_id_key=case_key,
    )
    local_counts = case_sequences.value_counts(sort=False).to_dict()
    normalized_pm4py_counts = {
        tuple(sequence): int(count) for sequence, count in pm4py_counts.items()
    }
    if normalized_pm4py_counts != local_counts:
        raise RuntimeError(
            "Local case grouping disagrees with PM4Py variant extraction"
        )

    variant_rows = [
        {
            "variant_id": variant_id(sequence),
            "activity_sequence": json.dumps(sequence, ensure_ascii=False),
            "frequency": int(frequency),
            "sequence_length": len(sequence),
        }
        for sequence, frequency in local_counts.items()
    ]
    variants = pd.DataFrame(variant_rows).sort_values(
        ["frequency", "variant_id"],
        ascending=[False, True],
        kind="mergesort",
        ignore_index=True,
    )

    sequence_to_id = {
        sequence: variant_id(sequence) for sequence in local_counts
    }
    case_map = (
        case_sequences.apply(sequence_to_id.__getitem__)
        .rename("variant_id")
        .reset_index()
    )
    case_map = case_map.rename(columns={case_key: "case_id"})

    summary = ExtractionSummary(
        source=source,
        lifecycle_policy_requested=lifecycle_policy,
        lifecycle_policy_applied=applied_policy,
        events_before_filter=event_count_before,
        events_after_filter=len(ordered),
        cases=len(case_map),
        activities=int(ordered[activity_key].nunique()),
        variants=len(variants),
        timestamp_tie_events=timestamp_tie_events,
        min_variant_length=int(variants["sequence_length"].min()),
        max_variant_length=int(variants["sequence_length"].max()),
    )
    return variants, case_map, summary


def extract_xes_to_directory(
    xes_path: Path,
    output_dir: Path,
    *,
    lifecycle_policy: LifecyclePolicy = "auto",
    case_key: str = "case:concept:name",
    activity_key: str = "concept:name",
    timestamp_key: str = "time:timestamp",
    lifecycle_key: str = "lifecycle:transition",
) -> ExtractionSummary:
    events = pm4py.read_xes(
        str(xes_path),
        return_legacy_log_object=False,
        show_progress_bar=False,
    )
    variants, case_map, summary = extract_variants(
        events,
        source=str(xes_path),
        lifecycle_policy=lifecycle_policy,
        case_key=case_key,
        activity_key=activity_key,
        timestamp_key=timestamp_key,
        lifecycle_key=lifecycle_key,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    variants.to_parquet(output_dir / "variants.parquet", index=False)
    variants.to_csv(output_dir / "variants.csv", index=False, encoding="utf-8")
    case_map.to_parquet(output_dir / "case_variant_map.parquet", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary
