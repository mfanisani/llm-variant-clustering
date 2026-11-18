from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from collections.abc import Collection

import pandas as pd


def _hash_score(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def _tertile(series: pd.Series) -> pd.Series:
    if len(series) < 3:
        return pd.Series("middle", index=series.index, dtype="string")
    percentile = series.rank(method="average", pct=True)
    return pd.cut(
        percentile,
        bins=[0.0, 1 / 3, 2 / 3, 1.0],
        labels=["low", "middle", "high"],
        include_lowest=True,
    ).astype("string")


def stratified_variant_sample(
    variants: pd.DataFrame,
    *,
    sample_size: int,
    seed: str,
    excluded_variant_ids: Collection[str] = (),
) -> pd.DataFrame:
    required = {
        "variant_id",
        "activity_sequence",
        "frequency",
        "sequence_length",
    }
    missing = sorted(required.difference(variants.columns))
    if missing:
        raise ValueError(f"Variant table is missing columns: {missing}")
    if sample_size < 1:
        raise ValueError("sample_size must be positive")

    candidates = variants.loc[
        ~variants["variant_id"].isin(set(excluded_variant_ids))
    ].copy()
    if sample_size > len(candidates):
        raise ValueError(
            f"Requested {sample_size} variants from {len(candidates)} candidates"
        )
    if candidates["variant_id"].duplicated().any():
        raise ValueError("variant_id values must be unique")

    candidates["_sequence"] = candidates["activity_sequence"].map(json.loads)
    if candidates["_sequence"].map(len).eq(0).any():
        raise ValueError("Activity sequences must not be empty")
    if not (
        candidates["_sequence"].map(len) == candidates["sequence_length"]
    ).all():
        raise ValueError("sequence_length disagrees with activity_sequence")

    candidates["frequency_tier"] = _tertile(candidates["frequency"])
    candidates["length_tier"] = _tertile(candidates["sequence_length"])
    candidates["has_repetition"] = candidates["_sequence"].map(
        lambda sequence: len(sequence) != len(set(sequence))
    )
    candidates["terminal_activity"] = candidates["_sequence"].map(
        lambda sequence: sequence[-1]
    )
    candidates["_stratum"] = candidates.apply(
        lambda row: (
            str(row["frequency_tier"]),
            str(row["length_tier"]),
            bool(row["has_repetition"]),
            str(row["terminal_activity"]),
        ),
        axis=1,
    )
    candidates["_hash"] = candidates["variant_id"].map(
        lambda variant: _hash_score(seed, variant)
    )

    buckets: dict[tuple[str, str, bool, str], deque[int]] = defaultdict(deque)
    ordered = candidates.sort_values("_hash", kind="mergesort")
    for index, stratum in zip(ordered.index, ordered["_stratum"], strict=True):
        buckets[stratum].append(index)

    stratum_order = sorted(
        buckets,
        key=lambda stratum: _hash_score(
            seed, json.dumps(stratum, ensure_ascii=False)
        ),
    )
    selected: list[int] = []
    while len(selected) < sample_size:
        added = False
        for stratum in stratum_order:
            if buckets[stratum] and len(selected) < sample_size:
                selected.append(buckets[stratum].popleft())
                added = True
        if not added:
            raise RuntimeError("Sampling exhausted candidates unexpectedly")

    result = candidates.loc[selected].copy()
    result.insert(0, "selection_order", range(1, len(result) + 1))
    result = result.drop(columns=["_sequence", "_stratum", "_hash"])
    return result.reset_index(drop=True)

