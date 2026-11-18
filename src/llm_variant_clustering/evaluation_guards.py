from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def variant_id_digest(variant_ids: Collection[str]) -> str:
    canonical = json.dumps(
        sorted(variant_ids),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assert_disjoint_splits(
    splits: Mapping[str, Collection[str]],
) -> None:
    names = list(splits)
    normalized = {name: set(splits[name]) for name in names}
    for name, values in normalized.items():
        if len(values) != len(splits[name]):
            raise ValueError(f"Split {name!r} contains duplicate variant IDs")
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = normalized[left].intersection(normalized[right])
            if overlap:
                raise ValueError(
                    f"Splits {left!r} and {right!r} overlap: "
                    f"{sorted(overlap)[:5]}"
                )


def build_split_manifest(
    split_files: Mapping[str, Path],
    *,
    seed: str,
    protected_split: str,
    examined_variant_ids: Collection[str],
) -> dict[str, Any]:
    split_ids: dict[str, list[str]] = {}
    entries: dict[str, Any] = {}
    for name, path in split_files.items():
        frame = pd.read_parquet(path)
        if "variant_id" not in frame:
            raise ValueError(f"Split file lacks variant_id: {path}")
        ids = frame["variant_id"].astype(str).tolist()
        split_ids[name] = ids
        entries[name] = {
            "path": str(path),
            "size": len(ids),
            "variant_id_sha256": variant_id_digest(ids),
            "file_sha256": sha256_file(path),
            "locked": name == protected_split,
        }

    assert_disjoint_splits(split_ids)
    examined = set(examined_variant_ids)
    protected_ids = set(split_ids[protected_split])
    if overlap := examined.intersection(protected_ids):
        raise ValueError(
            f"Protected split contains examined variants: {sorted(overlap)[:5]}"
        )
    return {
        "schema_version": 1,
        "seed": seed,
        "protected_split": protected_split,
        "examined_variant_id_sha256": variant_id_digest(examined),
        "examined_variant_count": len(examined),
        "splits": entries,
        "non_overlap_verified": True,
        "protected_unexamined_verified": True,
    }


def verify_split_manifest(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_ids: dict[str, list[str]] = {}
    for name, entry in manifest["splits"].items():
        path = Path(entry["path"])
        if not path.exists():
            raise ValueError(f"Split artifact is missing: {path}")
        if sha256_file(path) != entry["file_sha256"]:
            raise ValueError(f"Split artifact hash changed: {path}")
        ids = pd.read_parquet(path)["variant_id"].astype(str).tolist()
        if variant_id_digest(ids) != entry["variant_id_sha256"]:
            raise ValueError(f"Split variant IDs changed: {path}")
        split_ids[name] = ids
    assert_disjoint_splits(split_ids)

