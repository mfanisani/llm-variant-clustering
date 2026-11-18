from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from llm_variant_clustering.diagnostic_controls import (
    load_activity_embedding_map,
    run_diagnostic_controls,
)

FROZEN_P7_CANDIDATES = (
    "embeddings_p7_ordered.npy",
    "p7_ordered_embeddings.npy",
    "ordered_pyramid_embeddings.npy",
)
ACTIVITY_EMBEDDING_CANDIDATES = (
    "compact_activity_embeddings.npy",
    "activity_embeddings.npy",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic negative controls for a frozen P7 representation."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing the frozen P7 embedding artifacts.",
    )
    parser.add_argument(
        "--variants",
        type=Path,
        help="Variant table; defaults to variants.parquet in --dataset-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to <dataset-dir>/diagnostic_controls.",
    )
    parser.add_argument(
        "--fixed-k",
        "--clusters",
        dest="fixed_k",
        type=int,
        help=(
            "Fixed cluster count. If omitted, infer it from "
            "assignments_p7_ordered_auto.parquet."
        ),
    )
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    variants_path = args.variants or dataset_dir / "variants.parquet"
    if not variants_path.exists():
        raise FileNotFoundError(
            f"Variant table not found at {variants_path}; pass --variants explicitly"
        )
    output_dir = args.output_dir or dataset_dir / "diagnostic_controls"
    frozen_path = _find_frozen_p7(dataset_dir)
    activity_embeddings_path = _find_activity_embeddings(dataset_dir)
    fixed_k, k_source = infer_fixed_k(dataset_dir, args.fixed_k)

    variants = pd.read_parquet(variants_path)
    variants, variant_selection = _select_frozen_variants(dataset_dir, variants)
    activity_embeddings = load_activity_embedding_map(
        dataset_dir / "activity_embedding_index.parquet",
        activity_embeddings_path,
    )
    frozen = np.load(frozen_path, allow_pickle=False)
    frozen, row_alignment = _align_frozen_rows(dataset_dir, variants, frozen)

    run_diagnostic_controls(
        variants,
        activity_embeddings,
        frozen,
        fixed_k=fixed_k,
        output_dir=output_dir,
        seed=args.seed,
        manifest_context={
            "dataset_dir": str(dataset_dir),
            "variants_path": str(variants_path.resolve()),
            "activity_embedding_index": "activity_embedding_index.parquet",
            "activity_embeddings": activity_embeddings_path.name,
            "frozen_p7_embeddings": frozen_path.name,
            "frozen_row_alignment": row_alignment,
            "variant_selection": variant_selection,
            "fixed_k_source": k_source,
        },
    )


def infer_fixed_k(dataset_dir: Path, override: int | None) -> tuple[int, str]:
    if override is not None:
        return override, "cli_override"

    assignment_path = dataset_dir / "assignments_p7_ordered_auto.parquet"
    if not assignment_path.exists():
        raise ValueError(
            "--fixed-k is required when assignments_p7_ordered_auto.parquet "
            "is absent"
        )
    assignments = pd.read_parquet(assignment_path)
    if "cluster_label" not in assignments:
        raise ValueError("Auto assignments are missing cluster_label")
    fixed_k = int(assignments["cluster_label"].nunique(dropna=True))
    if assignments["cluster_label"].isna().any() or fixed_k < 2:
        raise ValueError("Auto assignments must define at least two non-null clusters")
    return fixed_k, assignment_path.name


def _find_frozen_p7(dataset_dir: Path) -> Path:
    for filename in FROZEN_P7_CANDIDATES:
        candidate = dataset_dir / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No frozen P7 embedding file found; expected one of "
        + ", ".join(FROZEN_P7_CANDIDATES)
    )


def _find_activity_embeddings(dataset_dir: Path) -> Path:
    for filename in ACTIVITY_EMBEDDING_CANDIDATES:
        candidate = dataset_dir / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No activity embedding file found; expected one of "
        + ", ".join(ACTIVITY_EMBEDDING_CANDIDATES)
    )


def _align_frozen_rows(
    dataset_dir: Path,
    variants: pd.DataFrame,
    frozen: np.ndarray,
) -> tuple[np.ndarray, str]:
    index_path = dataset_dir / "embedding_index.parquet"
    if not index_path.exists():
        if frozen.ndim != 2 or frozen.shape[0] != len(variants):
            raise ValueError("Frozen P7 rows do not match the variant table")
        return frozen, "variant_table_order_assumed_no_embedding_index"

    index = pd.read_parquet(index_path)
    if "variant_id" not in index or index["variant_id"].astype(str).duplicated().any():
        raise ValueError("Embedding index must contain unique variant_id values")
    if frozen.ndim != 2 or frozen.shape[0] != len(index):
        raise ValueError("Embedding index and frozen P7 row counts do not align")
    position = {
        str(variant_id): row
        for row, variant_id in enumerate(index["variant_id"])
    }
    requested = variants["variant_id"].astype(str).tolist()
    missing = sorted(set(requested).difference(position))
    if missing:
        raise ValueError(f"Frozen P7 embedding index is missing {len(missing)} variants")
    return (
        np.vstack([frozen[position[variant_id]] for variant_id in requested]),
        "reordered_by_embedding_index.parquet",
    )


def _select_frozen_variants(
    dataset_dir: Path,
    variants: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    index_path = dataset_dir / "embedding_index.parquet"
    if not index_path.exists():
        return variants, "all_variant_table_rows_no_embedding_index"
    index = pd.read_parquet(index_path)
    if "variant_id" not in index or index["variant_id"].astype(str).duplicated().any():
        raise ValueError("Embedding index must contain unique variant_id values")
    selected_ids = frozenset(index["variant_id"].astype(str))
    variant_ids = variants["variant_id"].astype(str)
    missing = sorted(selected_ids.difference(variant_ids))
    if missing:
        raise ValueError(f"Variant table is missing {len(missing)} frozen variants")
    selected = variants.loc[variant_ids.isin(selected_ids)].copy()
    selection = (
        "all_variant_table_rows"
        if len(selected) == len(variants)
        else f"embedding_index_subset_excluded_{len(variants) - len(selected)}_rows"
    )
    return selected, selection


if __name__ == "__main__":
    main()
