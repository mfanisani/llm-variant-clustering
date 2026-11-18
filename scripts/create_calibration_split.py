from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from llm_variant_clustering.evaluation_guards import (
    build_split_manifest,
    verify_split_manifest,
)
from llm_variant_clustering.sampling import stratified_variant_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variants", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sample-size", type=int, default=45)
    parser.add_argument("--seed", required=True)
    args = parser.parse_args()

    variants = pd.read_parquet(args.variants)
    calibration = stratified_variant_sample(
        variants,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    reserved = variants[
        ~variants["variant_id"].isin(calibration["variant_id"])
    ].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = args.output_dir / "calibration.parquet"
    reserved_path = args.output_dir / "reserved_evaluation.parquet"
    calibration.to_parquet(calibration_path, index=False)
    reserved.to_parquet(reserved_path, index=False)
    manifest = build_split_manifest(
        {
            "calibration": calibration_path,
            "reserved_evaluation": reserved_path,
        },
        seed=args.seed,
        protected_split="reserved_evaluation",
        examined_variant_ids=set(calibration["variant_id"]),
    )
    manifest["dataset_variant_count"] = len(variants)
    manifest["dataset_case_count"] = int(variants["frequency"].sum())
    manifest["calibration_case_count"] = int(calibration["frequency"].sum())
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    verify_split_manifest(manifest_path)


if __name__ == "__main__":
    main()

