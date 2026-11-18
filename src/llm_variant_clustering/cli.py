from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .variants import extract_xes_to_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract frequency-preserving process variants from XES."
    )
    parser.add_argument("xes_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--lifecycle-policy",
        choices=("auto", "complete", "all"),
        default="auto",
    )
    parser.add_argument("--case-key", default="case:concept:name")
    parser.add_argument("--activity-key", default="concept:name")
    parser.add_argument("--timestamp-key", default="time:timestamp")
    parser.add_argument("--lifecycle-key", default="lifecycle:transition")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = extract_xes_to_directory(
        args.xes_path,
        args.output_dir,
        lifecycle_policy=args.lifecycle_policy,
        case_key=args.case_key,
        activity_key=args.activity_key,
        timestamp_key=args.timestamp_key,
        lifecycle_key=args.lifecycle_key,
    )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()

