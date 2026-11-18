"""Copy the shipped GPT-4.1 activity descriptors to the paths the pipeline reads.

The files in frozen_descriptors/ are the exact LLM-Desc descriptors behind the paper.
Restoring them replaces step 2 of the README (the GPT-4.1 normalization), so the paper's
numbers can be reproduced without Azure OpenAI access. Every file is checked against
frozen_descriptors/SHA256SUMS first. Existing files are left alone unless --overwrite.

    python scripts/restore_frozen_descriptors.py
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "frozen_descriptors"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--overwrite", action="store_true",
                        help="replace descriptor files that already exist")
    args = parser.parse_args()

    entries = [line.split() for line in (SOURCE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    for digest, name, target in entries:
        source = SOURCE / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            sys.exit(f"checksum mismatch: {source}")
        destination = ROOT / target
        if destination.exists() and not args.overwrite:
            print(f"kept     {target} (exists)")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"restored {target}")


if __name__ == "__main__":
    main()
