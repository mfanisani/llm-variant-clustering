"""Verify downloaded event logs against datasets/metadata/checksums.sha256.

Usage (from the repository root):  python datasets/verify_checksums.py
Missing files are reported but only mismatches make the script fail.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"
LIST = Path(__file__).resolve().parent / "metadata" / "checksums.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    failures = 0
    for line in LIST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(" ", 1)
        path = RAW / name.lstrip("*")
        if not path.exists():
            print(f"MISSING   {path.relative_to(RAW.parent.parent)}")
            continue
        status = "OK" if sha256(path) == expected else "MISMATCH"
        failures += status == "MISMATCH"
        print(f"{status:9} {path.relative_to(RAW.parent.parent)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
