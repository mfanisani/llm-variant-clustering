from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from llm_variant_clustering.evaluation_guards import (
    assert_disjoint_splits,
    build_split_manifest,
    verify_split_manifest,
)


def test_disjoint_guard_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_splits({"development": ["v1"], "test": ["v1"]})


def test_manifest_detects_artifact_change(tmp_path: Path) -> None:
    development_path = tmp_path / "development.parquet"
    test_path = tmp_path / "test.parquet"
    pd.DataFrame({"variant_id": ["v1"]}).to_parquet(
        development_path, index=False
    )
    pd.DataFrame({"variant_id": ["v2"]}).to_parquet(test_path, index=False)
    manifest = build_split_manifest(
        {"development": development_path, "protected_test": test_path},
        seed="seed",
        protected_split="protected_test",
        examined_variant_ids={"v1"},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        __import__("json").dumps(manifest),
        encoding="utf-8",
    )
    verify_split_manifest(manifest_path)

    pd.DataFrame({"variant_id": ["v3"]}).to_parquet(test_path, index=False)
    with pytest.raises(ValueError, match="hash changed"):
        verify_split_manifest(manifest_path)

