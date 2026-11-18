"""Fast checks of scripts/build_paper_tables_v2.py against the frozen sources."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_paper_tables_v2 as tables  # noqa: E402

pytestmark = pytest.mark.skipif(
    not tables.BENCH_CSV.exists() or not tables.FULL_SIZE_CSV.exists(),
    reason="frozen evaluation outputs not available",
)


@pytest.fixture(scope="module")
def built():
    return tables.build(None)


def test_every_emitted_value_matches_the_paper(built):
    assert built["errors"] == []
    checks = built["checks"]
    assert bool(checks.loc[~checks["pending"], "passed"].all())
    # only replay cells whose run has not finished may be pending
    assert set(checks.loc[checks["pending"], "table"]) <= {"table3"}


def test_table2_columns_reproduce_the_printed_values(built):
    t2 = built["table2"].set_index("log")
    printed = {
        "tagger": (0.567, 0.585, 0.432, 0.275, 0.881),
        "p7": (0.571, 0.628, 0.469, 0.316, 0.882),
        "raw": (0.539, 0.519, 0.445, 0.323, 0.799),
        "activity2vec_d16": (0.526, 0.652, 0.389, 0.329, 0.840),
        "ngram_own_k": (0.435, 0.323, 0.319, 0.146, 0.771),
    }
    for column, values in printed.items():
        for log, value in zip(tables.LOGS, values):
            assert abs(t2.loc[log, column] - value) <= tables.TOLERANCE, (column, log)
    assert t2["ngram_k"].tolist() == [6, 4, 2, 8, 3]


def test_full_size_intervals_replace_only_the_capped_logs(built):
    published = built["table2"].set_index("log")
    full = built["table2_full_size"].set_index("log")
    for log in tables.LOGS:
        capped = tables.EXPECTED_TABLE1[log][0] > tables.BOOTSTRAP_CAP
        for label in ("p7_minus_raw", "p7_minus_a2v16"):
            # the full-size run recomputes the point from rebuilt float32 distances
            assert full.loc[log, f"{label}_point"] == pytest.approx(published.loc[log, f"{label}_point"], abs=1e-6)
            same = full.loc[log, f"{label}_ci95_lower"] == published.loc[log, f"{label}_ci95_lower"]
            assert same != capped, (log, label)
            assert ("capped" in published.loc[log, f"{label}_bootstrap"]) == capped
            assert full.loc[log, f"{label}_bootstrap"].startswith("full-size")


def test_provenance_covers_every_numeric_table_cell(built):
    prov = built["provenance"]
    assert prov["source_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    keys = set(zip(prov["table"], prov["row"], prov["column"]))
    t2 = built["table2"]
    for _, row in t2.iterrows():
        for column in ("k", "p7", "raw", "activity2vec_d16", "tagger", "ngram_own_k", "ngram_k",
                       "p7_minus_raw_ci95_lower", "p7_minus_a2v16_ci95_upper"):
            name = column.replace("_ci95", "")
            assert ("table2", row["log"], name) in keys, (row["log"], column)


def test_table3_matched_k_replay_when_available(built):
    if built["table3"] is None:
        pytest.skip(built["table3_status"])
    t3 = built["table3"].set_index("log")
    assert abs(t3.loc["bpi2020", "raw"] - 0.624) <= tables.TOLERANCE
    assert t3.loc["bpi2019", "tagger_status"].startswith("infeasible")
    assert (t3["p7_k"] == [9, 2, 6, 2]).all()
    keys = set(zip(built["provenance"]["table"], built["provenance"]["row"], built["provenance"]["column"]))
    for log, row in t3.iterrows():
        for column in tables.TABLE3_COLUMNS.values():
            if row[f"{column}_status"] == "ok":
                assert ("table3", log, column) in keys, (log, column)
            else:
                assert pd.isna(row[column]), (log, column)
