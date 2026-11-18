from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import run_replay_matched_k as rk
from llm_variant_clustering.process_evaluation import select_variant_holdout_split


def _case_map(sizes: dict[str, int]) -> pd.DataFrame:
    rows = [(f"{variant}_case{i}", variant) for variant, n in sizes.items() for i in range(n)]
    return pd.DataFrame(rows, columns=["case_id", "variant_id"])


def _toy():
    sizes = {f"v{i:02d}": 1 + (i % 4) for i in range(40)}
    return _case_map(sizes)


def test_share_valid_seeds_are_ascending_and_in_window():
    counts = rk.variant_cases(_toy())
    found = rk.share_valid_seeds(counts, 5)
    seeds = [seed for seed, _ in found]
    assert seeds == sorted(seeds) and len(seeds) == 5
    for seed, split in found:
        assert 0.15 <= rk.test_case_share(split) <= 0.25
    # every skipped seed below the last one violates the window
    for seed in range(seeds[-1]):
        if seed not in seeds:
            share = rk.test_case_share(rk.split_for_seed(counts, seed))
            assert not 0.15 <= share <= 0.25


def test_split_for_seed_matches_library_selection():
    case_map = _toy()
    counts = rk.variant_cases(case_map)
    variants = list(counts.index)
    # two-cluster partition, both clusters large enough to be covered on most seeds
    frame = pd.DataFrame({"variant_id": variants,
                          "cluster_id": ["a" if i % 2 else "b" for i in range(len(variants))]})
    library, seed = select_variant_holdout_split(case_map, {"m": frame}, test_fraction=0.2,
                                                 case_fraction_tolerance=0.05)
    first = next(s for s, split in rk.share_valid_seeds(counts, 50)
                 if rk.coverage(frame, split)["feasible"])
    assert first == seed
    assert rk.split_for_seed(counts, seed).equals(library.reset_index(drop=True))


def test_coverage_reports_missing_test_and_train():
    split = pd.DataFrame({"variant_id": ["a", "b", "c", "d", "e"],
                          "split": ["train", "test", "train", "train", "test"]})
    ok = pd.DataFrame({"variant_id": list("abcde"), "cluster_id": ["x", "x", "y", "y", "y"]})
    assert rk.coverage(ok, split)["feasible"]
    bad = pd.DataFrame({"variant_id": list("abcde"), "cluster_id": ["x", "y", "z", "z", "y"]})
    result = rk.coverage(bad, split)
    assert not result["feasible"]
    assert result["clusters_without_test"] == "x;z"
    assert result["clusters_without_train"] == "y"


def test_aggregate_is_test_case_weighted():
    rows = pd.DataFrame({
        "scope": ["cluster", "cluster"], "test_cases": [1, 3],
        "fitness": [1.0, 0.5], "precision": [0.2, 0.6], "fitness_precision_f1": [0.4, 0.8],
    })
    result = rk.aggregate(rows)
    assert result["fitness_precision_f1"] == pytest.approx((0.4 + 3 * 0.8) / 4)
    assert result["fitness"] == pytest.approx((1.0 + 1.5) / 4)
    whole = pd.DataFrame({"scope": ["whole_log"], "test_cases": [9], "fitness": [0.9],
                          "precision": [0.3], "fitness_precision_f1": [0.45]})
    assert rk.aggregate(whole)["fitness_precision_f1"] == 0.45


def test_paired_summary_uses_only_jointly_feasible_splits():
    rows = []
    for seed, (p7, raw, raw_ok) in enumerate([(0.6, 0.5, True), (0.6, 0.7, True), (0.6, 0.1, False),
                                              (0.5, 0.5, True)]):
        rows.append({"log": "x", "hash_seed": seed, "method": "p7", "feasible": True,
                     "fitness_precision_f1": p7})
        rows.append({"log": "x", "hash_seed": seed, "method": "raw_label", "feasible": raw_ok,
                     "fitness_precision_f1": raw if raw_ok else np.nan})
    result = rk.paired_summary(pd.DataFrame(rows), others=("raw_label",)).iloc[0]
    assert result["n_paired_splits"] == 3
    assert result["median_diff"] == pytest.approx(0.0)
    assert result["min_diff"] == pytest.approx(-0.1)
    assert result["max_diff"] == pytest.approx(0.1)
    assert result["n_first_greater"] == 1 and result["n_ties"] == 1
    assert result["share_first_greater"] == pytest.approx(1 / 3)


def test_method_summary_counts_feasible_splits():
    per_split = pd.DataFrame({
        "log": ["x"] * 3, "method": ["p7"] * 3, "hash_seed": [0, 1, 2], "feasible": [True, False, True],
        "fitness_precision_f1": [0.2, np.nan, 0.4], "fitness": [0.9, np.nan, 0.8],
        "precision": [0.1, np.nan, 0.3]})
    row = rk.method_summary(per_split).iloc[0]
    assert row["n_splits"] == 3 and row["n_feasible"] == 2 and row["infeasible_seeds"] == "1"
    assert row["median_f1"] == pytest.approx(0.3)


def test_variants_moved_is_label_permutation_invariant():
    assert rk.variants_moved([0, 0, 1, 1], [1, 1, 0, 0]) == 0
    assert rk.variants_moved([0, 0, 1, 1], [1, 1, 0, 1]) == 1


@pytest.mark.skipif(not (rk.BENCH / "rtfm" / f"{rk.FROZEN}_variant_split.parquet").exists(),
                    reason="frozen RTFM split not available")
def test_frozen_rtfm_split_is_reproduced_by_seed_listing():
    frozen = rk.frozen_split("rtfm")
    case_map = rk.case_map_for("rtfm", frozenset(frozen["variant_id"]))
    counts = rk.variant_cases(case_map)
    seed = int(rk.frozen_manifest("rtfm")["split_seed"])
    assert rk.split_for_seed(counts, seed).equals(frozen.reset_index(drop=True))
    assert seed in [s for s, _ in rk.share_valid_seeds(counts, 20)]
