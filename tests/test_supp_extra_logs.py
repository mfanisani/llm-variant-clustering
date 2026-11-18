import json

import numpy as np
import pandas as pd
import pytest

import run_supp_extra_logs as supp
from llm_variant_clustering.uncertainty import paired_silhouette_difference


def _metrics(tmp_path, statuses: dict[str, str]) -> None:
    rows = [
        {"method": method, "status": status, "selected_k": 3 if status == "selected" else None}
        for method, status in statuses.items()
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "metrics_dynamic_k.csv", index=False)


def test_matched_k_uses_p7_automatic_k(tmp_path):
    _metrics(tmp_path, {supp.P7: "selected", supp.RAW: "selected", supp.NGRAM: "selected"})
    assert supp.matched_k(tmp_path) == (3, "dynamic")


def test_matched_k_falls_back_to_k8_on_abstention(tmp_path):
    _metrics(tmp_path, {supp.P7: "no_eligible_k", supp.RAW: "selected", supp.NGRAM: "selected"})
    assert supp.matched_k(tmp_path) == (supp.FALLBACK_K, "k8")


def test_heldout_uses_k8_only_for_abstaining_method(tmp_path):
    _metrics(tmp_path, {supp.P7: "no_eligible_k", supp.RAW: "selected", supp.NGRAM: "selected"})
    names = [path.name for path in supp.heldout_assignments(tmp_path)]
    assert names == [
        f"assignments_{supp.NGRAM}_dynamic.parquet",
        f"assignments_{supp.P7}_k8.parquet",
        f"assignments_{supp.RAW}_dynamic.parquet",
    ]


def test_bonferroni_rereads_stored_replicates_and_widens():
    rng = np.random.default_rng(0)
    points = np.vstack([rng.normal(0, 0.3, (20, 2)), rng.normal(3, 0.3, (20, 2))])
    from sklearn.metrics import pairwise_distances

    distances = pairwise_distances(points)
    good = np.repeat([0, 1], 20)
    bad = rng.integers(0, 2, 40)
    weights = np.ones(40)
    difference = paired_silhouette_difference(
        distances, good, distances, bad, weights, resamples=200
    )
    record = {"dataset": "toy", "first": supp.FIRST, "second": "raw_activity_label_bge",
              **difference.as_dict("difference_")}
    family = [record, {**record, "second": "activity2vec_log_context_d16"},
              {**record, "second": "activity2vec_log_context_d23"}]
    samples = {("toy", r["second"]): difference.samples for r in family}
    frame = supp.bonferroni(family, samples)
    assert len(frame) == 2
    assert (frame["bonferroni_family_size"] == 2).all()
    assert frame["bonferroni_confidence"].iloc[0] == pytest.approx(0.975)
    assert (frame["bonferroni_ci_lower"] <= frame["difference_ci_lower"]).all()
    assert (frame["bonferroni_ci_upper"] >= frame["difference_ci_upper"]).all()
    expected = difference.adjusted_interval(0.975)
    assert frame["bonferroni_ci_lower"].iloc[0] == pytest.approx(expected.lower)


def test_heldout_f1_intervals_ignore_summary_rows(tmp_path):
    rows = [
        {"method": "m", "scope": "whole_log", "fitness_precision_f1": 0.1, "test_cases": 100},
        {"method": "m", "scope": "cluster", "fitness_precision_f1": 0.5, "test_cases": 10},
        {"method": "m", "scope": "cluster", "fitness_precision_f1": 0.7, "test_cases": 30},
        {"method": "m", "scope": "test_case_weighted_aggregate",
         "fitness_precision_f1": 0.65, "test_cases": 40},
    ]
    path = tmp_path / "heldout.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    (result,) = supp.heldout_f1_intervals(path)
    assert result["cluster_count"] == 2
    assert result["f1_point_estimate"] == pytest.approx(0.65)


def test_llm_cost_sums_cache_usage(tmp_path, monkeypatch):
    cache = tmp_path / "evaluation/supp_extra_logs_v1/bpi2012/azure_activity_cache/ab"
    cache.mkdir(parents=True)
    for index, (prompt, completion) in enumerate([(500, 40), (520, 60)]):
        (cache / f"{index}.json").write_text(json.dumps({"provider_metadata": {
            "attempt_count": 1,
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
        }}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    cost = supp.llm_cost("bpi2012")
    assert cost == {"dataset": "bpi2012", "cache_entries": 2, "provider_attempts": 2,
                    "prompt_tokens": 1020, "completion_tokens": 100}


def test_log_registry_points_to_downloaded_raw_files():
    for log, spec in supp.LOGS.items():
        assert spec["xes"].startswith("datasets/raw/"), log
        assert spec["xes"].endswith((".xes", ".xes.gz")), log


def test_bonferroni_family_size_can_be_fixed_to_prespecified_family():
    rng = np.random.default_rng(1)
    points = np.vstack([rng.normal(0, 0.3, (15, 2)), rng.normal(3, 0.3, (15, 2))])
    from sklearn.metrics import pairwise_distances

    distances = pairwise_distances(points)
    difference = paired_silhouette_difference(
        distances, np.repeat([0, 1], 15), distances, rng.integers(0, 2, 30),
        np.ones(30), resamples=200,
    )
    record = {"dataset": "toy", "first": supp.FIRST, "second": "raw_activity_label_bge",
              **difference.as_dict("difference_")}
    frame = supp.bonferroni([record], {("toy", record["second"]): difference.samples},
                            family_size=supp.PRESPECIFIED_FAMILY_SIZE)
    assert supp.PRESPECIFIED_FAMILY_SIZE == 4
    assert frame["bonferroni_family_size"].iloc[0] == 4
    assert frame["bonferroni_confidence"].iloc[0] == pytest.approx(0.9875)


def test_p7_degenerate_detects_missing_benchmark_metrics(tmp_path):
    assert supp.p7_degenerate(tmp_path)
    _metrics(tmp_path, {supp.P7: "selected", supp.RAW: "selected", supp.NGRAM: "selected"})
    assert not supp.p7_degenerate(tmp_path)


def test_candidate_summary_reads_selected_or_abstains(tmp_path):
    pd.DataFrame({"k": [2, 3], "selected": [False, True],
                  "case_weighted_silhouette": [0.1, 0.2], "variant_silhouette": [0.1, 0.3],
                  "mean_seed_stability_ari": [1.0, 0.9], "min_cluster_cases": [5, 4]}
                 ).to_csv(tmp_path / f"dynamic_candidates_{supp.RAW}.csv", index=False)
    summary = supp._candidate_summary(tmp_path, supp.RAW)
    assert summary["status"] == "selected" and summary["selected_k"] == 3
    pd.DataFrame({"k": [2], "selected": [False]}).to_csv(
        tmp_path / f"dynamic_candidates_{supp.NGRAM}.csv", index=False)
    assert supp._candidate_summary(tmp_path, supp.NGRAM)["status"] == "no_eligible_k"
