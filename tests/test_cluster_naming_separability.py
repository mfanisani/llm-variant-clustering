import json
import math
import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx
import numpy as np
import openai
import pandas as pd
import pytest

import llm_variant_clustering.azure_openai_provider as azure_openai_provider
import run_cluster_naming_separability as cns
from llm_variant_clustering.azure_openai_provider import AzureOpenAIToolProvider
from llm_variant_clustering.prompts import PromptArtifact


def _toy_variants() -> pd.DataFrame:
    # Two obviously separable behaviours: a short "fast" path and a longer
    # "escalated" path that revisits "Review".
    rows = [
        ("v1", ["Create", "Approve", "Close"], 10, 3),
        ("v2", ["Create", "Approve", "Close"], 5, 3),
        ("v3", ["Create", "Review", "Review", "Approve", "Close"], 4, 5),
        ("v4", ["Create", "Review", "Review", "Reject", "Close"], 1, 5),
    ]
    return pd.DataFrame(
        rows, columns=["variant_id", "activity_sequence", "frequency", "sequence_length"]
    )


def _toy_partition() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variant_id": ["v1", "v2", "v3", "v4"],
            "cluster_label": [0, 0, 1, 1],
        }
    )


class TestComputeClusterProfiles:
    def test_profile_shapes_and_shares(self) -> None:
        profiles = cns.compute_cluster_profiles(_toy_variants(), _toy_partition())

        assert set(profiles) == {"0", "1"}
        cluster0 = profiles["0"]
        assert cluster0["size"]["n_variants"] == 2
        assert cluster0["size"]["n_cases"] == 15
        # 15 / (15 + 5) cases are in cluster 0.
        assert cluster0["size"]["case_share_of_log"] == pytest.approx(0.75)
        assert cluster0["most_common_start_activity"] == "Create"
        assert cluster0["most_common_end_activity"] == "Close"

    def test_evidence_ids_are_cluster_scoped(self) -> None:
        profiles = cns.compute_cluster_profiles(_toy_variants(), _toy_partition())

        for key, profile in profiles.items():
            cited_in_tables = {row["id"] for row in profile["activity_prevalence"]}
            cited_in_tables |= {row["id"] for row in profile["distinctive_activities"]}
            cited_in_tables |= {row["id"] for row in profile["top_directly_follows_transitions"]}
            cited_in_tables |= {row["id"] for row in profile["distinctive_transitions"]}
            cited_in_tables |= {row["id"] for row in profile["top_variants"]}
            # Every ID shown in a cluster's own tables must be declared in
            # that same cluster's evidence_ids (IDs like "A1" are reused
            # across clusters, since each cluster has its own ID namespace).
            assert cited_in_tables <= set(profile["evidence_ids"]), key

    def test_cluster_1_has_review_as_distinctive(self) -> None:
        profiles = cns.compute_cluster_profiles(_toy_variants(), _toy_partition())
        cluster1_activities = {
            row["activity"] for row in profiles["1"]["distinctive_activities"]
        }
        assert "Review" in cluster1_activities


class TestValidateBusinessClusterNames:
    def _profiles_by_code(self) -> dict:
        profiles = cns.compute_cluster_profiles(_toy_variants(), _toy_partition())
        letter_codes = cns.assign_letter_codes(list(profiles), log="toy", partition="p7")
        return {
            letter_codes[key]: {**value, "cluster_code": letter_codes[key]}
            for key, value in profiles.items()
        }, letter_codes

    def _valid_response(self, profiles_by_code: dict) -> dict:
        return {
            "schema_version": "p8",
            "partition_code": "toy:p7",
            "cluster_count": len(profiles_by_code),
            "clusters": [
                {
                    "cluster_code": code,
                    "business_name": f"Routine Order Processing Path {code}",
                    "definition": "Cases that move through the process without rework.",
                    "distinguishing_features": [
                        {
                            "description": "Straight-through handling",
                            "evidence_ids": [profile["evidence_ids"][0]],
                        }
                    ],
                    "evidence_ids": profile["evidence_ids"][:2],
                    "confidence": "high",
                    "ambiguity_note": "none",
                }
                for code, profile in profiles_by_code.items()
            ],
        }

    def test_valid_response_passes(self) -> None:
        profiles_by_code, letter_codes = self._profiles_by_code()
        response = self._valid_response(profiles_by_code)
        cns.validate_business_cluster_names(
            response,
            partition_code="toy:p7",
            letter_codes=letter_codes,
            profiles_by_code=profiles_by_code,
        )

    def test_rejects_evidence_from_another_cluster(self) -> None:
        profiles_by_code, letter_codes = self._profiles_by_code()
        response = self._valid_response(profiles_by_code)
        # Evidence IDs (A1, T1, V1, ...) are re-used symbols scoped to each
        # cluster's own profile, so a fabricated ID that cannot exist in any
        # profile is the only unambiguous way to test the "foreign evidence"
        # rejection path.
        response["clusters"][0]["evidence_ids"] = ["A999-does-not-exist"]
        with pytest.raises(ValueError, match="cites evidence outside"):
            cns.validate_business_cluster_names(
                response,
                partition_code="toy:p7",
                letter_codes=letter_codes,
                profiles_by_code=profiles_by_code,
            )

    def test_rejects_evidence_from_another_cluster_in_distinguishing_feature(self) -> None:
        profiles_by_code, letter_codes = self._profiles_by_code()
        response = self._valid_response(profiles_by_code)
        response["clusters"][0]["distinguishing_features"][0]["evidence_ids"] = [
            "A999-does-not-exist"
        ]
        with pytest.raises(ValueError, match="distinguishing feature cites evidence"):
            cns.validate_business_cluster_names(
                response,
                partition_code="toy:p7",
                letter_codes=letter_codes,
                profiles_by_code=profiles_by_code,
            )

    def test_rejects_banned_word(self) -> None:
        profiles_by_code, letter_codes = self._profiles_by_code()
        response = self._valid_response(profiles_by_code)
        response["clusters"][0]["business_name"] = "Miscellaneous Cluster Handling Path"
        with pytest.raises(ValueError, match="banned word"):
            cns.validate_business_cluster_names(
                response,
                partition_code="toy:p7",
                letter_codes=letter_codes,
                profiles_by_code=profiles_by_code,
            )

    def test_rejects_duplicate_name(self) -> None:
        profiles_by_code, letter_codes = self._profiles_by_code()
        response = self._valid_response(profiles_by_code)
        for cluster in response["clusters"]:
            cluster["business_name"] = "Routine Order Processing Path"
        with pytest.raises(ValueError, match="Duplicate business_name"):
            cns.validate_business_cluster_names(
                response,
                partition_code="toy:p7",
                letter_codes=letter_codes,
                profiles_by_code=profiles_by_code,
            )

    def test_rejects_missing_cluster(self) -> None:
        # The naming schema itself requires >= 2 clusters (K is always >= 2
        # in this pipeline), so a missing cluster is caught by schema
        # validation before the custom "Expected N cluster entries" check.
        profiles_by_code, letter_codes = self._profiles_by_code()
        response = self._valid_response(profiles_by_code)
        response["clusters"].pop()
        with pytest.raises(ValueError, match="Schema validation failed"):
            cns.validate_business_cluster_names(
                response,
                partition_code="toy:p7",
                letter_codes=letter_codes,
                profiles_by_code=profiles_by_code,
            )


class TestAssignLetterCodes:
    def test_is_a_bijection_onto_the_alphabet_prefix(self) -> None:
        codes = cns.assign_letter_codes(["b", "a", "c"], log="toy", partition="p7")
        assert set(codes) == {"a", "b", "c"}
        assert set(codes.values()) == {"A", "B", "C"}

    def test_is_deterministic_for_the_same_log_and_partition(self) -> None:
        first = cns.assign_letter_codes(["0", "1", "2"], log="bpi2020", partition="p7")
        second = cns.assign_letter_codes(["0", "1", "2"], log="bpi2020", partition="p7")
        assert first == second

    def test_differs_across_partitions_of_the_same_log(self) -> None:
        p7 = cns.assign_letter_codes(["0", "1", "2"], log="bpi2020", partition="p7")
        raw = cns.assign_letter_codes(["0", "1", "2"], log="bpi2020", partition="raw")
        assert p7 != raw


class TestMakeRandomPartition:
    def test_preserves_cluster_size_multiset(self) -> None:
        base = pd.DataFrame(
            {
                "variant_id": [f"v{i}" for i in range(20)],
                "cluster_label": [0] * 12 + [1] * 5 + [2] * 3,
            }
        )
        shuffled = cns.make_random_partition(base, seed=1)

        base_sizes = sorted(base["cluster_label"].value_counts().tolist())
        shuffled_sizes = sorted(shuffled["cluster_label"].value_counts().tolist())
        assert base_sizes == shuffled_sizes
        assert set(shuffled["variant_id"]) == set(base["variant_id"])

    def test_is_deterministic_given_a_seed(self) -> None:
        base = pd.DataFrame(
            {"variant_id": [f"v{i}" for i in range(10)], "cluster_label": [0] * 5 + [1] * 5}
        )
        first = cns.make_random_partition(base, seed=42)
        second = cns.make_random_partition(base, seed=42)
        pd.testing.assert_frame_equal(
            first.sort_values("variant_id").reset_index(drop=True),
            second.sort_values("variant_id").reset_index(drop=True),
        )

    def test_different_seeds_can_change_the_assignment(self) -> None:
        base = pd.DataFrame(
            {"variant_id": [f"v{i}" for i in range(30)], "cluster_label": list(range(3)) * 10}
        )
        first = cns.make_random_partition(base, seed=1)
        second = cns.make_random_partition(base, seed=2)
        assert not first["cluster_label"].equals(second["cluster_label"])


class TestCaseWeightedCohenKappa:
    def test_matches_hand_computed_example(self) -> None:
        # Confusion matrix (weighted):
        #            pred A  pred B
        # true A        40      10
        # true B        10      40
        # p_o = 80/100 = 0.8
        # p_e = (50/100 * 50/100) + (50/100 * 50/100) = 0.5
        # kappa = (0.8 - 0.5) / (1 - 0.5) = 0.6
        true = np.array(["A"] * 50 + ["B"] * 50)
        pred = np.array(["A"] * 40 + ["B"] * 10 + ["A"] * 10 + ["B"] * 40)
        weights = np.ones(100)
        kappa = cns.case_weighted_cohen_kappa(true, pred, weights)
        assert kappa == pytest.approx(0.6, abs=1e-9)

    def test_perfect_agreement_is_one(self) -> None:
        true = np.array(["A", "B", "A", "B"])
        kappa = cns.case_weighted_cohen_kappa(true, true.copy(), np.array([3, 1, 2, 4]))
        assert kappa == pytest.approx(1.0)

    def test_respects_case_weights_not_just_row_counts(self) -> None:
        # One heavily-weighted disagreement should pull kappa down more than
        # an unweighted count would suggest.
        true = np.array(["A", "A", "B"])
        pred = np.array(["A", "B", "B"])
        light = cns.case_weighted_cohen_kappa(true, pred, np.array([1.0, 1.0, 1.0]))
        heavy = cns.case_weighted_cohen_kappa(true, pred, np.array([1.0, 100.0, 1.0]))
        assert heavy < light


class TestKishEffectiveSampleSizeAmendment2:
    """Amendment 2: ESS is frequency-only (no LLM output) and quantifies how
    many "independent" judgements the case-weighted metrics rest on."""

    def test_uniform_weights_ess_equals_n(self) -> None:
        weights = np.ones(10)
        assert cns._kish_effective_sample_size(weights) == pytest.approx(10.0)

    def test_skewed_weights_give_a_much_smaller_ess_than_n(self) -> None:
        # One dominant variant (weight 1000) among 9 tiny ones: ESS should
        # collapse towards ~1, not stay near n=10, matching the pilot's
        # RTFM finding (ESS=2.1 on n=46 held-out variants).
        weights = np.array([1000.0] + [1.0] * 9)
        ess = cns._kish_effective_sample_size(weights)
        assert 1.0 <= ess < 2.0

    def test_two_equal_weight_classes_give_ess_near_two(self) -> None:
        weights = np.array([50.0, 50.0])
        assert cns._kish_effective_sample_size(weights) == pytest.approx(2.0)


class TestSeparabilityRowUnweightedIsPrimaryAmendment2:
    """Amendment 2: separability_row's ``kappa``/``accuracy``/etc. are the
    PRIMARY unweighted variant-level metrics; ``*_case_weighted`` fields are
    secondary/descriptive and may legitimately disagree when the case
    weights are skewed."""

    def test_unweighted_and_case_weighted_metrics_can_diverge(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(cns, "OUTPUT", tmp_path)
        # One heavily-weighted variant is judged wrong; four lightly-weighted
        # variants are judged right. Case-weighted accuracy should look much
        # worse than unweighted (per-variant) accuracy.
        weights_map = {"v1": 10_000, "v2": 1, "v3": 1, "v4": 1, "v5": 1}
        monkeypatch.setattr(cns, "_variant_weights", lambda log: weights_map)
        names_dir = tmp_path / "names" / "toylog"
        names_dir.mkdir(parents=True)
        (names_dir / "p7.json").write_text(
            json.dumps(
                {"response": {"clusters": [{"business_name": "Name One"}, {"business_name": "Name Two"}]}}
            ),
            encoding="utf-8",
        )
        frame = pd.DataFrame(
            {
                "variant_id": ["v1", "v2", "v3", "v4", "v5"],
                "true_name_code": ["N1", "N1", "N1", "N1", "N1"],
                "chosen_name_code": ["N2", "N1", "N1", "N1", "N1"],
                "correct": [False, True, True, True, True],
            }
        )
        _write_judgement_frame(tmp_path, "deepseek_v4_pro", "toylog", "p7", frame)

        row = cns.separability_row("deepseek_v4_pro", "toylog", "p7")

        assert row["accuracy"] == pytest.approx(0.8)  # unweighted: 4/5 correct
        assert row["accuracy_case_weighted"] == pytest.approx(4 / 10_004)  # dominated by v1's error
        assert row["ess"] < 2.0  # one dominant variant among 5

    def test_single_class_true_labels_give_nan_kappa_not_zero(
        self, tmp_path, monkeypatch
    ) -> None:
        # PI review: when the held-out true labels have zero variance (e.g.
        # RTFM's random2 partition, all true class N1), Cohen's kappa is
        # mathematically degenerate -- it evaluates to exactly 0 regardless
        # of prediction quality, because chance agreement equals observed
        # agreement by construction. That is not a real kappa of 0 and must
        # be reported as undefined (NaN, kappa_defined=False), both
        # unweighted and case-weighted.
        monkeypatch.setattr(cns, "OUTPUT", tmp_path)
        monkeypatch.setattr(cns, "_variant_weights", lambda log: {"v1": 1, "v2": 2, "v3": 3})
        names_dir = tmp_path / "names" / "toylog"
        names_dir.mkdir(parents=True)
        (names_dir / "random2.json").write_text(
            json.dumps({"response": {"clusters": [{"business_name": "Name One"}]}}),
            encoding="utf-8",
        )
        frame = pd.DataFrame(
            {
                "variant_id": ["v1", "v2", "v3"],
                "true_name_code": ["N1", "N1", "N1"],
                "chosen_name_code": ["N1", "N2", "N1"],
                "correct": [True, False, True],
            }
        )
        _write_judgement_frame(tmp_path, "deepseek_v4_pro", "toylog", "random2", frame)

        row = cns.separability_row("deepseek_v4_pro", "toylog", "random2")

        assert row["kappa_defined"] is False
        assert math.isnan(row["kappa"])
        assert math.isnan(row["kappa_case_weighted"])


class TestClassify:
    def test_positive_when_interval_excludes_zero_above(self) -> None:
        assert cns._classify(0.01, 0.2) == "positive"

    def test_negative_when_interval_excludes_zero_below(self) -> None:
        assert cns._classify(-0.2, -0.01) == "negative"

    def test_includes_zero_otherwise(self) -> None:
        assert cns._classify(-0.1, 0.1) == "includes_zero"
        assert cns._classify(0.0, 0.1) == "includes_zero"
        assert cns._classify(-0.1, 0.0) == "includes_zero"

    def test_undefined_when_bounds_are_nan(self) -> None:
        # All bootstrap resamples dropped (e.g. a tiny held-out set where
        # every resample is single-class on some side): the CI is undefined,
        # not "includes zero" -- must not be misread as a null result.
        assert cns._classify(float("nan"), float("nan")) == "undetermined"


class TestPairedKappaBootstrap:
    def test_uses_identical_resample_indices_for_both_partitions(self) -> None:
        rng = np.random.default_rng(0)
        n = 60
        true = rng.choice(["A", "B", "C"], size=n)
        pred_a = true.copy()
        # pred_b disagrees with the truth half the time.
        pred_b = true.copy()
        flip_mask = rng.random(n) < 0.5
        pred_b[flip_mask] = rng.choice(["A", "B", "C"], size=int(flip_mask.sum()))
        weights = rng.integers(1, 5, size=n).astype(np.float64)

        indices = cns.resample_indices(n, 300, seed=123)
        result = cns.paired_kappa_bootstrap(
            true, pred_a, true, pred_b, weights, indices=indices
        )

        # A should be a perfect predictor and B an imperfect one, so A - B > 0.
        assert result["point_estimate"] > 0
        assert result["class"] == "positive"
        assert result["resamples"] <= indices.shape[0]

    def test_resample_indices_is_deterministic_given_a_seed(self) -> None:
        first = cns.resample_indices(25, 50, seed=7)
        second = cns.resample_indices(25, 50, seed=7)
        np.testing.assert_array_equal(first, second)

    def test_resample_indices_shared_across_partitions_gives_identical_subsets(self) -> None:
        # The whole point of sharing `indices` across p7/raw/a2v comparisons
        # for the same log is that every comparison resamples the *same*
        # held-out variants on each bootstrap draw.
        indices_a = cns.resample_indices(40, 20, seed=99)
        indices_b = cns.resample_indices(40, 20, seed=99)
        np.testing.assert_array_equal(indices_a, indices_b)


def _write_judgement_frame(
    tmp_path, judge_key: str, log: str, partition: str, frame: pd.DataFrame
) -> None:
    out_dir = tmp_path / "judgements" / judge_key / log
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_dir / f"{partition}.parquet", index=False)


class TestAlignedArraysB1Regression:
    """B1 regression: two partitions covering the identical variant_id set
    (or overlapping variant_id sets), with on-disk rows in different order,
    must never be silently misaligned -- and a genuine mismatch must use
    each partition's OWN true codes, never reuse p7's true codes for the
    other partition (the historic B1 bug)."""

    def test_identical_variant_set_in_reversed_row_order_is_still_aligned(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(cns, "OUTPUT", tmp_path)
        variant_ids = [f"v{i}" for i in range(1, 7)]
        true_codes = ["N1", "N1", "N1", "N2", "N2", "N2"]
        pred_codes = ["N1", "N1", "N2", "N2", "N2", "N1"]  # v3 and v6 wrong

        p7_frame = pd.DataFrame(
            {
                "variant_id": variant_ids,
                "true_name_code": true_codes,
                "chosen_name_code": pred_codes,
                "correct": [t == p for t, p in zip(true_codes, pred_codes)],
            }
        )
        # Same underlying variant_id -> true/pred mapping, but inserted in
        # reverse row order: exactly the historic B1 trigger (two partitions
        # covering an identical variant_id set whose per-partition judge-batch
        # shuffle put them on disk in a different row order).
        reversed_frame = p7_frame.iloc[::-1].reset_index(drop=True)

        _write_judgement_frame(tmp_path, "deepseek_v4_pro", "toylog", "p7", p7_frame)
        _write_judgement_frame(
            tmp_path, "deepseek_v4_pro", "toylog", "random1", reversed_frame
        )

        weights_map = {v: 1 for v in variant_ids}
        true_a, pred_a, weights_a, index_a = cns._aligned_arrays(
            "deepseek_v4_pro", "toylog", "p7", weights_map
        )
        true_b, pred_b, weights_b, index_b = cns._aligned_arrays(
            "deepseek_v4_pro", "toylog", "random1", weights_map
        )

        # Both frames describe the identical judgements, just written in a
        # different row order, so arrays aligned by variant_id must be
        # identical -- not merely the same length.
        assert list(index_a) == list(index_b) == sorted(variant_ids)
        np.testing.assert_array_equal(true_a, true_b)
        np.testing.assert_array_equal(pred_a, pred_b)
        np.testing.assert_array_equal(weights_a, weights_b)

    def test_run_analyze_diff_uses_each_partitions_own_true_codes(
        self, tmp_path, monkeypatch
    ) -> None:
        """When p7's and another partition's judged samples only partially
        overlap, ``run_analyze``'s ``_diff`` closure must restrict to the
        common variants and score each side using its OWN true/pred codes.
        If it instead (buggily) reused p7's true codes to score the other
        partition's predictions, this specific fixture would look like
        perfect agreement (point estimate ~0.0) instead of the real,
        strongly positive difference (~2.0)."""
        monkeypatch.setattr(cns, "OUTPUT", tmp_path)
        monkeypatch.setattr(cns, "LOG_SPECS", {"toylog": {}})
        monkeypatch.setattr(
            cns, "_variant_weights", lambda log: {f"v{i}": 1 for i in range(1, 9)}
        )
        # Bypass the real (network-loading) BGE embedder entirely.
        monkeypatch.setattr(cns, "name_distinctness", lambda names: 0.0)

        names_dir = tmp_path / "names" / "toylog"
        names_dir.mkdir(parents=True)
        for partition in ("p7", "raw"):
            (names_dir / f"{partition}.json").write_text(
                json.dumps(
                    {
                        "response": {
                            "clusters": [
                                {"business_name": "Name One"},
                                {"business_name": "Name Two"},
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

        # p7: variants v1..v6, perfect agreement -> kappa = +1.0 on any subset.
        p7_frame = pd.DataFrame(
            {
                "variant_id": ["v1", "v2", "v3", "v4", "v5", "v6"],
                "true_name_code": ["N1", "N1", "N1", "N1", "N2", "N2"],
                "chosen_name_code": ["N1", "N1", "N1", "N1", "N2", "N2"],
                "correct": [True] * 6,
            }
        )
        # raw: variants v3..v8 (only v3-v6 overlap with p7). raw's OWN true
        # codes are the flip of p7's true codes for the shared variants, and
        # raw's chosen (predicted) codes are made to equal p7's true codes
        # for those same variants. If the diff computation (buggily) reused
        # p7's true codes to score raw's predictions, it would look like
        # perfect agreement (kappa=+1.0); using raw's own true codes, it is
        # total disagreement (kappa=-1.0).
        raw_frame = pd.DataFrame(
            {
                "variant_id": ["v8", "v7", "v6", "v5", "v4", "v3"],
                "true_name_code": ["N1", "N2", "N2", "N2", "N1", "N1"],
                "chosen_name_code": ["N2", "N1", "N1", "N1", "N2", "N2"],
                "correct": [False] * 6,
            }
        )
        _write_judgement_frame(tmp_path, "deepseek_v4_pro", "toylog", "p7", p7_frame)
        _write_judgement_frame(tmp_path, "deepseek_v4_pro", "toylog", "raw", raw_frame)

        cns.run_analyze(allow_incomplete=False)

        differences = pd.read_csv(tmp_path / "separability_differences.csv")
        # Amendment 2: run_analyze now emits one row per weighting scheme
        # (unweighted primary + case-weighted secondary); this fixture's
        # uniform weights=1 make both schemes identical, so pick the
        # primary (unweighted) row for the B1 assertion below.
        row = differences[
            (differences["judge"] == "deepseek_v4_pro")
            & (differences["log"] == "toylog")
            & (differences["comparison"] == "p7_minus_raw")
            & (differences["weighting"] == "unweighted")
        ]
        assert len(row) == 1
        point = row.iloc[0]["nominal_point_estimate"]
        # Fixed computation: ~2.0 (kappa_p7=+1.0 on the common v3-v6 subset
        # using p7's own codes, kappa_raw=-1.0 using raw's own codes). The
        # historic bug (reusing p7's true codes for raw) would silently
        # collapse this to ~0.0.
        assert point == pytest.approx(2.0, abs=1e-9)


def _write_names_file(tmp_path, log: str, partition: str, n_clusters: int = 2) -> None:
    names_dir = tmp_path / "names" / log
    names_dir.mkdir(parents=True, exist_ok=True)
    (names_dir / f"{partition}.json").write_text(
        json.dumps(
            {
                "response": {
                    "clusters": [
                        {"business_name": f"Name {i}"} for i in range(n_clusters)
                    ]
                }
            }
        ),
        encoding="utf-8",
    )


class TestHN1IncompleteRandomsAmendment1b:
    """Amendment 1b: H-N1 requires all 3 random draws (random1-3) present
    for a log; with fewer than 3, the log is "incomplete", not
    informative/uninformative."""

    def _write_p7_and_random(self, tmp_path, monkeypatch, random_partitions):
        monkeypatch.setattr(cns, "OUTPUT", tmp_path)
        monkeypatch.setattr(cns, "LOG_SPECS", {"toylog": {}})
        monkeypatch.setattr(
            cns, "_variant_weights", lambda log: {f"v{i}": 1 for i in range(1, 7)}
        )
        monkeypatch.setattr(cns, "name_distinctness", lambda names: 0.0)

        variant_ids = [f"v{i}" for i in range(1, 7)]
        true_codes = ["N1", "N1", "N1", "N2", "N2", "N2"]
        p7_pred = ["N1", "N1", "N1", "N2", "N2", "N2"]  # perfect
        random_pred = ["N2", "N1", "N2", "N1", "N2", "N1"]  # much worse

        def _frame(pred):
            return pd.DataFrame(
                {
                    "variant_id": variant_ids,
                    "true_name_code": true_codes,
                    "chosen_name_code": pred,
                    "correct": [t == p for t, p in zip(true_codes, pred)],
                }
            )

        _write_names_file(tmp_path, "toylog", "p7")
        _write_judgement_frame(tmp_path, "deepseek_v4_pro", "toylog", "p7", _frame(p7_pred))
        for partition in random_partitions:
            _write_names_file(tmp_path, "toylog", partition)
            _write_judgement_frame(
                tmp_path, "deepseek_v4_pro", "toylog", partition, _frame(random_pred)
            )

    def test_fewer_than_three_randoms_marks_the_log_incomplete(
        self, tmp_path, monkeypatch
    ) -> None:
        self._write_p7_and_random(tmp_path, monkeypatch, ["random1"])
        cns.run_analyze(allow_incomplete=False)
        decision = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
        assert decision["h_n1"]["incomplete_logs"] == ["toylog"]
        assert "toylog" not in decision["h_n1"]["informative_logs"]
        assert "toylog" not in decision["h_n1"]["uninformative_logs"]

    def test_all_three_randoms_present_is_classified_normally(
        self, tmp_path, monkeypatch
    ) -> None:
        self._write_p7_and_random(tmp_path, monkeypatch, ["random1", "random2", "random3"])
        cns.run_analyze(allow_incomplete=False)
        decision = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
        assert decision["h_n1"]["incomplete_logs"] == []
        assert decision["h_n1"]["informative_logs"] == ["toylog"]


class TestHN1PrimaryMetricAndSecondaryReportingAmendment2:
    """Amendment 2: the primary metric driving H-N1/H-N2/H-N3 is unweighted
    variant-level kappa; case-weighted results are still reported as
    secondary/descriptive, and the per-log Kish ESS is recorded."""

    def test_decision_reports_primary_metric_and_case_weighted_secondary(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(cns, "OUTPUT", tmp_path)
        monkeypatch.setattr(cns, "LOG_SPECS", {"toylog": {}})
        monkeypatch.setattr(
            cns, "_variant_weights", lambda log: {f"v{i}": 1 for i in range(1, 7)}
        )
        monkeypatch.setattr(cns, "name_distinctness", lambda names: 0.0)

        variant_ids = [f"v{i}" for i in range(1, 7)]
        true_codes = ["N1", "N1", "N1", "N2", "N2", "N2"]
        p7_pred = ["N1", "N1", "N1", "N2", "N2", "N2"]
        random_pred = ["N2", "N1", "N2", "N1", "N2", "N1"]

        def _frame(pred):
            return pd.DataFrame(
                {
                    "variant_id": variant_ids,
                    "true_name_code": true_codes,
                    "chosen_name_code": pred,
                    "correct": [t == p for t, p in zip(true_codes, pred)],
                }
            )

        for partition, pred in (
            ("p7", p7_pred),
            ("random1", random_pred),
            ("random2", random_pred),
            ("random3", random_pred),
        ):
            _write_names_file(tmp_path, "toylog", partition)
            _write_judgement_frame(tmp_path, "deepseek_v4_pro", "toylog", partition, _frame(pred))

        cns.run_analyze(allow_incomplete=False)

        decision = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
        assert decision["h_n1"]["primary_metric"] == "unweighted_variant_level_kappa"
        assert len(decision["h_n1"]["secondary_case_weighted"]) == 1
        assert decision["h_n1"]["secondary_case_weighted"][0]["log"] == "toylog"
        assert decision["ess_by_log"]["toylog"] == pytest.approx(6.0)  # uniform weights

        differences = pd.read_csv(tmp_path / "separability_differences.csv")
        random_rows = differences[
            (differences["log"] == "toylog") & (differences["comparison"] == "p7_minus_random")
        ]
        # Both weighting schemes must be present for every comparison.
        assert sorted(random_rows["weighting"]) == ["case_weighted", "unweighted"]


class TestAllResamplesDroppedNeverInformativeAnalyzeCrash:
    """Regression for the real full-run `analyze` crash (IndexError from
    ``np.percentile`` on an empty resample array, RTFM-sized p7_minus_random):
    a log where every bootstrap resample is single-class must come out of
    ``run_analyze`` as "undetermined"/low-information, never informative,
    and must not raise."""

    def test_single_class_p7_log_is_undetermined_not_informative(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(cns, "OUTPUT", tmp_path)
        monkeypatch.setattr(cns, "LOG_SPECS", {"toylog": {}})
        monkeypatch.setattr(cns, "_variant_weights", lambda log: {"v1": 1, "v2": 1})
        monkeypatch.setattr(cns, "name_distinctness", lambda names: 0.0)

        variant_ids = ["v1", "v2"]
        true_codes = ["N1", "N1"]  # single true class: every resample subset
        # of this 2-variant log is single-class, so every bootstrap resample
        # is dropped for any comparison involving p7.

        def _frame(pred):
            return pd.DataFrame(
                {
                    "variant_id": variant_ids,
                    "true_name_code": true_codes,
                    "chosen_name_code": pred,
                    "correct": [t == p for t, p in zip(true_codes, pred)],
                }
            )

        for partition, pred in (
            ("p7", ["N1", "N1"]),
            ("raw", ["N1", "N1"]),
            ("random1", ["N1", "N1"]),
            ("random2", ["N1", "N1"]),
            ("random3", ["N1", "N1"]),
        ):
            _write_names_file(tmp_path, "toylog", partition)
            _write_judgement_frame(tmp_path, "deepseek_v4_pro", "toylog", partition, _frame(pred))
            _write_judgement_frame(tmp_path, "gpt4o", "toylog", partition, _frame(pred))

        cns.run_analyze(allow_incomplete=False)  # must not raise IndexError

        decision = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
        assert "toylog" not in decision["h_n1"]["informative_logs"]
        assert "toylog" in decision["h_n1"]["low_information_logs"]
        # PI review: an "undetermined" (0-valid-resample) log is its own
        # bucket, distinct from "uninformative" (a defined-but-non-significant
        # interval). It must never land in uninformative_logs.
        assert "toylog" in decision["h_n1"]["undetermined_logs"]
        assert "toylog" not in decision["h_n1"]["uninformative_logs"]

        differences = pd.read_csv(tmp_path / "separability_differences.csv")
        random_row = differences[
            (differences["log"] == "toylog")
            & (differences["comparison"] == "p7_minus_random")
            & (differences["weighting"] == "unweighted")
        ].iloc[0]
        assert random_row["nominal_class"] == "undetermined"
        assert random_row["nominal_resamples"] == 0
        assert random_row["nominal_dropped_resamples_fraction"] == pytest.approx(1.0)
        assert random_row["nominal_low_information"] == True  # noqa: E712


class TestLowInformationFlagAmendment2:
    """Amendment 2: >5% of resamples dropped for a log x comparison is
    flagged "low-information" (visible in the tables and decision.json);
    it still counts towards the decision, it is not an exclusion."""

    def test_small_single_class_prone_sample_is_flagged_low_information(self) -> None:
        # n=4, two classes of 2 each: many bootstrap resamples will draw a
        # single-class subset, so the dropped fraction should clear 5%.
        true = np.array(["N1", "N1", "N2", "N2"])
        pred_a = np.array(["N1", "N1", "N2", "N2"])
        pred_b = np.array(["N2", "N2", "N1", "N1"])
        weights = np.ones(4)
        indices = cns.resample_indices(4, 500, seed=1)
        result = cns.paired_kappa_bootstrap(true, pred_a, true, pred_b, weights, indices=indices)
        assert result["dropped_resamples_fraction"] > cns.LOW_INFORMATION_DROPPED_FRACTION

    def test_large_balanced_sample_is_not_flagged_low_information(self) -> None:
        rng = np.random.default_rng(0)
        n = 200
        true = rng.choice(["N1", "N2"], size=n)
        pred_a = true.copy()
        pred_b = true.copy()
        weights = np.ones(n)
        indices = cns.resample_indices(n, 500, seed=2)
        result = cns.paired_kappa_bootstrap(true, pred_a, true, pred_b, weights, indices=indices)
        assert result["dropped_resamples_fraction"] <= cns.LOW_INFORMATION_DROPPED_FRACTION

    def test_all_resamples_dropped_does_not_crash_and_is_undefined(self) -> None:
        # n=2 single-class subsets are nearly guaranteed on every resample of
        # size 2 (both draws will very often be the same element), so every
        # resample can plausibly be dropped. Force the degenerate case
        # directly via an empty samples array, as real RTFM-sized data does
        # for p7_minus_random when every side needs >=2 classes at once
        # (B7/B2): this must not raise ``IndexError`` from ``np.percentile``
        # on an empty array.
        result = cns._percentile_interval(
            0.1234,
            np.asarray([]),
            0.95,
            dropped_resamples=2000,
            requested_resamples=2000,
        )
        assert result["resamples"] == 0
        assert result["dropped_resamples_fraction"] == pytest.approx(1.0)
        assert result["low_information"] is True
        assert math.isnan(result["ci_lower"])
        assert math.isnan(result["ci_upper"])
        assert cns._classify(result["ci_lower"], result["ci_upper"]) == "undetermined"

    def test_random_mean_diff_all_dropped_feeds_percentile_interval_safely(self) -> None:
        # Reproduces the real p7_minus_random crash path end-to-end: a tiny
        # held-out set (n=3) where p7's true codes are single-class, so
        # _random_mean_diff_samples drops every resample and
        # _percentile_interval must not crash on the empty result.
        true_p7 = np.array(["N1", "N1", "N1"])
        pred_p7 = np.array(["N1", "N1", "N2"])
        weights = np.ones(3)
        random_frames = [(true_p7.copy(), pred_p7.copy())]
        indices = cns.resample_indices(3, 50, seed=3)
        point, samples, dropped = cns._random_mean_diff_samples(
            true_p7, pred_p7, weights, random_frames, indices
        )
        assert samples.size == 0
        assert dropped == indices.shape[0]
        interval = cns._percentile_interval(
            point, samples, 0.95, dropped_resamples=dropped, requested_resamples=indices.shape[0]
        )
        assert interval["low_information"] is True
        assert math.isnan(interval["ci_lower"])


class TestPromptsDoNotLeakIdentifiersB3:
    """B3: prompts rendered for the namer and judge must never contain raw
    log names, partition names, or batch_id identifiers in natural/leaky
    form -- only opaque hashed codes (``_opaque_code``)."""

    def test_naming_prompt_does_not_leak_log_or_partition_name(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(cns, "OUTPUT", tmp_path)
        profiles_dir = tmp_path / "profiles" / "rtfm"
        profiles_dir.mkdir(parents=True)
        profiles = cns.compute_cluster_profiles(_toy_variants(), _toy_partition())
        (profiles_dir / "random1.json").write_text(json.dumps(profiles), encoding="utf-8")

        real_render_prompt = cns.render_prompt
        captured: dict[str, str] = {}

        def spy_render_prompt(*args, **kwargs):
            artifact = real_render_prompt(*args, **kwargs)
            captured["naming"] = artifact.rendered_prompt
            return artifact

        monkeypatch.setattr(cns, "render_prompt", spy_render_prompt)

        result = cns.name_log_partition(
            "rtfm",
            "random1",
            endpoint="https://example.test/",
            deployment="dep",
            model_version="v1",
            expected_model="dep",
            max_tokens=100,
            repair_max_tokens=200,
            cache_dir=tmp_path / "cache",
            dry_run=True,
        )
        assert result["dry_run"] is True
        rendered = captured["naming"]
        assert "rtfm" not in rendered.lower()
        assert "random1" not in rendered.lower()

    def test_judging_prompt_does_not_leak_log_or_partition_name(
        self, monkeypatch, tmp_path
    ) -> None:
        real_render_prompt = cns.render_prompt
        captured: dict[str, str] = {}

        def spy_render_prompt(*args, **kwargs):
            artifact = real_render_prompt(*args, **kwargs)
            captured["judging"] = artifact.rendered_prompt
            return artifact

        monkeypatch.setattr(cns, "render_prompt", spy_render_prompt)

        candidate_set = {
            "candidate_names": [
                {"name_code": "N1", "business_name": "Routine Handling", "definition": "d"},
                {"name_code": "unclear", "business_name": "unclear", "definition": "none"},
            ],
            "name_codes": ["N1"],
            "name_code_for_letter": {"A": "N1"},
            "label_map": {"v1": "0", "v2": "0"},
            "letter_codes": {"0": "A"},
        }
        sequence_map = {"v1": ["Create", "Approve"], "v2": ["Create", "Reject"]}

        result = cns.judge_one_batch(
            "rtfm",
            "random1",
            0,
            ["v1", "v2"],
            judge_key="deepseek_v4_pro",
            endpoint="https://example.test/",
            deployment="dep",
            model_version="v1",
            expected_model="dep",
            max_tokens=100,
            repair_max_tokens=200,
            cache_dir=tmp_path / "cache",
            sequence_map=sequence_map,
            candidate_set=candidate_set,
            dry_run=True,
        )
        assert result["log"] == "rtfm" and result["partition"] == "random1"
        rendered = captured["judging"]
        assert "rtfm" not in rendered.lower()
        assert "random1" not in rendered.lower()


def _fake_tool_completion(
    schema_name: str, payload: dict[str, Any], *, model: str = "deepseek-v4-pro"
) -> SimpleNamespace:
    """Mimic the OpenAI SDK's chat-completion-with-tool-call response shape
    that ``AzureOpenAIToolProvider.generate`` consumes."""
    return SimpleNamespace(
        id="resp-1",
        model=model,
        usage=None,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name=schema_name, arguments=json.dumps(payload)
                            )
                        )
                    ]
                )
            )
        ],
    )


def _make_rate_limit_error(status_code: int = 429) -> openai.RateLimitError:
    request = httpx.Request("POST", "https://example.test/chat/completions")
    response = httpx.Response(status_code, request=request)
    return openai.RateLimitError("rate limited", response=response, body=None)


class TestProviderRetryBatteryB6:
    """B6: ``AzureOpenAIToolProvider.generate``'s retry/backoff behaviour on
    transient (429) errors."""

    def _provider(self, *, max_attempts: int = 3) -> AzureOpenAIToolProvider:
        return AzureOpenAIToolProvider(
            endpoint="https://example.test/",
            deployment="deepseek-v4-pro",
            api_version="2025-01-01-preview",
            parameters=cns.GenerationParameters(
                temperature=0.0, seed=None, max_output_tokens=100
            ),
            schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
            schema_name="probe",
            max_attempts=max_attempts,
            backoff_seconds=0.001,
            max_backoff_seconds=0.001,
        )

    def test_429_then_success_on_retry(self, monkeypatch) -> None:
        monkeypatch.setattr(azure_openai_provider.time, "sleep", lambda s: None)
        provider = self._provider(max_attempts=3)
        success = _fake_tool_completion("probe", {"ok": True})
        mock_create = MagicMock(side_effect=[_make_rate_limit_error(), success])
        monkeypatch.setattr(provider.client.chat.completions, "create", mock_create)

        generation = provider.generate("SYSTEM\ns\n\nUSER\nu")

        assert generation.response == {"ok": True}
        assert mock_create.call_count == 2

    def test_429_exhausted_raises_after_max_attempts(self, monkeypatch) -> None:
        monkeypatch.setattr(azure_openai_provider.time, "sleep", lambda s: None)
        provider = self._provider(max_attempts=3)
        mock_create = MagicMock(
            side_effect=[_make_rate_limit_error() for _ in range(3)]
        )
        monkeypatch.setattr(provider.client.chat.completions, "create", mock_create)

        with pytest.raises(openai.RateLimitError):
            provider.generate("SYSTEM\ns\n\nUSER\nu")
        # No further retry beyond max_attempts.
        assert mock_create.call_count == 3


class TestGenerateWithRetriesAndRepairsB6:
    """B6: fatal-vs-retryable classification and the validation/repair loop
    inside ``generate_with_retries_and_repairs``."""

    def _base_params(self, max_tokens: int = 100) -> cns.GenerationParameters:
        return cns.GenerationParameters(
            temperature=0.0, seed=None, max_output_tokens=max_tokens
        )

    def test_repair_path_succeeds_and_raises_max_tokens_on_repair_attempt(self) -> None:
        recorded_max_tokens: list[int] = []
        call_count = {"n": 0}

        def fake_provider_factory(parameters: cns.GenerationParameters):
            recorded_max_tokens.append(parameters.max_output_tokens)

            class _FakeProvider:
                def generate(self, prompt: str) -> cns.ProviderGeneration:
                    call_count["n"] += 1
                    if call_count["n"] == 1:
                        # Simulates finish_reason="length"/no tool
                        # call/bad JSON: AzureOpenAIToolProvider.generate
                        # raises ValueError for all of these.
                        raise ValueError("no tool call returned")
                    return cns.ProviderGeneration(
                        response={"ok": True},
                        provider_response_id="r2",
                        provider_model="deepseek-v4-pro",
                        usage={},
                    )

            return _FakeProvider()

        result = cns.generate_with_retries_and_repairs(
            "SYSTEM\ns\n\nUSER\nu",
            provider_factory=fake_provider_factory,
            base_parameters=self._base_params(100),
            repair_max_tokens=500,
            validator=lambda response: None,
            expected_model="deepseek-v4-pro",
        )

        assert result.response == {"ok": True}
        assert recorded_max_tokens == [100, 500]

    def test_401_aborts_immediately_as_fatal_with_no_retries(self) -> None:
        call_count = {"n": 0}

        class _Status401(Exception):
            status_code = 401

        def fake_provider_factory(parameters: cns.GenerationParameters):
            class _FakeProvider:
                def generate(self, prompt: str) -> cns.ProviderGeneration:
                    call_count["n"] += 1
                    raise _Status401("unauthorized")

            return _FakeProvider()

        with pytest.raises(cns.FatalCallError):
            cns.generate_with_retries_and_repairs(
                "SYSTEM\ns\n\nUSER\nu",
                provider_factory=fake_provider_factory,
                base_parameters=self._base_params(),
                repair_max_tokens=500,
                validator=lambda response: None,
                expected_model="deepseek-v4-pro",
            )
        assert call_count["n"] == 1

    def test_404_aborts_as_fatal(self) -> None:
        call_count = {"n": 0}

        class _Status404(Exception):
            status_code = 404

        def fake_provider_factory(parameters: cns.GenerationParameters):
            class _FakeProvider:
                def generate(self, prompt: str) -> cns.ProviderGeneration:
                    call_count["n"] += 1
                    raise _Status404("not found")

            return _FakeProvider()

        with pytest.raises(cns.FatalCallError):
            cns.generate_with_retries_and_repairs(
                "SYSTEM\ns\n\nUSER\nu",
                provider_factory=fake_provider_factory,
                base_parameters=self._base_params(),
                repair_max_tokens=500,
                validator=lambda response: None,
                expected_model="deepseek-v4-pro",
            )
        assert call_count["n"] == 1

    def test_model_mismatch_aborts_as_fatal(self) -> None:
        call_count = {"n": 0}

        def fake_provider_factory(parameters: cns.GenerationParameters):
            class _FakeProvider:
                def generate(self, prompt: str) -> cns.ProviderGeneration:
                    call_count["n"] += 1
                    return cns.ProviderGeneration(
                        response={"ok": True},
                        provider_response_id="r1",
                        provider_model="gpt-4o-2024-08-06",  # wrong model
                        usage={},
                    )

            return _FakeProvider()

        with pytest.raises(cns.ModelMismatchError):
            cns.generate_with_retries_and_repairs(
                "SYSTEM\ns\n\nUSER\nu",
                provider_factory=fake_provider_factory,
                base_parameters=self._base_params(),
                repair_max_tokens=500,
                validator=lambda response: None,
                expected_model="deepseek-v4-pro",
            )
        assert call_count["n"] == 1
        # ModelMismatchError must be a FatalCallError subclass.
        assert issubclass(cns.ModelMismatchError, cns.FatalCallError)


class TestCircuitBreakerB6:
    """B6: ``CircuitBreaker`` trips after too many consecutive failures or
    too great a fraction of overall failures."""

    def test_trips_after_exceeding_consecutive_failure_limit(self) -> None:
        breaker = cns.CircuitBreaker(
            total_units=1000, consecutive_limit=3, fraction_limit=0.9
        )
        for _ in range(3):
            breaker.record(succeeded=False)
            breaker.check()  # exactly at the limit, not yet over it
        breaker.record(succeeded=False)  # 4th consecutive failure: now over
        with pytest.raises(cns.CircuitBreakerTripped):
            breaker.check()

    def test_trips_after_exceeding_failure_fraction_limit(self) -> None:
        # total_units=10, fraction_limit=0.2: the breaker trips once
        # failures exceed 2 (i.e. on the 3rd), even with a success between
        # every failure, which alone would never trip the consecutive path.
        breaker = cns.CircuitBreaker(
            total_units=10, consecutive_limit=1000, fraction_limit=0.2
        )
        for _ in range(2):
            breaker.record(succeeded=False)
            breaker.record(succeeded=True)
            breaker.check()
        breaker.record(succeeded=False)  # 3rd failure: 3/10 = 0.3 > 0.2
        with pytest.raises(cns.CircuitBreakerTripped):
            breaker.check()

    def test_check_does_not_raise_before_tripped(self) -> None:
        breaker = cns.CircuitBreaker(
            total_units=100, consecutive_limit=5, fraction_limit=0.5
        )
        for _ in range(5):
            breaker.record(succeeded=False)
        breaker.check()  # exactly at the limit (5), not strictly over it


class TestPairedExclusionB6:
    """B6: when a batch ultimately fails for one partition, the
    paired-exclusion rule must drop its variants from EVERY partition's
    judged sample for that log+judge, not just the partition that failed."""

    def test_failed_batch_variants_are_excluded_from_every_partition(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(cns, "OUTPUT", tmp_path)
        variant_ids = [f"v{i}" for i in range(1, 26)]  # 25 held-out variants

        variants = pd.DataFrame(
            {
                "variant_id": variant_ids,
                "activity_sequence": [["Create", "Close"] for _ in variant_ids],
            }
        )
        monkeypatch.setattr(cns, "load_variants", lambda log: variants)
        monkeypatch.setattr(
            cns, "sample_heldout_variants", lambda log, **kw: list(variant_ids)
        )
        # One variant per batch -> deterministic, trivial-to-target units.
        monkeypatch.setattr(
            cns, "build_judge_batches", lambda sample, **kw: [[v] for v in sample]
        )

        fixed_candidate_set = {
            "candidate_names": [
                {"name_code": "N1", "business_name": "A", "definition": "d"},
                {"name_code": "unclear", "business_name": "unclear", "definition": "none"},
            ],
            "name_codes": ["N1"],
            "name_code_for_letter": {"A": "N1"},
            "label_map": {v: "0" for v in variant_ids},
            "letter_codes": {"0": "A"},
        }
        monkeypatch.setattr(
            cns,
            "_judge_candidate_set",
            lambda log, partition, dry_run=False: fixed_candidate_set,
        )
        monkeypatch.setattr(
            cns, "run_startup_smoke_probe", lambda keys: {k: cns.MODELS[k] for k in keys}
        )
        monkeypatch.setattr(cns, "require_prespec_confirmation", lambda confirm: "ok")

        names_dir = tmp_path / "names" / "toylog"
        names_dir.mkdir(parents=True)
        (names_dir / "p7.json").write_text("{}", encoding="utf-8")
        (names_dir / "random1.json").write_text("{}", encoding="utf-8")

        # Fail exactly the batch for "v13" on "random1" only (batch_index 12
        # under the one-variant-per-batch mock above).
        failing_batch_id = cns._opaque_code("judge", "toylog", "random1", "12")

        class _FakeJudgeProvider:
            def __init__(self, **kwargs: Any) -> None:
                pass

            def generate(self, prompt: str) -> cns.ProviderGeneration:
                batch_match = re.search(r"Batch ID: (\S+)", prompt)
                assert batch_match is not None
                batch_id = batch_match.group(1)
                if batch_id == failing_batch_id:
                    raise RuntimeError("synthetic batch failure")
                slot_match = re.search(r'"slot_id":\s*"([^"]+)"', prompt)
                assert slot_match is not None
                slot_id = slot_match.group(1)
                return cns.ProviderGeneration(
                    response={
                        "schema_version": "p9",
                        "batch_id": batch_id,
                        "answers": [
                            {"slot_id": slot_id, "chosen_name_code": "N1", "confidence": 5}
                        ],
                    },
                    provider_response_id="r",
                    provider_model="deepseek-v4-pro",
                    usage={},
                )

        monkeypatch.setattr(
            azure_openai_provider, "AzureOpenAIToolProvider", _FakeJudgeProvider
        )

        summary = cns.run_judge_step(
            "deepseek_v4_pro", logs=["toylog"], dry_run=False, confirm_prespec="anything"
        )

        log_summary = summary["logs"]["toylog"]
        assert log_summary["excluded_variants"] == ["v13"]
        assert log_summary["n_sample"] == 25
        assert log_summary["incomplete"] is False  # 1/25 = 4% < 5% threshold

        for partition in ("p7", "random1"):
            frame = pd.read_parquet(
                tmp_path / "judgements" / "deepseek_v4_pro" / "toylog" / f"{partition}.parquet"
            )
            assert len(frame) == 24
            assert "v13" not in set(frame["variant_id"])

        summary_on_disk = json.loads(
            (
                tmp_path / "judgements" / "deepseek_v4_pro" / "toylog" / "_summary.json"
            ).read_text(encoding="utf-8")
        )
        assert summary_on_disk["excluded_variants"] == ["v13"]


class TestResumabilityB6:
    """B6: ``name``/``judge`` steps are cache-backed so re-running after a
    simulated partial failure/crash does not re-call the provider for units
    that already succeeded."""

    def test_execute_cached_request_resumes_without_recalling_provider(
        self, tmp_path
    ) -> None:
        cache_root = tmp_path / "cache"
        call_count = {"n": 0}

        def fake_generate(rendered_prompt: str) -> cns.ProviderGeneration:
            call_count["n"] += 1
            return cns.ProviderGeneration(
                response={"schema_version": "p9", "batch_id": "b1", "answers": []},
                provider_response_id="r1",
                provider_model="deepseek-v4-pro",
                usage={},
            )

        artifact = PromptArtifact(
            prompt_version="p9",
            rendered_prompt="SYSTEM\ns\n\nUSER\nu",
            prompt_sha256="a" * 64,
            schema_sha256="b" * 64,
        )
        model = cns.ModelIdentity(
            provider="azure_openai_entra", model="deepseek-v4-pro", model_version="v1"
        )
        parameters = cns.GenerationParameters(
            temperature=0.0, seed=None, max_output_tokens=100
        )

        first_cache = cns.StructuredResponseCache(cache_root)
        response_1, cache_hit_1 = cns.execute_cached_request(
            cache=first_cache,
            model=model,
            parameters=parameters,
            prompt=artifact,
            input_sha256="c" * 64,
            generate=fake_generate,
            validate=lambda response: None,
        )
        assert cache_hit_1 is False
        assert call_count["n"] == 1

        # Simulate a fresh process after a crash: a brand new cache object
        # rooted at the same directory, with no in-memory state carried over.
        second_cache = cns.StructuredResponseCache(cache_root)
        response_2, cache_hit_2 = cns.execute_cached_request(
            cache=second_cache,
            model=model,
            parameters=parameters,
            prompt=artifact,
            input_sha256="c" * 64,
            generate=fake_generate,
            validate=lambda response: None,
        )
        assert cache_hit_2 is True
        assert call_count["n"] == 1  # the provider was NOT called again
        assert response_2 == response_1

    def test_judge_one_batch_resumes_from_cache_without_recalling_provider(
        self, tmp_path, monkeypatch
    ) -> None:
        call_count = {"n": 0}

        class _FakeJudgeProvider:
            def __init__(self, **kwargs: Any) -> None:
                pass

            def generate(self, prompt: str) -> cns.ProviderGeneration:
                call_count["n"] += 1
                slot_match = re.search(r'"slot_id":\s*"([^"]+)"', prompt)
                batch_match = re.search(r"Batch ID: (\S+)", prompt)
                assert slot_match is not None and batch_match is not None
                return cns.ProviderGeneration(
                    response={
                        "schema_version": "p9",
                        "batch_id": batch_match.group(1),
                        "answers": [
                            {
                                "slot_id": slot_match.group(1),
                                "chosen_name_code": "N1",
                                "confidence": 5,
                            }
                        ],
                    },
                    provider_response_id="r",
                    provider_model="deepseek-v4-pro",
                    usage={},
                )

        monkeypatch.setattr(
            azure_openai_provider, "AzureOpenAIToolProvider", _FakeJudgeProvider
        )
        candidate_set = {
            "candidate_names": [{"name_code": "N1", "business_name": "A", "definition": "d"}],
            "name_codes": ["N1"],
            "name_code_for_letter": {"A": "N1"},
            "label_map": {"v1": "0"},
            "letter_codes": {"0": "A"},
        }
        kwargs = dict(
            judge_key="deepseek_v4_pro",
            endpoint="https://example.test/",
            deployment="dep",
            model_version="v1",
            expected_model="deepseek-v4-pro",
            max_tokens=100,
            repair_max_tokens=200,
            cache_dir=tmp_path / "cache",
            sequence_map={"v1": ["Create", "Close"]},
            candidate_set=candidate_set,
            dry_run=False,
        )

        first = cns.judge_one_batch("toylog", "p7", 0, ["v1"], **kwargs)
        assert call_count["n"] == 1
        assert all(row["cache_hit"] is False for row in first)

        # A brand new call with identical arguments (as a resumed run would
        # make) must be a pure cache hit -- no second provider call.
        second = cns.judge_one_batch("toylog", "p7", 0, ["v1"], **kwargs)
        assert call_count["n"] == 1
        assert all(row["cache_hit"] is True for row in second)

        def _without_cache_hit(rows):
            return [{k: v for k, v in row.items() if k != "cache_hit"} for row in rows]

        assert _without_cache_hit(first) == _without_cache_hit(second)


class TestFrozenJudgePayloadS3:
    """S3: the exact payload hashed as the judge batch's ``input_sha256``
    must stay frozen, so an unintentional shape change in the future fails
    this test loudly."""

    def test_judge_batch_payload_is_frozen(self, tmp_path, monkeypatch) -> None:
        real_canonical_json_hash = cns.canonical_json_hash
        captured: dict[str, Any] = {}

        def spy_canonical_json_hash(value: Any) -> str:
            if isinstance(value, dict) and set(value) == {"candidates", "variants"}:
                captured["payload"] = value
            return real_canonical_json_hash(value)

        monkeypatch.setattr(cns, "canonical_json_hash", spy_canonical_json_hash)

        class _FakeJudgeProvider:
            def __init__(self, **kwargs: Any) -> None:
                pass

            def generate(self, prompt: str) -> cns.ProviderGeneration:
                batch_match = re.search(r"Batch ID: (\S+)", prompt)
                assert batch_match is not None
                return cns.ProviderGeneration(
                    response={
                        "schema_version": "p9",
                        "batch_id": batch_match.group(1),
                        "answers": [
                            {"slot_id": "S1", "chosen_name_code": "N1", "confidence": 5}
                        ],
                    },
                    provider_response_id="r",
                    provider_model="deepseek-v4-pro",
                    usage={},
                )

        monkeypatch.setattr(
            azure_openai_provider, "AzureOpenAIToolProvider", _FakeJudgeProvider
        )

        candidate_set = {
            "candidate_names": [
                {"name_code": "N1", "business_name": "Routine Handling", "definition": "d"},
                {"name_code": "unclear", "business_name": "unclear", "definition": "none"},
            ],
            "name_codes": ["N1"],
            "name_code_for_letter": {"A": "N1"},
            "label_map": {"v1": "0"},
            "letter_codes": {"0": "A"},
        }

        cns.judge_one_batch(
            "toylog",
            "p7",
            0,
            ["v1"],
            judge_key="deepseek_v4_pro",
            endpoint="https://example.test/",
            deployment="dep",
            model_version="v1",
            expected_model="deepseek-v4-pro",
            max_tokens=100,
            repair_max_tokens=200,
            cache_dir=tmp_path / "cache",
            sequence_map={"v1": ["Create", "Approve"]},
            candidate_set=candidate_set,
            dry_run=False,
        )

        payload = captured["payload"]
        assert payload == {
            "candidates": [
                {"name_code": "N1", "business_name": "Routine Handling", "definition": "d"},
                {"name_code": "unclear", "business_name": "unclear", "definition": "none"},
            ],
            "variants": [{"slot_id": "S1", "activity_sequence": ["Create", "Approve"]}],
        }
        # S3: the judge must never see frequency, variant_id, or any hint
        # beyond the activity sequence itself.
        for entry in payload["variants"]:
            assert set(entry) == {"slot_id", "activity_sequence"}

        assert real_canonical_json_hash(payload) == (
            "150322f7ad098812c747506fffd01b11ceee80deffdf0243a122c79673ac0772"
        )
