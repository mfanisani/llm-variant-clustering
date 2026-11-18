from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd
import pytest

import run_supp_semantic_tagger as tagger
from llm_variant_clustering.semantic_compiler import (
    canonical_activity_text,
    validate_activity_semantics,
)

ROOT = Path(__file__).parents[1]
SCHEMA = ROOT / "configs/frozen/p7_activity_gpt41_v1/schema.json"


def _authors_filter_col(column: pd.Series) -> pd.Series:
    """preprocessing.preprocessor.filter_col of the tagger repository."""
    camel_1 = re.compile("(.)([A-Z][a-z]+)")
    camel_2 = re.compile("([a-z0-9])([A-Z])")
    non_alnum = re.compile("[^a-zA-Z]")
    return (
        column.astype(str)
        .str.replace(camel_1, r"\1 \2", regex=True)
        .str.replace(camel_2, r"\1 \2", regex=True)
        .str.lower()
        .str.replace(non_alnum, " ", regex=True)
        .str.replace(" +", " ", regex=True)
        .str.strip()
        .str.replace(r"\d+", "", regex=True)
        .str.strip()
    )


LABELS = [
    "Register_Request",
    "inspect request  by clerk",
    "A_Create Application",
    "Declaration FINAL_APPROVED by SUPERVISOR",
    "SRM: Created",
    "Send for Credit Collection",
    "W_Validate application",
    "O_Sent (mail and online)",
    "ER Sepsis Triage",
    "Add penalty",
    "123",
    "changePriceInvoice2",
]


def test_clean_label_equals_authors_filter_col():
    expected = _authors_filter_col(pd.Series(LABELS)).tolist()
    assert [tagger.clean_label(x) for x in LABELS] == expected
    assert tagger.clean_label("Register_Request") == "register request"
    assert tagger.clean_label("123") == ""


def test_find_role_matches_tag_runs():
    split = ["categorize", "and", "assign", "request", "by", "clerk"]
    tags = ["A", "X", "A", "BO", "X", "ACTOR"]
    assert tagger.find_role(split, tags, "A") == ["categorize", "assign"]
    for role in ("A", "BO", "ACTOR", "X"):
        assert tagger.find_role(split, tags, role) == [
            text for tag, text in tagger.tag_runs(split, tags) if tag == role
        ]
    assert tagger.find_role(["final", "approved"], ["ASTATE", "ASTATE"], "ASTATE") == ["final approved"]


def test_check_tok_is_identity_on_lowercased_tokens():
    split = ["bo", "request"]
    assert tagger.check_tok_for_object_type(split, ["BO", "A"]) == ["BO", "A"]
    with pytest.raises(AssertionError):
        tagger.check_tok_for_object_type(["BO", "x"], ["BO", "X"])


def _valid(semantics):
    validate_activity_semantics(
        semantics, schema_path=SCHEMA, expected_activity_label=semantics["activity_label"]
    )
    return semantics


def test_mapping_state_stage_role():
    label = "Declaration FINAL_APPROVED by SUPERVISOR"
    out = _valid(tagger.map_to_p7(
        label,
        ["declaration", "final", "approved", "by", "supervisor"],
        ["BO", "ASTATE", "ASTATE", "X", "ACTOR"],
    ))
    assert out["business_object"] == "declaration"
    assert out["action"] == "unknown"
    assert out["explicit_state"] == "final_approved"
    assert out["process_stage"] == "decision"
    assert out["explicit_role"] == "supervisor"
    assert out["ambiguity"] == tagger.AMBIGUITY
    assert canonical_activity_text(out) == (
        "object: declaration; action: unknown; stage: decision; "
        "state: final approved; role: supervisor"
    )


def test_mapping_missing_role_other_state_and_canceled():
    out = _valid(tagger.map_to_p7(
        "x", ["declaration", "rejected", "by", "missing"], ["BO", "BOSTATE", "X", "ACTOR"]
    ))
    assert out["explicit_role"] == "none explicit"
    assert out["explicit_state"] == "rejected"
    out = _valid(tagger.map_to_p7("x", ["order", "pending"], ["BO", "ASTATE"]))
    assert (out["explicit_state"], out["process_stage"]) == ("other", "unknown")
    out = _valid(tagger.map_to_p7("x", ["order", "pending", "canceled"], ["BO", "ASTATE", "BOSTATE"]))
    assert (out["explicit_state"], out["process_stage"]) == ("cancelled", "closure")


def test_mapping_actor_then_recipient_and_ignored_tags():
    out = _valid(tagger.map_to_p7(
        "x", ["inform", "customer", "by", "clerk", "now"], ["A", "REC", "X", "ACTOR", "[PAD]"]
    ))
    assert out["explicit_role"] == "clerk, customer"
    assert out["action"] == "inform"
    assert out["business_object"] == "unknown"


def test_mapping_empty_label_is_valid_and_unknown():
    out = _valid(tagger.map_to_p7("123", [], []))
    assert (out["action"], out["business_object"], out["explicit_state"]) == ("unknown",) * 3
    assert out["explicit_role"] == "none explicit"
    with pytest.raises(ValueError):
        tagger.map_to_p7("x", ["a"], [])


def test_state_lexicon_covers_schema_enum():
    import json

    enum = json.loads(SCHEMA.read_text(encoding="utf-8"))["properties"]["explicit_state"]["enum"]
    assert set(tagger.STATE_LEXICON.values()) == set(enum) - {"other", "unknown"}
    assert set(tagger.STAGE_BY_STATE) == set(enum) - {"other", "unknown"}


def test_coverage_problems():
    assignments = pd.DataFrame({"variant_id": ["a", "b", "c", "d"], "cluster_id": ["c0", "c0", "c1", "c1"]})
    split = pd.DataFrame({"variant_id": ["a", "b", "c", "d"], "split": ["train", "test", "train", "train"]})
    assert tagger.coverage_problems(assignments, split) == ["c1 has no test variant"]
    split.loc[3, "split"] = "test"
    assert tagger.coverage_problems(assignments, split) == []
    assert tagger.coverage_problems(assignments, split.iloc[:3]) == ["variants missing from the frozen split"]


def test_token_jaccard():
    assert tagger.token_jaccard("final approve", "approve") == 0.5
    assert tagger.token_jaccard("", "") == 1.0


def test_script_parses_as_python_39():
    source = Path(tagger.__file__).read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 9))
