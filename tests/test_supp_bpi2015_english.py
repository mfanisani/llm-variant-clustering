import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from run_supp_bpi2015_english import (
    chunked_distances,
    field_fill_rates,
    modal_english_names,
    q1_reading,
    support_label,
)


def test_modal_english_name_ties_missing_and_shared_text():
    events = pd.DataFrame({
        "concept:name": ["A", "A", "A", "B", "B", "C", "D", "D"],
        "activityNameEN": ["send", "receive", "send", "zeta", "alpha", None, "send", " "],
    })
    mapping = modal_english_names(events).set_index("code")
    assert mapping.loc["A", "english_text"] == "send"
    assert mapping.loc["A", "events_with_modal_name"] == 2
    assert mapping.loc["B", "english_text"] == "alpha"
    assert bool(mapping.loc["B", "tie_at_mode"])
    assert mapping.loc["C", "english_text"] == "C"
    assert bool(mapping.loc["C", "fallback_to_code"])
    assert mapping.loc["D", "english_text"] == "send"
    assert mapping.loc["D", "events_missing_english"] == 1
    assert mapping.loc["A", "codes_sharing_text"] == 2


def test_field_fill_rates_text_and_code_weighting():
    base = {"ambiguity": "none", "explicit_state": "unknown", "process_stage": "other",
            "explicit_role": "none explicit"}
    semantics = {
        "send receipt": {**base, "action": "send", "business_object": "receipt"},
        "x": {**base, "action": "unknown", "business_object": "unknown",
              "explicit_role": "clerk", "ambiguity": "opaque"},
    }
    rates = field_fill_rates(semantics)
    assert rates["action_fill"] == 0.5
    assert rates["explicit_role_fill"] == 0.5
    assert rates["process_stage_fill"] == 1.0 and rates["process_stage_other"] == 1.0
    assert rates["explicit_state_fill"] == 0.0
    weighted = field_fill_rates(semantics, {"send receipt": 3, "x": 1})
    assert weighted["action_fill"] == 0.75
    assert weighted["action_and_object_unknown"] == 0.25


def test_support_and_q1_rules():
    assert support_label(0.01, 0.05, 0.002, 0.06) == "higher_robust"
    assert support_label(0.01, 0.05, -0.001, 0.06) == "higher"
    assert support_label(-0.01, 0.05, -0.02, 0.06) == "not_separated"
    assert support_label(-0.05, -0.01, -0.06, 0.001) == "lower"
    assert q1_reading(0.02, [0.01, 0.02], [0.1, 0.1]) == "consistent_help"
    assert q1_reading(0.005, [0.01, 0.02], [0.1, 0.1]) == "mixed_or_not_separated"
    assert q1_reading(-0.02, [-0.1, -0.1], [-0.01, -0.02]) == "consistent_against"
    assert q1_reading(None, [0.01, 0.02], [0.1, 0.1]).startswith("partial_positive")


def test_chunked_distances_match_full_matrix():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(300, 5))
    np.testing.assert_allclose(chunked_distances(matrix), pairwise_distances(matrix, metric="cosine"),
                               rtol=0, atol=1e-12)
