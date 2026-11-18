import pandas as pd
import pytest

import run_normalizer_robustness as nr


def test_fields_parses_descriptor_string():
    parsed = nr._fields(
        "object: Purchase Order; action: approve; stage: review; state: approved; role: manager"
    )
    assert list(parsed) == list(nr.FIELDS)
    assert parsed[nr.FIELDS[0]] == "purchase order"
    assert parsed[nr.FIELDS[4]] == "manager"


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [(0.01, 0.2, "positive"), (-0.2, -0.01, "negative"), (-0.1, 0.1, "includes_zero"), (0.0, 0.1, "includes_zero")],
)
def test_classify(lower, upper, expected):
    assert nr._classify(lower, upper) == expected


def _norm_rows(models, failing=None):
    rows = []
    for model in models:
        for dataset, frozen in nr.FROZEN_CLASS.items():
            cls = "includes_zero" if (model, dataset) == failing else frozen
            rows.append(
                {
                    "normalizer": model,
                    "dataset": dataset,
                    "status": "complete",
                    "bonferroni_class": cls,
                    "preserves_frozen_class": cls == frozen,
                }
            )
    return pd.DataFrame(rows)


def _emb_rows(failing=None):
    rows = []
    for dataset, frozen in nr.FROZEN_CLASS.items():
        cls = "includes_zero" if dataset == failing else frozen
        rows.append(
            {"embedder": "e", "dataset": dataset, "bonferroni_class": cls, "preserves_frozen_class": cls == frozen}
        )
    return pd.DataFrame(rows)


def _rule_models():
    return [k for k, spec in nr.NORMALIZERS.items() if spec["decision_rule"]]


def test_decide_final_and_robust_when_all_rule_models_preserve_class():
    decision = nr._decide(_norm_rows(_rule_models()), _emb_rows())
    assert decision["normalizer_verdict"] == "robust_to_normalizer"
    assert decision["normalizer_verdict_status"] == "final"
    assert decision["embedder_verdict"] == "robust_to_embedder"
    assert decision["null_logs_class_changes"] == []


def test_decide_provisional_when_rule_model_missing():
    models = _rule_models()
    decision = nr._decide(_norm_rows(models[:-1]), _emb_rows())
    assert decision["normalizer_verdict_status"] == "provisional"
    assert decision["normalizer_models_missing"] == [models[-1]]


def test_decide_flags_failures_on_positive_logs_only():
    positive = next(d for d, c in nr.FROZEN_CLASS.items() if c == "positive")
    models = _rule_models()
    decision = nr._decide(_norm_rows(models, failing=(models[0], positive)), _emb_rows(failing=positive))
    assert decision["normalizer_verdict"] == "normalizer_dependent"
    assert decision["embedder_verdict"] == "embedder_dependent"


def test_decide_ignores_exploratory_model_failures():
    exploratory = [k for k, spec in nr.NORMALIZERS.items() if not spec["decision_rule"]]
    if not exploratory:
        pytest.skip("no exploratory normalizer configured")
    positive = next(d for d, c in nr.FROZEN_CLASS.items() if c == "positive")
    frame = pd.concat(
        [_norm_rows(_rule_models()), _norm_rows(exploratory, failing=(exploratory[0], positive))]
    )
    assert nr._decide(frame, _emb_rows())["normalizer_verdict"] == "robust_to_normalizer"
