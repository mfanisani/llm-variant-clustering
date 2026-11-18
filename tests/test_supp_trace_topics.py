import json

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from run_supp_trace_topics import (
    FALLBACK_TOPICS,
    _assignment_frame,
    ari_matrix,
    fit_nmf,
    select_topic_count,
    tfidf_matrix,
    token_count_matrix,
    topic_encoding,
    topic_grid,
    trace_tokens,
)


def _sequences() -> list[list[str]]:
    base = [
        ["submit", "approve", "pay"],
        ["submit", "reject", "submit", "approve", "pay"],
        ["submit", "approve", "approve", "pay"],
        ["create", "check", "close"],
        ["create", "check", "escalate", "close"],
        ["create", "escalate", "check", "close"],
        ["submit", "reject"],
        ["create", "close"],
    ]
    extra = [sequence + ["archive"] for sequence in base]
    more = [["register"] + sequence for sequence in base]
    return base + extra + more


def test_trace_tokens_cover_unigrams_bigrams_and_trigrams() -> None:
    tokens = trace_tokens(["a", "b", "a"])
    assert tokens[:3] == ["act:a", "act:b", "act:a"]
    assert tokens.count(f"2g:{json.dumps(['a', 'b'])}") == 1
    assert tokens.count(f"2g:{json.dumps(['b', 'a'])}") == 1
    assert tokens.count(f"3g:{json.dumps(['a', 'b', 'a'])}") == 1
    assert len(tokens) == 3 + 2 + 1
    assert trace_tokens(["only"]) == ["act:only"]


def test_tokens_are_unambiguous_for_labels_containing_separators() -> None:
    first = trace_tokens(["a->b", "c"])
    second = trace_tokens(["a", "b->c"])
    assert set(first).isdisjoint({token for token in second if token.startswith("2g:")})


def test_tfidf_matches_smoothed_idf_raw_tf_l2_formula() -> None:
    counts, vocabulary = token_count_matrix([["a", "b"], ["a", "a"], ["c"]])
    dense = counts.toarray()
    n_docs = dense.shape[0]
    df = (dense > 0).sum(axis=0)
    idf = np.log((1 + n_docs) / (1 + df)) + 1
    expected = dense * idf
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(tfidf_matrix(counts).toarray(), expected, atol=1e-12)
    assert "act:a" in vocabulary


@pytest.mark.parametrize("config", ["original", "traceprism"])
def test_nmf_is_nonnegative_deterministic_and_reports_relative_error(config) -> None:
    counts, _ = token_count_matrix(_sequences())
    tfidf = tfidf_matrix(counts)
    fit = fit_nmf(tfidf, 5, config)
    again = fit_nmf(tfidf, 5, config)
    assert fit.weights.shape == (tfidf.shape[0], 5)
    assert fit.components.shape == (5, tfidf.shape[1])
    assert (fit.weights >= 0).all() and (fit.components >= 0).all()
    np.testing.assert_array_equal(fit.weights, again.weights)
    residual = tfidf.toarray() - fit.weights @ fit.components
    direct = np.linalg.norm(residual) / np.linalg.norm(tfidf.toarray())
    assert fit.relative_error == pytest.approx(direct, rel=1e-6)
    if config == "traceprism":
        assert fit.iterations == 50


def test_topic_grid_is_truncated_by_rank() -> None:
    assert topic_grid(1000, 1000) == list(range(5, 22))
    assert topic_grid(9, 1000) == [5, 6, 7, 8]
    with pytest.raises(ValueError):
        topic_grid(5, 1000)


def test_knee_rule_picks_elbow_and_breaks_ties_to_smaller_count() -> None:
    counts = [5, 6, 7, 8, 9]
    chosen, status, scores = select_topic_count(counts, [1.0, 0.5, 0.45, 0.42, 0.40])
    assert (chosen, status) == (6, "knee")
    assert scores[1] == pytest.approx(scores.max())

    tied, _, scores = select_topic_count([5, 6, 7, 8, 9], [1.0, 0.5, 0.25, 0.2, 0.0])
    assert scores[1] == scores[2]
    assert tied == 6


def test_knee_rule_falls_back_without_a_decreasing_curve() -> None:
    chosen, status, _ = select_topic_count([5, 6, 7, 8, 9, 10, 11], [0.5] * 7)
    assert (chosen, status) == (FALLBACK_TOPICS, "fallback_no_decreasing_curve")
    linear = [1.0, 0.75, 0.5, 0.25, 0.0]
    chosen, status, _ = select_topic_count([5, 6, 7, 8, 9], linear)
    assert status == "fallback_no_knee" and chosen == 9
    with pytest.raises(ValueError):
        select_topic_count([6, 5], [1.0, 0.5])


def test_topic_encoding_selects_from_the_declared_grid() -> None:
    sequences = _sequences()
    encoding = topic_encoding(sequences, "original")
    assert encoding.rule_status in {"knee", "fallback_no_knee", "fallback_no_decreasing_curve"}
    assert encoding.sweep["selected"].sum() == 1
    assert encoding.n_topics == int(encoding.sweep.loc[encoding.sweep["selected"], "n_topics"].iloc[0])
    assert encoding.weights.shape == (len(sequences), encoding.n_topics)
    repeat = topic_encoding(sequences, "original")
    np.testing.assert_array_equal(encoding.weights, repeat.weights)

    fixed = topic_encoding(sequences, "original", n_topics=5)
    assert fixed.rule_status == "fixed" and fixed.weights.shape[1] == 5


def test_ari_matrix_and_assignment_frame() -> None:
    labelings = [np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0]), np.array([0, 1, 0, 1])]
    matrix = ari_matrix(labelings)
    np.testing.assert_allclose(np.diag(matrix), 1.0)
    np.testing.assert_allclose(matrix, matrix.T)
    assert matrix[0, 1] == pytest.approx(1.0)

    frame = _assignment_frame(["v1", "v2"], np.array([0, 1]), "stt_nmf", 2)
    assert list(frame.columns) == ["variant_id", "cluster_label", "cluster_id"]
    assert frame["cluster_id"].tolist() == ["stt_nmf_k2_c00", "stt_nmf_k2_c01"]
    assert isinstance(frame, pd.DataFrame)
    assert csr_matrix(np.eye(2)).shape == (2, 2)
