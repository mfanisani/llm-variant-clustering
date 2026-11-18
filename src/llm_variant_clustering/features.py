from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class FeatureSet:
    name: str
    matrix: csr_matrix
    variant_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    metric: Literal["euclidean", "cosine"]


def parse_sequences(variants: pd.DataFrame) -> list[list[str]]:
    required = {"variant_id", "activity_sequence", "frequency"}
    missing = sorted(required.difference(variants.columns))
    if missing:
        raise ValueError(f"Variant table is missing columns: {missing}")
    if variants["variant_id"].duplicated().any():
        raise ValueError("variant_id values must be unique")

    sequences = variants["activity_sequence"].map(json.loads).tolist()
    if any(not sequence for sequence in sequences):
        raise ValueError("Activity sequences must not be empty")
    if any(not all(isinstance(activity, str) for activity in sequence) for sequence in sequences):
        raise ValueError("Every activity must be a string")
    return sequences


def activity_profile_features(variants: pd.DataFrame) -> FeatureSet:
    sequences = parse_sequences(variants)
    rows = [
        {f"activity={activity}": sequence.count(activity) for activity in set(sequence)}
        for sequence in sequences
    ]
    return _dict_features(
        "activity_profile",
        rows,
        variants,
        transform="l2",
        metric="cosine",
    )


def dfg_profile_features(variants: pd.DataFrame) -> FeatureSet:
    sequences = parse_sequences(variants)
    rows: list[dict[str, int]] = []
    for sequence in sequences:
        row: dict[str, int] = {}
        for source, target in zip(sequence, sequence[1:], strict=False):
            key = f"dfg={json.dumps([source, target], ensure_ascii=False)}"
            row[key] = row.get(key, 0) + 1
        rows.append(row)
    return _dict_features(
        "dfg_profile",
        rows,
        variants,
        transform="l2",
        metric="cosine",
    )


def ngram_tfidf_features(
    variants: pd.DataFrame,
    *,
    min_n: int = 1,
    max_n: int = 3,
) -> FeatureSet:
    if min_n < 1 or max_n < min_n:
        raise ValueError("Invalid n-gram range")
    sequences = parse_sequences(variants)
    rows: list[dict[str, int]] = []
    for sequence in sequences:
        row: dict[str, int] = {}
        for size in range(min_n, max_n + 1):
            for start in range(len(sequence) - size + 1):
                ngram = sequence[start : start + size]
                key = f"{size}gram={json.dumps(ngram, ensure_ascii=False)}"
                row[key] = row.get(key, 0) + 1
        rows.append(row)
    return _dict_features(
        f"ngram_tfidf_{min_n}_{max_n}",
        rows,
        variants,
        transform="tfidf",
        metric="cosine",
    )


def trivial_sequence_features(variants: pd.DataFrame) -> FeatureSet:
    sequences = parse_sequences(variants)
    rows = []
    for sequence in sequences:
        rows.append(
            {
                "sequence_length": len(sequence),
                "unique_activity_count": len(set(sequence)),
                "repetition_count": len(sequence) - len(set(sequence)),
                f"terminal={sequence[-1]}": 1,
            }
        )
    return _dict_features(
        "trivial_structure",
        rows,
        variants,
        transform="l2",
        metric="cosine",
    )


def activity_context_embeddings(
    variants: pd.DataFrame,
    *,
    context_window: int = 2,
    embedding_dim: int = 16,
) -> dict[str, np.ndarray]:
    """Return one log-derived SPPMI-SVD vector per distinct activity label.

    These vectors use only co-occurrence structure inside the log itself, so
    they provide a knowledge-free counterpart to LLM activity normalization.
    """
    embeddings, vocabulary, _ = _activity_context_space(
        variants,
        context_window=context_window,
        embedding_dim=embedding_dim,
    )
    return {
        activity: embeddings[index] for index, activity in enumerate(vocabulary)
    }


def activity2vec_svd_features(
    variants: pd.DataFrame,
    *,
    context_window: int = 2,
    embedding_dim: int = 16,
) -> FeatureSet:
    sequences = parse_sequences(variants)
    embeddings, vocabulary, dimensions = _activity_context_space(
        variants,
        context_window=context_window,
        embedding_dim=embedding_dim,
    )
    activity_index = {
        activity: index for index, activity in enumerate(vocabulary)
    }
    variant_embeddings = np.vstack(
        [
            np.mean(
                [embeddings[activity_index[activity]] for activity in sequence],
                axis=0,
            )
            for sequence in sequences
        ]
    )
    variant_embeddings = normalize(variant_embeddings, norm="l2")
    return FeatureSet(
        name=f"activity2vec_sppmi_svd_w{context_window}_d{dimensions}",
        matrix=csr_matrix(variant_embeddings),
        variant_ids=tuple(variants["variant_id"].astype(str)),
        feature_names=tuple(f"latent_{index:02d}" for index in range(dimensions)),
        metric="cosine",
    )


def _activity_context_space(
    variants: pd.DataFrame,
    *,
    context_window: int,
    embedding_dim: int,
) -> tuple[np.ndarray, list[str], int]:
    if context_window < 1:
        raise ValueError("context_window must be positive")
    if embedding_dim < 1:
        raise ValueError("embedding_dim must be positive")

    sequences = parse_sequences(variants)
    weights = variants["frequency"].to_numpy(dtype=np.float64)
    vocabulary = sorted({activity for sequence in sequences for activity in sequence})
    activity_index = {
        activity: index for index, activity in enumerate(vocabulary)
    }

    counts: dict[tuple[int, int], float] = {}
    for sequence, weight in zip(sequences, weights, strict=True):
        encoded = [activity_index[activity] for activity in sequence]
        for center, center_activity in enumerate(encoded):
            lower = max(0, center - context_window)
            upper = min(len(encoded), center + context_window + 1)
            for context in range(lower, upper):
                if context == center:
                    continue
                key = (center_activity, encoded[context])
                distance = abs(center - context)
                counts[key] = counts.get(key, 0.0) + weight / distance

    rows = np.fromiter((key[0] for key in counts), dtype=np.int64)
    columns = np.fromiter((key[1] for key in counts), dtype=np.int64)
    data = np.fromiter(counts.values(), dtype=np.float64)
    cooccurrence = coo_matrix(
        (data, (rows, columns)),
        shape=(len(vocabulary), len(vocabulary)),
    ).tocsr()
    total = float(cooccurrence.sum())
    row_sums = np.asarray(cooccurrence.sum(axis=1)).ravel()
    column_sums = np.asarray(cooccurrence.sum(axis=0)).ravel()

    pmi = cooccurrence.tocoo(copy=True)
    pmi.data = np.maximum(
        np.log(
            (pmi.data * total)
            / (row_sums[pmi.row] * column_sums[pmi.col])
        ),
        0.0,
    )
    pmi.eliminate_zeros()

    dimensions = min(embedding_dim, max(1, len(vocabulary) - 1))
    embeddings = TruncatedSVD(
        n_components=dimensions,
        n_iter=10,
        random_state=17,
    ).fit_transform(pmi.tocsr())
    embeddings = normalize(embeddings, norm="l2")
    return embeddings.astype(np.float32), vocabulary, dimensions


def _dict_features(
    name: str,
    rows: list[dict[str, int]],
    variants: pd.DataFrame,
    *,
    transform: Literal["l2", "tfidf"],
    metric: Literal["euclidean", "cosine"],
) -> FeatureSet:
    vectorizer = DictVectorizer(sparse=True, sort=True)
    matrix = vectorizer.fit_transform(rows).astype(np.float64)
    if transform == "l2":
        matrix = normalize(matrix, norm="l2", copy=False)
    elif transform == "tfidf":
        matrix = TfidfTransformer(
            norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True
        ).fit_transform(matrix)

    return FeatureSet(
        name=name,
        matrix=csr_matrix(matrix),
        variant_ids=tuple(variants["variant_id"].astype(str)),
        feature_names=tuple(vectorizer.get_feature_names_out()),
        metric=metric,
    )
