from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from llm_variant_clustering.diagnostic_controls import (
    ACTIVITY_PERMUTATION_NAME,
    RANDOM_EMBEDDING_NAME,
    WITHIN_VARIANT_SHUFFLE_NAME,
    load_activity_embedding_map,
    permute_activity_identity,
    random_gaussian_activity_embeddings,
    shuffle_within_variants,
)
from llm_variant_clustering.semantic_compiler import ordered_semantic_pyramid


def test_activity_embedding_loader_obeys_index_row_order(tmp_path: Path) -> None:
    pd.DataFrame({"activity_label": ["b", "a"]}).to_parquet(
        tmp_path / "activity_embedding_index.parquet",
        index=False,
    )
    matrix = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    np.save(tmp_path / "compact_activity_embeddings.npy", matrix)

    loaded = load_activity_embedding_map(
        tmp_path / "activity_embedding_index.parquet",
        tmp_path / "compact_activity_embeddings.npy",
    )

    assert list(loaded) == ["b", "a"]
    np.testing.assert_array_equal(loaded["b"], matrix[0])
    np.testing.assert_array_equal(loaded["a"], matrix[1])


def test_activity_identity_permutation_is_deterministic_derangement() -> None:
    embeddings = {
        label: np.eye(4, dtype=np.float32)[position]
        for position, label in enumerate(("a", "b", "c", "d"))
    }

    first = permute_activity_identity(embeddings, seed=23)
    second = permute_activity_identity(embeddings, seed=23)

    for label in embeddings:
        np.testing.assert_array_equal(first[label], second[label])
        assert not np.array_equal(first[label], embeddings[label])
    assert {tuple(vector) for vector in first.values()} == {
        tuple(vector) for vector in embeddings.values()
    }


def test_within_variant_shuffle_preserves_multisets_and_uses_local_seeds() -> None:
    sequences = [
        list("abcdefgh"),
        list("abcdefgh"),
        ["a", "a", "b", "c", "c"],
        ["only"],
    ]
    variant_ids = ["v1", "v2", "v3", "v4"]

    first = shuffle_within_variants(sequences, variant_ids, seed=41)
    second = shuffle_within_variants(sequences, variant_ids, seed=41)

    assert first == second
    assert sequences[0] == list("abcdefgh")
    assert [Counter(sequence) for sequence in first] == [
        Counter(sequence) for sequence in sequences
    ]
    assert first[0] != sequences[0]
    assert first[1] != sequences[1]
    assert first[0] != first[1]
    assert first[2] != sequences[2]
    assert first[3] == sequences[3]


def test_random_embeddings_are_dimension_matched_unit_vectors() -> None:
    first = random_gaussian_activity_embeddings(
        ["a", "b", "c"],
        dimension=7,
        seed=5,
    )
    second = random_gaussian_activity_embeddings(
        ["a", "b", "c"],
        dimension=7,
        seed=5,
    )

    matrix = np.vstack(list(first.values()))
    assert matrix.shape == (3, 7)
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(matrix, np.vstack(list(second.values())))


def test_runner_writes_metrics_agreement_assignments_manifest_and_summary(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "frozen"
    output_dir = tmp_path / "output"
    artifact_dir.mkdir()
    sequences = [
        ["a", "b", "c"],
        ["a", "c", "b"],
        ["a", "b", "b"],
        ["b", "a", "c"],
        ["d", "e", "f"],
        ["d", "f", "e"],
        ["d", "e", "e"],
        ["e", "d", "f"],
    ]
    variants = pd.DataFrame(
        {
            "variant_id": [f"v{index}" for index in range(len(sequences))],
            "activity_sequence": [json.dumps(sequence) for sequence in sequences],
            "frequency": [11, 7, 5, 3, 13, 8, 4, 2],
        }
    )
    variants_path = tmp_path / "variants.parquet"
    variants.to_parquet(variants_path, index=False)
    labels = ["c", "a", "f", "b", "e", "d"]
    activity_matrix = np.eye(6, dtype=np.float32)
    activity_map = dict(zip(labels, activity_matrix, strict=True))
    pd.DataFrame({"activity_label": labels}).to_parquet(
        artifact_dir / "activity_embedding_index.parquet",
        index=False,
    )
    np.save(artifact_dir / "compact_activity_embeddings.npy", activity_matrix)

    frozen = ordered_semantic_pyramid(sequences, activity_map)
    artifact_order = [3, 0, 7, 2, 5, 1, 6, 4]
    pd.DataFrame(
        {"variant_id": variants.iloc[artifact_order]["variant_id"]}
    ).to_parquet(artifact_dir / "embedding_index.parquet", index=False)
    np.save(
        artifact_dir / "embeddings_p7_ordered.npy",
        frozen[artifact_order],
    )
    pd.DataFrame(
        {
            "variant_id": variants["variant_id"],
            "cluster_label": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    ).to_parquet(
        artifact_dir / "assignments_p7_ordered_auto.parquet",
        index=False,
    )

    subprocess.run(
        [
            sys.executable,
            str(Path("scripts") / "run_diagnostic_controls.py"),
            "--dataset-dir",
            str(artifact_dir),
            "--variants",
            str(variants_path),
            "--output-dir",
            str(output_dir),
            "--seed",
            "19",
        ],
        check=True,
    )

    metrics = pd.read_csv(output_dir / "fixed_k_metrics.csv")
    assert set(metrics["representation"]) == {
        "baseline_p7_ordered",
        ACTIVITY_PERMUTATION_NAME,
        WITHIN_VARIANT_SHUFFLE_NAME,
        RANDOM_EMBEDDING_NAME,
    }
    agreement = pd.read_csv(
        output_dir / "partition_agreement_against_baseline.csv"
    )
    assert set(agreement["control"]) == {
        ACTIVITY_PERMUTATION_NAME,
        WITHIN_VARIANT_SHUFFLE_NAME,
        RANDOM_EMBEDDING_NAME,
    }
    assignments = pd.read_parquet(
        output_dir / "assignments_within_variant_shuffle_k2.parquet"
    )
    assert assignments["frequency"].tolist() == variants["frequency"].tolist()
    assert (output_dir / "manifest.json").exists()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert "no fixed pass/fail threshold" in summary["interpretation"]
