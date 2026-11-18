from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .features import FeatureSet, parse_sequences


@dataclass(frozen=True)
class EmbeddingProvenance:
    model_id: str
    model_revision: str
    model_license: str
    device: str
    embedding_dimension: int
    normalized: bool
    serialization: str
    input_count: int
    max_input_tokens: int
    model_max_sequence_length: int
    input_corpus_sha256: str
    created_at: str
    duration_seconds: float


def serialize_activity_sequence(sequence: list[str]) -> str:
    if not sequence:
        raise ValueError("Cannot serialize an empty activity sequence")
    if any(not isinstance(activity, str) or not activity.strip() for activity in sequence):
        raise ValueError("Activities must be nonempty strings")
    return " -> ".join(activity.strip() for activity in sequence)


def embed_raw_sequences(
    variants: pd.DataFrame,
    *,
    model_id: str,
    model_revision: str,
    model_license: str,
    cache_dir: Path,
    batch_size: int = 8,
    device: str = "cpu",
) -> tuple[np.ndarray, EmbeddingProvenance]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Install the project 'embeddings' extra to generate embeddings"
        ) from error

    sequences = parse_sequences(variants)
    texts = [serialize_activity_sequence(sequence) for sequence in sequences]
    corpus_hash = hashlib.sha256(
        json.dumps(
            texts,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    model = SentenceTransformer(
        model_id,
        revision=model_revision,
        cache_folder=str(cache_dir),
        device=device,
        trust_remote_code=False,
    )
    tokenized = model.tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    max_input_tokens = max(len(tokens) for tokens in tokenized["input_ids"])
    if max_input_tokens > model.max_seq_length:
        raise ValueError(
            f"Input requires {max_input_tokens} tokens but model limit is "
            f"{model.max_seq_length}; truncation is forbidden"
        )

    started_at = datetime.now(UTC)
    timer = perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    duration = perf_counter() - timer
    embeddings = np.asarray(embeddings, dtype=np.float32)
    validate_embedding_matrix(embeddings, expected_rows=len(variants))

    provenance = EmbeddingProvenance(
        model_id=model_id,
        model_revision=model_revision,
        model_license=model_license,
        device=device,
        embedding_dimension=embeddings.shape[1],
        normalized=True,
        serialization="activity labels joined in order with ' -> '",
        input_count=len(texts),
        max_input_tokens=max_input_tokens,
        model_max_sequence_length=int(model.max_seq_length),
        input_corpus_sha256=corpus_hash,
        created_at=started_at.isoformat(),
        duration_seconds=duration,
    )
    return embeddings, provenance


def validate_embedding_matrix(
    embeddings: np.ndarray, *, expected_rows: int
) -> None:
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"Expected a 2D matrix with {expected_rows} rows, got "
            f"{embeddings.shape}"
        )
    if embeddings.shape[1] < 1 or not np.isfinite(embeddings).all():
        raise ValueError("Embedding matrix contains no features or nonfinite values")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Embedding rows are not L2-normalized")


def save_embedding_artifacts(
    output_dir: Path,
    variants: pd.DataFrame,
    embeddings: np.ndarray,
    provenance: EmbeddingProvenance,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings, allow_pickle=False)
    variants[["variant_id"]].to_parquet(
        output_dir / "embedding_index.parquet", index=False
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(asdict(provenance), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def embedding_feature_set(
    name: str,
    variants: pd.DataFrame,
    embeddings: np.ndarray,
) -> FeatureSet:
    validate_embedding_matrix(embeddings, expected_rows=len(variants))
    from scipy.sparse import csr_matrix

    return FeatureSet(
        name=name,
        matrix=csr_matrix(embeddings),
        variant_ids=tuple(variants["variant_id"].astype(str)),
        feature_names=tuple(
            f"embedding_{index:04d}" for index in range(embeddings.shape[1])
        ),
        metric="cosine",
    )

