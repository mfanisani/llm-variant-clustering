from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator


def validate_activity_semantics(
    response: Mapping[str, Any],
    *,
    schema_path: Path,
    expected_activity_label: str,
) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(response),
        key=lambda error: list(error.path),
    )
    if errors:
        messages = [
            f"{'/'.join(map(str, error.path)) or '<root>'}: {error.message}"
            for error in errors
        ]
        raise ValueError("Schema validation failed: " + "; ".join(messages))
    if not _same_label(response["activity_label"], expected_activity_label):
        raise ValueError(
            "Activity normalizer echoed a different label: "
            f"{response['activity_label']!r} != {expected_activity_label!r}"
        )


def _same_label(echoed: Any, expected: str) -> bool:
    """Did the normalizer answer about the activity we asked about?

    The echo check exists to catch a model that drifts to a different activity,
    which would silently mis-assign a descriptor. It is deliberately insensitive
    to case and surrounding whitespace: some models apply the prompt's
    lowercasing rule to the echo field as well, which is a formatting quirk
    rather than a wrong answer. Being more permissive here cannot change any
    previously accepted response, only admit ones that were rejected on
    formatting alone.
    """
    return isinstance(echoed, str) and echoed.strip().casefold() == expected.strip().casefold()


def canonical_activity_text(activity: Mapping[str, Any]) -> str:
    return (
        f"object: {_normalize(activity['business_object'])}; "
        f"action: {_normalize(activity['action'])}; "
        f"stage: {_normalize(activity['process_stage'])}; "
        f"state: {_normalize(activity['explicit_state'])}; "
        f"role: {_normalize(activity['explicit_role'])}"
    )


def compose_variant_semantics(
    sequence: Sequence[str],
    activity_semantics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not sequence:
        raise ValueError("Cannot compose an empty sequence")
    missing = sorted(set(sequence).difference(activity_semantics))
    if missing:
        raise ValueError(f"Missing activity semantics: {missing}")

    normalized = [activity_semantics[label] for label in sequence]
    tokens = [canonical_activity_text(value) for value in normalized]
    stage_trace = _collapse_consecutive(
        [_normalize(value["process_stage"]) for value in normalized]
    )
    object_trace = _collapse_consecutive(
        [_normalize(value["business_object"]) for value in normalized]
    )
    state_trace = [
        _normalize(value["explicit_state"]) for value in normalized
    ]
    repetitions = [
        {
            "activity_label": label,
            "indices": [
                index for index, value in enumerate(sequence) if value == label
            ],
        }
        for label, count in sorted(Counter(sequence).items())
        if count > 1
    ]
    transitions = [
        {
            "from_index": index,
            "to_index": index + 1,
            "from_token": tokens[index],
            "to_token": tokens[index + 1],
        }
        for index in range(len(sequence) - 1)
    ]
    rejection_resubmissions = []
    for rejection_index, value in enumerate(normalized):
        if _normalize(value["explicit_state"]) != "rejected":
            continue
        rejected_object = _normalize(value["business_object"])
        for later_index in range(rejection_index + 1, len(normalized)):
            later = normalized[later_index]
            if (
                _normalize(later["business_object"]) == rejected_object
                and _normalize(later["explicit_state"]) == "submitted"
            ):
                rejection_resubmissions.append(
                    {
                        "business_object": rejected_object,
                        "rejection_index": rejection_index,
                        "resubmission_index": later_index,
                    }
                )
                break

    return {
        "schema_version": "p7_composed",
        "sequence_length": len(sequence),
        "ordered_activity_labels": list(sequence),
        "ordered_semantic_tokens": tokens,
        "start": {
            "index": 0,
            "activity_label": sequence[0],
            "semantic_token": tokens[0],
        },
        "end": {
            "index": len(sequence) - 1,
            "activity_label": sequence[-1],
            "semantic_token": tokens[-1],
            "explicit_state": state_trace[-1],
        },
        "stage_trace": stage_trace,
        "object_trace": object_trace,
        "state_trace": state_trace,
        "repeated_activities": repetitions,
        "adjacent_transitions": transitions,
        "rejection_resubmissions": rejection_resubmissions,
    }


def ordered_semantic_pyramid(
    sequences: Sequence[Sequence[str]],
    activity_embeddings: Mapping[str, np.ndarray],
    *,
    levels: tuple[int, ...] = (1, 2, 4),
) -> np.ndarray:
    if not levels or any(level < 1 for level in levels):
        raise ValueError("Pyramid levels must be positive")
    if not activity_embeddings:
        raise ValueError("Activity embeddings must not be empty")
    dimensions = {np.asarray(value).shape for value in activity_embeddings.values()}
    if len(dimensions) != 1:
        raise ValueError("Activity embedding dimensions are inconsistent")
    dimension_shape = next(iter(dimensions))
    if len(dimension_shape) != 1 or dimension_shape[0] < 1:
        raise ValueError("Activity embeddings must be one-dimensional vectors")
    dimension = dimension_shape[0]

    rows = []
    for sequence in sequences:
        if not sequence:
            raise ValueError("Cannot embed an empty semantic sequence")
        missing = sorted(set(sequence).difference(activity_embeddings))
        if missing:
            raise ValueError(f"Missing activity embeddings: {missing}")
        matrix = np.vstack([activity_embeddings[label] for label in sequence])
        pooled = []
        for level in levels:
            for indices in np.array_split(np.arange(len(sequence)), level):
                if len(indices):
                    pooled.append(matrix[indices].mean(axis=0))
                else:
                    pooled.append(np.zeros(dimension, dtype=np.float32))
        row = np.concatenate(pooled).astype(np.float32)
        norm = np.linalg.norm(row)
        if norm == 0:
            raise ValueError("Semantic pyramid produced a zero vector")
        rows.append(row / norm)
    return np.vstack(rows)


def mean_semantic_embeddings(
    sequences: Sequence[Sequence[str]],
    activity_embeddings: Mapping[str, np.ndarray],
) -> np.ndarray:
    rows = []
    for sequence in sequences:
        if not sequence:
            raise ValueError("Cannot embed an empty semantic sequence")
        matrix = np.vstack([activity_embeddings[label] for label in sequence])
        row = matrix.mean(axis=0)
        norm = np.linalg.norm(row)
        if norm == 0:
            raise ValueError("Mean semantic embedding produced a zero vector")
        rows.append((row / norm).astype(np.float32))
    return np.vstack(rows)


def _collapse_consecutive(values: Sequence[str]) -> list[str]:
    collapsed = []
    for value in values:
        if not collapsed or collapsed[-1] != value:
            collapsed.append(value)
    return collapsed


def _normalize(value: Any) -> str:
    return " ".join(str(value).casefold().replace("_", " ").split())

