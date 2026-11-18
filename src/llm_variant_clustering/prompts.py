from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from jsonschema import Draft202012Validator

PLACEHOLDER_PATTERN = re.compile(r"<([A-Z][A-Z0-9_]*)>")


@dataclass(frozen=True)
class PromptArtifact:
    prompt_version: str
    rendered_prompt: str
    prompt_sha256: str
    schema_sha256: str


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def render_prompt(
    template_path: Path,
    schema_path: Path,
    values: Mapping[str, str | int],
    *,
    prompt_version: str,
) -> PromptArtifact:
    template = template_path.read_text(encoding="utf-8")
    required = set(PLACEHOLDER_PATTERN.findall(template))
    supplied = set(values)
    missing = sorted(required.difference(supplied))
    extra = sorted(supplied.difference(required))
    if missing or extra:
        raise ValueError(
            f"Prompt values do not match template; missing={missing}, extra={extra}"
        )

    rendered = template
    for key in sorted(required):
        rendered = rendered.replace(f"<{key}>", str(values[key]))
    unresolved = sorted(set(PLACEHOLDER_PATTERN.findall(rendered)))
    if unresolved:
        raise ValueError(f"Unresolved prompt placeholders: {unresolved}")

    schema_text = schema_path.read_text(encoding="utf-8")
    json.loads(schema_text)
    return PromptArtifact(
        prompt_version=prompt_version,
        rendered_prompt=rendered,
        prompt_sha256=_sha256_text(rendered),
        schema_sha256=_sha256_text(schema_text),
    )


def validate_response(
    response: Mapping[str, Any],
    schema_path: Path,
    *,
    expected_variant_id: str | None = None,
    sequence_length: int | None = None,
    activity_sequence: Sequence[str] | None = None,
) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(response), key=lambda error: list(error.path))
    if errors:
        messages = [
            f"{'/'.join(map(str, error.path)) or '<root>'}: {error.message}"
            for error in errors
        ]
        raise ValueError("Schema validation failed: " + "; ".join(messages))

    if (
        expected_variant_id is not None
        and response.get("variant_id") != expected_variant_id
    ):
        raise ValueError(
            "Response variant_id does not match the requested variant_id"
        )

    if activity_sequence is not None:
        sequence_length = len(activity_sequence)
    if sequence_length is not None:
        _validate_activity_indices(response, sequence_length)
    _validate_claim_consistency(response, sequence_length)
    if activity_sequence is not None:
        _validate_structural_evidence(response, activity_sequence)


def validate_taxonomy(
    response: Mapping[str, Any],
    schema_path: Path,
    *,
    expected_mode: Literal["open", "fixed_k"],
    requested_k: int | None = None,
) -> None:
    validate_response(response, schema_path)
    if response["mode"] != expected_mode:
        raise ValueError(
            f"Taxonomy mode {response['mode']!r} does not match {expected_mode!r}"
        )

    clusters = response["clusters"]
    cluster_count = len(clusters)
    if response["proposed_cluster_count"] != cluster_count:
        raise ValueError("proposed_cluster_count does not match clusters length")
    if requested_k is not None and cluster_count != requested_k:
        raise ValueError(
            f"Fixed taxonomy requires {requested_k} clusters, got {cluster_count}"
        )

    cluster_ids = [cluster["cluster_id"] for cluster in clusters]
    if len(cluster_ids) != len(set(cluster_ids)):
        raise ValueError("Taxonomy contains duplicate cluster_id values")
    normalized_names = [
        " ".join(cluster["provisional_name"].casefold().split())
        for cluster in clusters
    ]
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError("Taxonomy contains duplicate provisional names")


def validate_taxonomy_assignments(
    response: Mapping[str, Any],
    schema_path: Path,
    taxonomy: Mapping[str, Any],
    *,
    expected_variant_ids: set[str] | None = None,
) -> None:
    validate_response(response, schema_path)
    cluster_ids = {
        cluster["cluster_id"] for cluster in taxonomy.get("clusters", [])
    }
    if not cluster_ids:
        raise ValueError("Frozen taxonomy does not contain clusters")

    assignments = response["assignments"]
    assigned_variant_ids = [assignment["variant_id"] for assignment in assignments]
    if len(assigned_variant_ids) != len(set(assigned_variant_ids)):
        raise ValueError("Assignments contain duplicate variant_id values")
    if (
        expected_variant_ids is not None
        and set(assigned_variant_ids) != expected_variant_ids
    ):
        missing = sorted(expected_variant_ids.difference(assigned_variant_ids))
        extra = sorted(set(assigned_variant_ids).difference(expected_variant_ids))
        raise ValueError(
            f"Assignments do not match requested variants; missing={missing}, "
            f"extra={extra}"
        )

    for assignment in assignments:
        primary = assignment["primary_cluster_id"]
        alternative = assignment["alternative_cluster_id"]
        if primary not in cluster_ids:
            raise ValueError(f"Unknown primary_cluster_id: {primary}")
        if alternative is not None and alternative not in cluster_ids:
            raise ValueError(f"Unknown alternative_cluster_id: {alternative}")
        if alternative == primary:
            raise ValueError("Alternative cluster must differ from primary cluster")


def canonical_signature_text(
    response: Mapping[str, Any],
    *,
    view: Literal["semantic_core", "semantic_plus_structure"] = "semantic_core",
) -> str:
    def claim_value(key: str, fallback_key: str | None = None) -> str:
        value = response.get(key)
        if value is None and fallback_key is not None:
            value = response.get(fallback_key)
        if isinstance(value, Mapping):
            value = value.get("value")
        return _normalize_semantic_text(str(value or "unknown"))

    stages = response.get("ordered_stages", response.get("main_stages", []))
    stage_names = [
        _normalize_semantic_text(str(stage.get("name", "unknown")))
        if isinstance(stage, Mapping)
        else _normalize_semantic_text(str(stage))
        for stage in stages
    ]
    semantic_core: dict[str, Any] = {
        "primary_object": claim_value("primary_object"),
        "apparent_goal": claim_value("apparent_goal"),
        "ordered_stages": stage_names,
        "explicit_outcome": claim_value("explicit_outcome"),
        "exception_or_correction": claim_value(
            "exception_or_correction", "exception_pattern"
        ),
        "short_description": claim_value("short_description"),
    }
    if view == "semantic_plus_structure":
        if response.get("schema_version") != "p3":
            raise ValueError("semantic_plus_structure view requires a P3 response")
        semantic_core["control_flow_shape"] = _normalize_semantic_text(
            response["control_flow_shape"]
        )
        semantic_core["rework"] = claim_value("rework")
        semantic_core["semantic_backtracking"] = claim_value(
            "semantic_backtracking"
        )
        completion = response["apparent_completion_state"]
        semantic_core["apparent_completion_state"] = _normalize_semantic_text(
            completion["state"]
            if isinstance(completion, Mapping)
            else str(completion)
        )

    return json.dumps(
        semantic_core, ensure_ascii=False, separators=(",", ":"), sort_keys=False
    )


def _validate_activity_indices(value: Any, sequence_length: int) -> None:
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")

    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {
                "evidence_activity_indices",
                "activity_indices",
                "start_indices",
            }:
                _assert_indices_in_bounds(child, sequence_length, key)
            elif key in {"from_activity_index", "to_activity_index"}:
                _assert_indices_in_bounds([child], sequence_length, key)
            else:
                _validate_activity_indices(child, sequence_length)
    elif isinstance(value, list):
        for child in value:
            _validate_activity_indices(child, sequence_length)


def _assert_indices_in_bounds(
    indices: Any, sequence_length: int, field_name: str
) -> None:
    if not isinstance(indices, list):
        indices = [indices]
    invalid = [
        index
        for index in indices
        if not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or index >= sequence_length
    ]
    if invalid:
        raise ValueError(
            f"{field_name} contains indices outside a sequence of length "
            f"{sequence_length}: {invalid}"
        )


def _validate_claim_consistency(
    value: Any, sequence_length: int | None = None
) -> None:
    if isinstance(value, Mapping):
        if {
            "value",
            "inference_type",
            "evidence_activity_indices",
        }.issubset(value):
            inference_type = value["inference_type"]
            claim_value = str(value["value"]).casefold()
            evidence = value["evidence_activity_indices"]
            if inference_type == "unknown":
                if claim_value != "unknown" or evidence:
                    raise ValueError(
                        "Unknown claims require value 'unknown' and no evidence"
                    )
            elif inference_type == "observed_absence":
                if claim_value != "none observed" or not evidence:
                    raise ValueError(
                        "Observed absence requires value 'none observed' and evidence"
                    )
                if (
                    sequence_length is not None
                    and set(evidence) != set(range(sequence_length))
                ):
                    raise ValueError(
                        "Observed absence must cite the complete activity sequence"
                    )
            elif claim_value == "unknown" or not evidence:
                raise ValueError(
                    "Observed or inferred claims require a value and evidence"
                )
        for child in value.values():
            _validate_claim_consistency(child, sequence_length)
    elif isinstance(value, list):
        for child in value:
            _validate_claim_consistency(child, sequence_length)


def _validate_structural_evidence(
    value: Any, activity_sequence: Sequence[str]
) -> None:
    if isinstance(value, Mapping):
        if {"activity", "activity_indices"}.issubset(value):
            expected = value["activity"]
            if any(
                activity_sequence[index] != expected
                for index in value["activity_indices"]
            ):
                raise ValueError("Repeated-activity evidence does not match sequence")
        if {"activities", "start_indices"}.issubset(value):
            activities = list(value["activities"])
            for start in value["start_indices"]:
                if list(activity_sequence[start : start + len(activities)]) != activities:
                    raise ValueError(
                        "Repeated-subsequence evidence does not match sequence"
                    )
        if {"from_activity_index", "to_activity_index"}.issubset(value):
            if value["to_activity_index"] != value["from_activity_index"] + 1:
                raise ValueError("Important transitions must reference adjacent events")
        if "terminal_activity_index" in value:
            if value["terminal_activity_index"] != len(activity_sequence) - 1:
                raise ValueError(
                    "Completion state must reference the final activity index"
                )
        for child in value.values():
            _validate_structural_evidence(child, activity_sequence)
    elif isinstance(value, list):
        for child in value:
            _validate_structural_evidence(child, activity_sequence)


def _normalize_semantic_text(value: str) -> str:
    return " ".join(
        value.casefold().replace("_", " ").replace("-", " ").split()
    )
