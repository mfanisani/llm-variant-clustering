"""Cluster naming and semantic separability (paper 1 post-hoc evaluation).

Pre-specification: ``evaluation/cluster_naming_v1/PRESPEC.md``.

This adds a post-hoc business-naming step on top of the already-frozen P7
clustering and tests whether an LLM judge can recover cluster membership of
held-out variants from the names better than it can for baseline partitions
at the same K. ``evaluation/full_benchmark/``,
``evaluation/representation_sources/``, ``evaluation/statistical_
uncertainty_v2/``, ``configs/frozen/`` and ``datasets/`` are read only. All
outputs go to ``evaluation/cluster_naming_v1/``.

Subcommands: ``profile``, ``name``, ``judge --judge {deepseek_v4_pro,gpt4o}``,
``analyze``. Steps are idempotent and cache backed.
"""

from __future__ import annotations

import os

import argparse
import hashlib
import json
import math
import re
import string
import sys
import threading
import time
import traceback
import zlib
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, datetime
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from llm_variant_clustering.llm_cache import (
    GenerationParameters,
    ModelIdentity,
    ProviderGeneration,
    StructuredResponseCache,
    canonical_json_hash,
    execute_cached_request,
)
from llm_variant_clustering.partition_similarity import partition_similarity
from llm_variant_clustering.prompts import render_prompt, validate_response

ROOT = Path(__file__).parents[1]
DATASETS = ROOT / "datasets" / "processed"
REPR_SOURCES = ROOT / "evaluation" / "representation_sources"
FULL_BENCHMARK = ROOT / "evaluation" / "full_benchmark"
OUTPUT = ROOT / "evaluation" / "cluster_naming_v1"

NAMING_PROMPT = ROOT / "configs/prompts/p8_business_cluster_names.txt"
NAMING_SCHEMA = ROOT / "configs/schemas/business_cluster_names.schema.json"
JUDGE_PROMPT = ROOT / "configs/prompts/p9_judge_variant_match.txt"
JUDGE_SCHEMA = ROOT / "configs/schemas/judge_variant_match.schema.json"

SEED = 20240601
RANDOM_SEEDS = {
    "random1": 20240601,
    "random2": 20240602,
    "random3": 20240603,
}
NAMED_PARTITIONS = ("p7", "raw", "a2v", "tfidf", "random1", "random2", "random3")
API_VERSION = "2025-01-01-preview"
PRIMARY_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")

# dataset -> folder, matched K, role, frozen-partition suffix, whether the
# variant set is restricted to evaluation/full_benchmark/<log>/
# embedding_index.parquet (excludes calibration variants), and the
# activity2vec rank to use (the one that is not d16; sepsis has only d16).
LOG_SPECS: dict[str, dict[str, Any]] = {
    "bpi2020": {
        "folder": "bpi2020_international_declarations",
        "k": 9,
        "role": "primary",
        "frozen_suffix": "dynamic",
        "restrict": False,
        "a2v_rank": 34,
    },
    "bpi2017": {
        "folder": "bpi2017",
        "k": 2,
        "role": "primary",
        "frozen_suffix": "dynamic",
        "restrict": True,
        "a2v_rank": 23,
    },
    "bpi2019": {
        "folder": "bpi2019",
        "k": 6,
        "role": "primary",
        "frozen_suffix": "dynamic",
        "restrict": True,
        "a2v_rank": 42,
    },
    "sepsis": {
        "folder": "sepsis",
        "k": 8,
        "role": "stress_test",
        "frozen_suffix": "k8",
        "restrict": False,
        "a2v_rank": 16,
    },
    "rtfm": {
        "folder": "rtfm",
        "k": 2,
        "role": "stress_test",
        "frozen_suffix": "dynamic",
        "restrict": False,
        "a2v_rank": 11,
    },
}

MODELS = {
    "namer_gpt41": {
        "endpoint": PRIMARY_ENDPOINT,
        "deployment": "gpt-4.1-standard",
        "model_version": "2025-04-14",
        "expected_model": "gpt-4.1-2025-04-14",
        "max_tokens": 4000,
        "repair_max_tokens": 8000,
        "workers": 2,
    },
    "deepseek_v4_pro": {
        "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT_SECONDARY") or os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        "deployment": "deepseek-v4-pro",
        "model_version": "2026-04-23",
        "expected_model": "deepseek-v4-pro",
        "max_tokens": 2000,
        "repair_max_tokens": 4000,
        "workers": 4,
    },
    "gpt4o": {
        "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT_SECONDARY") or os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        "deployment": "gpt-4o-2024-08-06-global",
        "model_version": "2024-08-06",
        "expected_model": "gpt-4o-2024-08-06",
        "max_tokens": 1000,
        "repair_max_tokens": 2000,
        "workers": 2,
    },
}

# Fallback deployments (PRESPEC Amendment 1, S6): only used if a primary
# deployment 404s or is otherwise unavailable at startup smoke-test time, and
# only after the amendment is recorded. Never swaps the judge vendor.
FALLBACK_MODELS = {
    "namer_gpt41": {
        "endpoint": PRIMARY_ENDPOINT,
        "deployment": "gpt-4.1",
        "model_version": "2025-04-14",
        "expected_model": "gpt-4.1-2025-04-14",
        "max_tokens": 4000,
        "repair_max_tokens": 8000,
        "workers": 1,
    },
}

# Exception-handling / resilience knobs (Section 13, B6/B7).
MAX_PROVIDER_ATTEMPTS = 6
PROVIDER_BACKOFF_SECONDS = 2.0
PROVIDER_BACKOFF_CAP_SECONDS = 60.0
PROVIDER_JITTER_SECONDS = 1.0
MAX_VALIDATION_ATTEMPTS = 3
CIRCUIT_BREAKER_CONSECUTIVE_FAILURES = 10
CIRCUIT_BREAKER_FAILURE_FRACTION = 0.05
INCOMPLETE_DATA_THRESHOLD = 0.05
BOOTSTRAP_RESAMPLES = 2000
DROPPED_RESAMPLE_WARN_FRACTION = 0.01
# Amendment 2: >5% of resamples dropped for a log x comparison is flagged
# "low-information" in the tables and decision.json. It still counts; this
# is a visibility flag, not an exclusion rule.
LOW_INFORMATION_DROPPED_FRACTION = 0.05

BANNED_WORD_PATTERN = re.compile(
    r"\b(cluster\s*\d+|miscellaneous|standard\s+cases|success(ful(ly)?)?|"
    r"failure|compliance|cost|speed|ownership)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Fatal-vs-per-unit failure classification (B6)
# ---------------------------------------------------------------------------


class FatalCallError(RuntimeError):
    """A non-retryable failure that must abort the entire run immediately.

    Raised for authentication failures, missing deployments, content-filter
    rejections, and served-model mismatches. Never caught by the per-unit
    isolation wrapper; it is allowed to propagate and stop the step.
    """


class ModelMismatchError(FatalCallError):
    """The service returned a model string that does not match what was requested."""


_FATAL_HTTP_STATUSES = {401, 403, 404}


def _is_content_filter_error(error: Exception) -> bool:
    message = str(getattr(error, "message", "") or error).lower()
    body = getattr(error, "body", None)
    if body:
        try:
            message += " " + json.dumps(body, default=str).lower()
        except TypeError:
            message += " " + str(body).lower()
    return "content_filter" in message or "content management policy" in message


def classify_status_error_as_fatal(error: Exception) -> bool:
    """True if an HTTP-status failure should abort the whole run.

    Auth (401/403), not-found (404, e.g. a deployment that does not exist)
    and content-filter rejections (400) are never worth retrying and are not
    isolated per unit: they indicate the run itself is misconfigured or that
    a unit is unsafe to keep attempting under different inputs.
    """
    status = getattr(error, "status_code", None)
    if status in _FATAL_HTTP_STATUSES:
        return True
    if status == 400 and _is_content_filter_error(error):
        return True
    return False


def _opaque_code(*parts: str, length: int = 10) -> str:
    """A deterministic, non-reversible code that never reveals log/partition identity.

    Used anywhere a log or partition identifier would otherwise be embedded
    directly in LLM-visible prompt text (Section 13, B3). The real mapping is
    only ever held in-process / in our own output files, never sent to a model.
    """
    digest = hashlib.sha256(":".join((str(SEED), *parts)).encode("utf-8")).hexdigest()
    return digest[:length]


def _prefix_evidence_ids(profile: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    """Rewrite every evidence ID in ``profile`` to ``f"{prefix}-{id}"`` (S1).

    Evidence IDs (``A1``, ``T1``, ``V1``, ...) are generated independently
    per cluster by ``compute_cluster_profiles`` and are *not* globally
    unique: two different clusters in the same partition can legitimately
    both have an "A1". Prefixing with the cluster's own letter code before
    the profile is sent to the namer makes IDs globally unique, so the
    "cites evidence outside its own profile" validator check is watertight
    rather than merely checking "exists in *a* profile somewhere".
    """

    def rewrite_id(value: str) -> str:
        return f"{prefix}-{value}"

    out = dict(profile)
    for table_key in (
        "activity_prevalence",
        "top_directly_follows_transitions",
        "distinctive_activities",
        "distinctive_transitions",
        "top_variants",
    ):
        out[table_key] = [
            {**row, "id": rewrite_id(row["id"])} for row in profile.get(table_key, [])
        ]
    out["evidence_ids"] = [rewrite_id(e) for e in profile.get("evidence_ids", [])]
    return out


# ---------------------------------------------------------------------------
# Retry + validation-repair wrapper (Section 13, B5/B6)
# ---------------------------------------------------------------------------


def _repair_prompt(rendered_prompt: str, error: Exception) -> str:
    return (
        rendered_prompt
        + "\n\nVALIDATION CORRECTION\n"
        + "The previous tool arguments failed local validation. Produce a "
        + "fresh tool call for the same request. Correct every listed "
        + "problem without dropping required fields or inventing data:\n"
        + str(error)
    )


def generate_with_retries_and_repairs(
    rendered_prompt: str,
    *,
    provider_factory: Callable[[GenerationParameters], Any],
    base_parameters: GenerationParameters,
    repair_max_tokens: int,
    validator: Callable[[Mapping[str, Any]], None],
    expected_model: str,
    max_validation_attempts: int = MAX_VALIDATION_ATTEMPTS,
) -> ProviderGeneration:
    """Repair loop with ``provider.generate`` *inside* the per-attempt try.

    Unlike ``run_azure_prompt_pilot.generate_with_validation_repairs`` (which
    only repairs local schema-validation failures), this also repairs
    provider-level failures that manifest as ``ValueError`` from
    ``AzureOpenAIToolProvider.generate`` itself: ``finish_reason="length"``
    with no tool call, a missing tool call, or malformed JSON arguments
    (``json.JSONDecodeError`` is a ``ValueError`` subclass). On the first
    repair attempt ``max_tokens`` is raised to ``repair_max_tokens``.

    Raises ``ModelMismatchError`` (a ``FatalCallError``) if the served model
    does not start with ``expected_model``, and re-raises any HTTP-status
    error classified fatal by ``classify_status_error_as_fatal`` as a
    ``FatalCallError``. Both abort the whole run; callers must not catch
    ``FatalCallError`` as a per-unit failure.
    """
    validation_errors: list[str] = []
    total_provider_attempts = 0
    total_retry_after_waits = 0
    current_prompt = rendered_prompt
    generation: ProviderGeneration | None = None
    for attempt in range(1, max_validation_attempts + 1):
        parameters = (
            base_parameters
            if attempt == 1
            else replace(base_parameters, max_output_tokens=repair_max_tokens)
        )
        provider = provider_factory(parameters)
        try:
            generation = provider.generate(current_prompt)
        except ValueError as error:
            # Repairable: no tool call / finish_reason=length / bad JSON.
            validation_errors.append(f"provider: {error}")
            if attempt == max_validation_attempts:
                generation = None
                break
            current_prompt = _repair_prompt(rendered_prompt, error)
            continue
        except FatalCallError:
            raise
        except Exception as error:  # noqa: BLE001 - classify provider/HTTP errors
            if classify_status_error_as_fatal(error):
                raise FatalCallError(str(error)) from error
            raise

        total_provider_attempts += generation.attempt_count
        total_retry_after_waits += int(
            generation.request_metadata.get("retry_after_waits", 0) or 0
        )
        if expected_model and not str(generation.provider_model or "").startswith(
            expected_model
        ):
            raise ModelMismatchError(
                f"expected a model starting with {expected_model!r}, got "
                f"{generation.provider_model!r}"
            )
        try:
            validator(generation.response)
        except ValueError as error:
            validation_errors.append(f"validation: {error}")
            if attempt == max_validation_attempts:
                break
            current_prompt = _repair_prompt(rendered_prompt, error)
            continue
        else:
            return replace(
                generation,
                attempt_count=total_provider_attempts,
                request_metadata={
                    **generation.request_metadata,
                    "validation_attempt_count": attempt,
                    "prior_validation_errors": validation_errors,
                    "retry_after_waits": total_retry_after_waits,
                },
            )
    if generation is None:
        raise RuntimeError(
            f"All {max_validation_attempts} attempts failed before any usable "
            f"response: {validation_errors}"
        )
    return replace(
        generation,
        attempt_count=total_provider_attempts,
        request_metadata={
            **generation.request_metadata,
            "validation_attempt_count": max_validation_attempts,
            "prior_validation_errors": validation_errors,
            "retry_after_waits": total_retry_after_waits,
        },
    )


class CircuitBreakerTripped(RuntimeError):
    """A step aborted because too many units failed in a row or overall."""


class CircuitBreaker:
    """Abort a step after too many consecutive or too great a fraction of failures.

    Not thread-safe internals are guarded by a lock since steps run units
    concurrently via a thread pool.
    """

    def __init__(
        self,
        *,
        total_units: int,
        consecutive_limit: int = CIRCUIT_BREAKER_CONSECUTIVE_FAILURES,
        fraction_limit: float = CIRCUIT_BREAKER_FAILURE_FRACTION,
    ) -> None:
        self.total_units = max(total_units, 1)
        self.consecutive_limit = consecutive_limit
        self.fraction_limit = fraction_limit
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._total_failures = 0
        self._completed = 0
        self.tripped = False

    def record(self, *, succeeded: bool) -> None:
        with self._lock:
            self._completed += 1
            if succeeded:
                self._consecutive_failures = 0
                return
            self._consecutive_failures += 1
            self._total_failures += 1
            if self._consecutive_failures > self.consecutive_limit:
                self.tripped = True
            if self._total_failures / self.total_units > self.fraction_limit:
                self.tripped = True

    def check(self) -> None:
        if self.tripped:
            raise CircuitBreakerTripped(
                f"Circuit breaker tripped: {self._total_failures} failed of "
                f"{self._completed} completed (limit: >"
                f"{self.consecutive_limit} consecutive or >"
                f"{self.fraction_limit:.0%} of {self.total_units})"
            )


def write_failure_record(
    path: Path, *, unit: Mapping[str, Any], error: Exception
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "unit": dict(unit),
        "error_type": type(error).__name__,
        "error": str(error),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Startup smoke probe (B5): confirm each deployment is reachable and serves
# the expected model BEFORE spending the real per-unit call budget.
# ---------------------------------------------------------------------------

_SMOKE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


def smoke_test_model(model_key: str, model_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """One minimal structured call to confirm a deployment answers and that
    ``response.model`` matches what we expect (B5). Raises ``FatalCallError``
    on any failure, including a model mismatch, so the caller can abort
    before any production call is made.
    """
    from llm_variant_clustering.azure_openai_provider import AzureOpenAIToolProvider

    # 16 tokens was enough for the namer (gpt-4.1) but silently starved
    # DeepSeek's reasoning tokens before it ever reached the forced tool
    # call, making the smoke probe fail with "0 tool calls" for a perfectly
    # healthy deployment. 1024 is still a negligible smoke-test cost but
    # gives any reasoning model enough headroom to actually answer.
    parameters = GenerationParameters(temperature=0.0, seed=None, max_output_tokens=1024)
    try:
        provider = AzureOpenAIToolProvider(
            endpoint=model_cfg["endpoint"],
            deployment=model_cfg["deployment"],
            api_version=API_VERSION,
            parameters=parameters,
            schema=_SMOKE_SCHEMA,
            schema_name="smoke_probe",
            max_attempts=MAX_PROVIDER_ATTEMPTS,
            backoff_seconds=PROVIDER_BACKOFF_SECONDS,
            max_backoff_seconds=PROVIDER_BACKOFF_CAP_SECONDS,
            jitter_seconds=PROVIDER_JITTER_SECONDS,
            honor_retry_after=True,
        )
        generation = provider.generate(
            "SYSTEM\nYou are a smoke-test probe.\n\nUSER\n"
            "Reply with the single boolean field `ok` set to true."
        )
    except FatalCallError:
        raise
    except Exception as error:  # noqa: BLE001 - any smoke failure is fatal
        raise FatalCallError(
            f"{model_key} smoke probe failed against {model_cfg['endpoint']} "
            f"{model_cfg['deployment']!r}: {type(error).__name__}: {error}"
        ) from error

    expected = model_cfg["expected_model"]
    provider_model = generation.provider_model
    if not str(provider_model or "").startswith(expected):
        raise ModelMismatchError(
            f"{model_key} smoke probe: expected a model starting with "
            f"{expected!r}, got {provider_model!r} from "
            f"{model_cfg['endpoint']} {model_cfg['deployment']!r}"
        )
    return {
        "model_key": model_key,
        "endpoint": model_cfg["endpoint"],
        "deployment": model_cfg["deployment"],
        "expected_model": expected,
        "provider_model": provider_model,
        "ok": True,
    }


def _is_deployment_not_found_error(error: Exception | None) -> bool:
    return getattr(error, "status_code", None) == 404


def resolve_model_config(model_key: str) -> tuple[dict[str, Any], bool]:
    """S6: fallback-deployment policy.

    Smoke-tests the primary deployment; only on a 404 ("deployment not
    found") AND only if ``FALLBACK_MODELS`` declares an alternative for this
    exact ``model_key`` does it retry against the fallback. There is no
    fallback entry for either judge model, so this can never silently swap
    judge vendors -- it only ever applies to the namer. Returns
    ``(resolved_model_cfg, used_fallback)``.
    """
    primary_cfg = MODELS[model_key]
    try:
        result = smoke_test_model(model_key, primary_cfg)
    except FatalCallError as primary_error:
        fallback_cfg = FALLBACK_MODELS.get(model_key)
        if fallback_cfg is None or not _is_deployment_not_found_error(
            primary_error.__cause__
        ):
            raise
        print(
            f"[smoke] {model_key}: primary deployment not found (404), "
            f"trying fallback {fallback_cfg['endpoint']} "
            f"{fallback_cfg['deployment']!r}",
            flush=True,
        )
        result = smoke_test_model(model_key, fallback_cfg)
        print(
            f"[smoke] {model_key}: OK via fallback, served model="
            f"{result['provider_model']!r}",
            flush=True,
        )
        return {**fallback_cfg, "used_fallback": True}, True
    print(
        f"[smoke] {model_key}: OK, served model={result['provider_model']!r}",
        flush=True,
    )
    return {**primary_cfg, "used_fallback": False}, False


def run_startup_smoke_probe(model_keys: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Resolve (and smoke-test) every requested model, in order (B5/S6).

    Lets the first fatal failure propagate immediately (no point probing
    further deployments once one has failed) so no production call for
    ANY model is made this run. Returns the resolved model config for each
    key -- the primary unless S6's fallback policy kicked in -- so callers
    use the SAME deployment for real calls that was just smoke-tested.
    """
    resolved: dict[str, dict[str, Any]] = {}
    check_results: dict[str, dict[str, Any]] = {}
    for model_key in model_keys:
        model_cfg, used_fallback = resolve_model_config(model_key)
        resolved[model_key] = model_cfg
        check_results[model_key] = {
            "model_key": model_key,
            "endpoint": model_cfg["endpoint"],
            "deployment": model_cfg["deployment"],
            "expected_model": model_cfg["expected_model"],
            "used_fallback": used_fallback,
            "ok": True,
        }
    record_model_check(check_results)
    return resolved


_MODEL_CHECK_PATH_NAME = "model_check.json"


def record_model_check(results: Mapping[str, dict[str, Any]]) -> None:
    """Merge smoke-probe results into ``OUTPUT/model_check.json`` (for the manifest)."""
    path = OUTPUT / _MODEL_CHECK_PATH_NAME
    existing: dict[str, Any] = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.update(
        {
            key: {**value, "checked_at": datetime.now(UTC).isoformat()}
            for key, value in results.items()
        }
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_variants(log: str) -> pd.DataFrame:
    """Variants for ``log``, restricted to the matched-K embedding index."""
    spec = LOG_SPECS[log]
    variants = pd.read_parquet(DATASETS / spec["folder"] / "variants.parquet")
    variants = variants.copy()
    variants["activity_sequence"] = variants["activity_sequence"].map(json.loads)
    if spec["restrict"]:
        index = pd.read_parquet(FULL_BENCHMARK / log / "embedding_index.parquet")
        variants = variants.merge(index[["variant_id"]], on="variant_id", how="inner")
    return variants.reset_index(drop=True)


def build_sepsis_split(*, write: bool = True) -> pd.DataFrame:
    """A 20% uniform random held-out sample of Sepsis variants, seed 20240601."""
    variants = load_variants("sepsis")
    ids = np.array(sorted(variants["variant_id"]))
    rng = np.random.default_rng(SEED)
    n_test = round(0.2 * len(ids))
    test_positions = rng.choice(len(ids), size=n_test, replace=False)
    test_ids = set(ids[test_positions])
    split = pd.DataFrame(
        {
            "variant_id": ids,
            "split": ["test" if v in test_ids else "train" for v in ids],
        }
    )
    split = split.merge(
        variants[["variant_id", "frequency"]], on="variant_id", how="left"
    ).rename(columns={"frequency": "cases"})
    if write:
        out = OUTPUT / "splits" / "sepsis_variant_split.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        split.to_parquet(out, index=False)
    return split


def load_split(log: str) -> pd.DataFrame:
    frozen_path = (
        FULL_BENCHMARK / log / "process_model_metrics_dynamic_heldout_variant_split.parquet"
    )
    if frozen_path.exists():
        return pd.read_parquet(frozen_path)[["variant_id", "split", "cases"]]
    if log == "sepsis":
        own_path = OUTPUT / "splits" / "sepsis_variant_split.parquet"
        if own_path.exists():
            return pd.read_parquet(own_path)[["variant_id", "split", "cases"]]
        return build_sepsis_split()
    raise FileNotFoundError(f"No held-out split available for {log!r}")


def _partition_source_file(log: str, partition: str) -> Path:
    spec = LOG_SPECS[log]
    k = spec["k"]
    if partition == "p7":
        return REPR_SOURCES / log / f"assignments_p7_llm_semantic_bge_k{k}.parquet"
    if partition == "raw":
        return REPR_SOURCES / log / f"assignments_raw_activity_label_bge_k{k}.parquet"
    if partition == "a2v":
        rank = spec["a2v_rank"]
        return (
            REPR_SOURCES
            / log
            / f"assignments_activity2vec_log_context_d{rank}_k{k}.parquet"
        )
    if partition == "tfidf":
        suffix = "k8" if log == "sepsis" else "dynamic"
        return FULL_BENCHMARK / log / f"assignments_ngram_tfidf_1_3_{suffix}.parquet"
    raise ValueError(f"Unknown non-random partition: {partition!r}")


def frozen_partition_file(log: str) -> Path:
    spec = LOG_SPECS[log]
    return (
        FULL_BENCHMARK
        / log
        / f"assignments_p7_ordered_pyramid_1_2_4_bge_{spec['frozen_suffix']}.parquet"
    )


def make_random_partition(base: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Permute ``cluster_label`` across variants, preserving cluster sizes."""
    ordered = base[["variant_id", "cluster_label"]].sort_values("variant_id").reset_index(
        drop=True
    )
    labels = ordered["cluster_label"].to_numpy()
    permuted = np.random.default_rng(seed).permutation(labels)
    return pd.DataFrame(
        {"variant_id": ordered["variant_id"], "cluster_label": permuted}
    )


def load_partition(log: str, partition: str) -> pd.DataFrame:
    """``variant_id``, ``cluster_label`` for one partition of one log."""
    if partition in RANDOM_SEEDS:
        base = pd.read_parquet(_partition_source_file(log, "p7"))[
            ["variant_id", "cluster_label"]
        ]
        return make_random_partition(base, RANDOM_SEEDS[partition])
    path = _partition_source_file(log, partition)
    frame = pd.read_parquet(path)
    return frame[["variant_id", "cluster_label"]].copy()


# ---------------------------------------------------------------------------
# Pipeline check
# ---------------------------------------------------------------------------


def pipeline_check(log: str) -> dict[str, Any]:
    """Case-weighted ARI between the P7 partition and the frozen pyramid."""
    variants = load_variants(log)[["variant_id", "frequency"]]
    p7 = load_partition(log, "p7")
    frozen = pd.read_parquet(frozen_partition_file(log))[
        ["variant_id", "cluster_label"]
    ]
    merged = (
        variants.merge(p7, on="variant_id", how="inner", suffixes=("", "_p7"))
        .merge(frozen, on="variant_id", how="inner", suffixes=("_p7", "_frozen"))
    )
    if len(merged) != len(variants) or len(merged) != len(frozen):
        raise ValueError(
            f"{log}: P7 and frozen partitions do not cover the same variant set"
        )
    metrics = partition_similarity(
        merged["cluster_label_p7"].to_numpy(),
        merged["cluster_label_frozen"].to_numpy(),
        merged["frequency"].to_numpy().astype(np.int64),
    )
    passed = bool(metrics["case_weighted_ari"] == 1.0)
    return {
        "log": log,
        "n_variants": int(len(merged)),
        "case_weighted_ari": metrics["case_weighted_ari"],
        "variant_ari": metrics["variant_ari"],
        "passed": passed,
    }


def run_pipeline_check() -> pd.DataFrame:
    rows = [pipeline_check(log) for log in LOG_SPECS]
    frame = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "pipeline_check.csv", index=False)
    if not frame["passed"].all():
        failing = frame.loc[~frame["passed"], "log"].tolist()
        raise RuntimeError(f"Pipeline check failed for: {failing}")
    return frame


# ---------------------------------------------------------------------------
# Cluster profiles (deterministic, no LLM)
# ---------------------------------------------------------------------------


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return float(np.interp(q, cumulative, values))


def _case_prevalence(
    frame: pd.DataFrame, activities: list[str]
) -> tuple[dict[str, float], int]:
    total_cases = int(frame["frequency"].sum())
    counts = dict.fromkeys(activities, 0)
    for sequence, freq in zip(
        frame["activity_sequence"], frame["frequency"], strict=True
    ):
        for activity in set(sequence):
            counts[activity] = counts.get(activity, 0) + freq
    if total_cases == 0:
        return {activity: 0.0 for activity in activities}, 0
    return {
        activity: counts[activity] / total_cases for activity in activities
    }, total_cases


def _transition_shares(frame: pd.DataFrame) -> dict[tuple[str, str], float]:
    total_cases = int(frame["frequency"].sum())
    if total_cases == 0:
        return {}
    counts: dict[tuple[str, str], int] = {}
    for sequence, freq in zip(
        frame["activity_sequence"], frame["frequency"], strict=True
    ):
        seen: set[tuple[str, str]] = set()
        for left, right in zip(sequence, sequence[1:], strict=False):
            pair = (left, right)
            if pair not in seen:
                counts[pair] = counts.get(pair, 0) + freq
                seen.add(pair)
    return {pair: count / total_cases for pair, count in counts.items()}


def compute_cluster_profiles(
    variants_train: pd.DataFrame,
    partition_df: pd.DataFrame,
    *,
    top_activity_n: int = 20,
    top_dft_n: int = 15,
    top_distinctive_n: int = 10,
    distinctive_min_prevalence: float = 0.10,
    top_variant_n: int = 10,
    lift_epsilon: float = 1e-6,
) -> dict[str, dict[str, Any]]:
    """Deterministic per-cluster profile from training-split variants.

    ``variants_train`` must have ``variant_id``, ``activity_sequence`` (a
    list), ``frequency``, ``sequence_length``. ``partition_df`` must have
    ``variant_id``, ``cluster_label``. Only variants present in both are
    profiled. Every evidence item (activity, transition, variant) gets a
    stable ID so that generated names can cite evidence and citations can be
    checked mechanically.
    """
    merged = variants_train.merge(partition_df, on="variant_id", how="inner")
    if merged.empty:
        raise ValueError("No overlap between training variants and partition")
    total_cases_all = int(merged["frequency"].sum())
    activities = sorted({a for seq in merged["activity_sequence"] for a in seq})
    clusters = sorted(merged["cluster_label"].unique(), key=str)

    profiles: dict[str, dict[str, Any]] = {}
    for cluster in clusters:
        in_cluster = merged[merged["cluster_label"] == cluster]
        rest = merged[merged["cluster_label"] != cluster]

        cluster_prev, cluster_cases = _case_prevalence(in_cluster, activities)
        rest_prev, _ = _case_prevalence(rest, activities)
        cluster_trans = _transition_shares(in_cluster)
        rest_trans = _transition_shares(rest)

        prevalence_sorted = sorted(activities, key=lambda a: -cluster_prev[a])
        top_prevalence = [a for a in prevalence_sorted if cluster_prev[a] > 0][
            :top_activity_n
        ]
        activity_lift = {
            a: cluster_prev[a] / (rest_prev[a] + lift_epsilon) for a in activities
        }
        distinctive_activity_candidates = [
            a for a in activities if cluster_prev[a] >= distinctive_min_prevalence
        ]
        distinctive_activities_sorted = sorted(
            distinctive_activity_candidates, key=lambda a: -activity_lift[a]
        )[:top_distinctive_n]

        evidence_activities: list[str] = []
        for activity in top_prevalence + distinctive_activities_sorted:
            if activity not in evidence_activities:
                evidence_activities.append(activity)
        activity_id = {
            activity: f"A{index + 1}"
            for index, activity in enumerate(evidence_activities)
        }
        activity_prevalence_table = [
            {
                "id": activity_id[a],
                "activity": a,
                "case_share": round(cluster_prev[a], 4),
            }
            for a in top_prevalence
        ]
        distinctive_activities_table = [
            {
                "id": activity_id[a],
                "activity": a,
                "case_share": round(cluster_prev[a], 4),
                "lift": round(activity_lift[a], 3),
            }
            for a in distinctive_activities_sorted
        ]

        trans_sorted_by_share = sorted(cluster_trans, key=lambda t: -cluster_trans[t])
        top_dft = trans_sorted_by_share[:top_dft_n]
        trans_lift = {
            t: cluster_trans[t] / (rest_trans.get(t, 0.0) + lift_epsilon)
            for t in cluster_trans
        }
        distinctive_trans_candidates = [
            t for t in cluster_trans if cluster_trans[t] >= distinctive_min_prevalence
        ]
        distinctive_trans_sorted = sorted(
            distinctive_trans_candidates, key=lambda t: -trans_lift[t]
        )[:top_distinctive_n]

        evidence_transitions: list[tuple[str, str]] = []
        for transition in top_dft + distinctive_trans_sorted:
            if transition not in evidence_transitions:
                evidence_transitions.append(transition)
        transition_id = {
            transition: f"T{index + 1}"
            for index, transition in enumerate(evidence_transitions)
        }
        dft_table = [
            {
                "id": transition_id[t],
                "from": t[0],
                "to": t[1],
                "case_share": round(cluster_trans[t], 4),
            }
            for t in top_dft
        ]
        distinctive_transitions_table = [
            {
                "id": transition_id[t],
                "from": t[0],
                "to": t[1],
                "case_share": round(cluster_trans[t], 4),
                "lift": round(trans_lift[t], 3),
            }
            for t in distinctive_trans_sorted
        ]

        lengths = in_cluster["sequence_length"].to_numpy()
        weights = in_cluster["frequency"].to_numpy()
        q1 = _weighted_quantile(lengths, weights, 0.25)
        median = _weighted_quantile(lengths, weights, 0.5)
        q3 = _weighted_quantile(lengths, weights, 0.75)

        rework_mask = in_cluster["activity_sequence"].map(
            lambda sequence: len(sequence) != len(set(sequence))
        )
        rework_share = (
            float(in_cluster.loc[rework_mask, "frequency"].sum() / cluster_cases)
            if cluster_cases
            else 0.0
        )

        start_counts = in_cluster.groupby(
            in_cluster["activity_sequence"].map(lambda s: s[0])
        )["frequency"].sum()
        end_counts = in_cluster.groupby(
            in_cluster["activity_sequence"].map(lambda s: s[-1])
        )["frequency"].sum()

        top_variants = in_cluster.sort_values(
            ["frequency", "variant_id"], ascending=[False, True]
        ).head(top_variant_n)
        variant_table = [
            {
                "id": f"V{index + 1}",
                "variant_id": row.variant_id,
                "activity_sequence": list(row.activity_sequence),
                "cases": int(row.frequency),
            }
            for index, row in enumerate(top_variants.itertuples(index=False))
        ]

        evidence_ids = sorted(
            set(activity_id.values())
            | set(transition_id.values())
            | {entry["id"] for entry in variant_table}
        )

        profiles[str(cluster)] = {
            "cluster_label": cluster.item() if hasattr(cluster, "item") else cluster,
            "size": {
                "n_variants": int(len(in_cluster)),
                "n_cases": int(cluster_cases),
                "case_share_of_log": (
                    round(cluster_cases / total_cases_all, 4)
                    if total_cases_all
                    else 0.0
                ),
            },
            "activity_prevalence": activity_prevalence_table,
            "top_directly_follows_transitions": dft_table,
            "distinctive_activities": distinctive_activities_table,
            "distinctive_transitions": distinctive_transitions_table,
            "sequence_length": {
                "median": round(median, 3),
                "q1": round(q1, 3),
                "q3": round(q3, 3),
                "iqr": round(q3 - q1, 3),
            },
            "rework_share": round(rework_share, 4),
            "most_common_start_activity": (
                str(start_counts.idxmax()) if len(start_counts) else None
            ),
            "most_common_end_activity": (
                str(end_counts.idxmax()) if len(end_counts) else None
            ),
            "top_variants": variant_table,
            "evidence_ids": evidence_ids,
        }
    return profiles


def profile_log_partition(log: str, partition: str) -> dict[str, dict[str, Any]]:
    variants = load_variants(log)
    split = load_split(log)
    train_ids = set(split.loc[split["split"] == "train", "variant_id"])
    variants_train = variants[variants["variant_id"].isin(train_ids)]
    partition_df = load_partition(log, partition)
    return compute_cluster_profiles(variants_train, partition_df)


def run_profile_step(logs: list[str] | None = None) -> None:
    for log in logs or list(LOG_SPECS):
        for partition in NAMED_PARTITIONS:
            profiles = profile_log_partition(log, partition)
            out_dir = OUTPUT / "profiles" / log
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{partition}.json").write_text(
                json.dumps(profiles, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"[profile] {log}/{partition}: {len(profiles)} clusters", flush=True)


# ---------------------------------------------------------------------------
# Blinding: seeded letter codes
# ---------------------------------------------------------------------------


def assign_letter_codes(
    cluster_keys: list[str], *, log: str, partition: str, seed: int = SEED
) -> dict[str, str]:
    """A seeded, deterministic, random cluster-key -> letter-code mapping."""
    if len(cluster_keys) > 26:
        raise ValueError("More than 26 clusters is not supported by letter codes")
    letters = list(string.ascii_uppercase[: len(cluster_keys)])
    combined_seed = (seed + zlib.crc32(f"{log}:{partition}".encode())) % (2**32)
    shuffled = np.random.default_rng(combined_seed).permutation(letters)
    sorted_keys = sorted(cluster_keys, key=str)
    return {key: str(letter) for key, letter in zip(sorted_keys, shuffled, strict=True)}


# ---------------------------------------------------------------------------
# Naming (one LLM call per log x partition)
# ---------------------------------------------------------------------------


def validate_business_cluster_names(
    response: dict[str, Any],
    *,
    partition_code: str,
    letter_codes: dict[str, str],
    profiles_by_code: dict[str, dict[str, Any]],
) -> None:
    validate_response(response, NAMING_SCHEMA)
    if response["partition_code"] != partition_code:
        raise ValueError("partition_code does not match the request")
    expected_codes = set(letter_codes.values())
    if response["cluster_count"] != len(expected_codes):
        raise ValueError("cluster_count does not match the number of clusters")
    entries = response["clusters"]
    if len(entries) != len(expected_codes):
        raise ValueError(
            f"Expected {len(expected_codes)} cluster entries, got {len(entries)}"
        )

    seen_codes: set[str] = set()
    seen_names: set[str] = set()
    for entry in entries:
        code = entry["cluster_code"]
        if code not in expected_codes:
            raise ValueError(f"Unknown cluster_code {code!r}")
        if code in seen_codes:
            raise ValueError(f"Duplicate cluster_code {code!r}")
        seen_codes.add(code)

        name = entry["business_name"]
        word_count = len(name.split())
        if not 3 <= word_count <= 8:
            raise ValueError(
                f"business_name must contain 3-8 words, got {word_count}: {name!r}"
            )
        normalized_name = " ".join(name.casefold().split())
        if normalized_name in seen_names:
            raise ValueError(f"Duplicate business_name within partition: {name!r}")
        seen_names.add(normalized_name)

        allowed_ids = set(profiles_by_code[code]["evidence_ids"])
        cited = set(entry["evidence_ids"])
        if not cited <= allowed_ids:
            raise ValueError(
                f"Cluster {code} cites evidence outside its own profile: "
                f"{sorted(cited - allowed_ids)}"
            )
        for feature in entry["distinguishing_features"]:
            feature_ids = set(feature["evidence_ids"])
            if not feature_ids <= allowed_ids:
                raise ValueError(
                    f"Cluster {code} distinguishing feature cites evidence "
                    f"outside its own profile: {sorted(feature_ids - allowed_ids)}"
                )

        rendered = json.dumps(entry, ensure_ascii=False).casefold()
        if BANNED_WORD_PATTERN.search(rendered):
            raise ValueError(
                f"Cluster {code} name/definition contains a banned word or phrase"
            )

    if seen_codes != expected_codes:
        raise ValueError("Response does not cover every cluster code exactly once")


def name_log_partition(
    log: str,
    partition: str,
    *,
    endpoint: str,
    deployment: str,
    model_version: str,
    expected_model: str,
    max_tokens: int,
    repair_max_tokens: int,
    cache_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    profile_path = OUTPUT / "profiles" / log / f"{partition}.json"
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))
    letter_codes = assign_letter_codes(list(profiles), log=log, partition=partition)
    profiles_by_code = {
        letter_codes[key]: _prefix_evidence_ids(
            {**value, "cluster_code": letter_codes[key]}, letter_codes[key]
        )
        for key, value in profiles.items()
    }
    # Blinded payload: no cluster_label (the real assignment identity), no
    # dataset name, no partition source, and (B3) no log/partition identity
    # in any LLM-visible string.
    blinded_profiles = [
        {k: v for k, v in profile.items() if k != "cluster_label"}
        for profile in sorted(
            profiles_by_code.values(), key=lambda p: p["cluster_code"]
        )
    ]
    partition_code = _opaque_code("partition", log, partition)

    artifact = render_prompt(
        NAMING_PROMPT,
        NAMING_SCHEMA,
        {
            "PARTITION_CODE": partition_code,
            "CLUSTER_COUNT": str(len(blinded_profiles)),
            "CLUSTER_PROFILES_JSON": json.dumps(
                blinded_profiles, ensure_ascii=False
            ),
        },
        prompt_version="p8",
    )
    validator = partial(
        validate_business_cluster_names,
        partition_code=partition_code,
        letter_codes=letter_codes,
        profiles_by_code=profiles_by_code,
    )
    if dry_run:
        return {
            "log": log,
            "partition": partition,
            "letter_codes": letter_codes,
            "dry_run": True,
            "prompt_chars": len(artifact.rendered_prompt),
            "estimated_input_tokens": len(artifact.rendered_prompt) // 4,
            "estimated_output_tokens": max_tokens,
        }

    schema = json.loads(NAMING_SCHEMA.read_text(encoding="utf-8"))
    parameters = GenerationParameters(
        temperature=0.0, seed=None, max_output_tokens=max_tokens
    )
    identity = ModelIdentity(
        provider="azure_openai_entra", model=deployment, model_version=model_version
    )
    cache = StructuredResponseCache(cache_dir)

    def provider_factory(call_parameters: GenerationParameters) -> Any:
        from llm_variant_clustering.azure_openai_provider import (
            AzureOpenAIToolProvider,
        )

        return AzureOpenAIToolProvider(
            endpoint=endpoint,
            deployment=deployment,
            api_version=API_VERSION,
            parameters=call_parameters,
            schema=schema,
            schema_name="business_cluster_names_p8",
            max_attempts=MAX_PROVIDER_ATTEMPTS,
            backoff_seconds=PROVIDER_BACKOFF_SECONDS,
            max_backoff_seconds=PROVIDER_BACKOFF_CAP_SECONDS,
            jitter_seconds=PROVIDER_JITTER_SECONDS,
            honor_retry_after=True,
        )

    response, cache_hit = execute_cached_request(
        cache=cache,
        model=identity,
        parameters=parameters,
        prompt=artifact,
        input_sha256=canonical_json_hash(blinded_profiles),
        generate=partial(
            generate_with_retries_and_repairs,
            provider_factory=provider_factory,
            base_parameters=parameters,
            repair_max_tokens=repair_max_tokens,
            validator=validator,
            expected_model=expected_model,
        ),
        validate=validator,
    )
    return {
        "log": log,
        "partition": partition,
        "letter_codes": letter_codes,
        "response": response,
        "cache_hit": cache_hit,
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PrespecConfirmationError(RuntimeError):
    """Raised when ``--confirm-prespec`` is missing or stale (Section 13, B8)."""


def require_prespec_confirmation(confirm_prespec: str | None) -> str:
    """B8: ``name``/``judge`` must be invoked with the live PRESPEC.md SHA-256.

    This forces whoever runs real LLM calls to have actually read the current
    pre-specification (including Amendment 1) rather than running against a
    stale copy. Returns the live hash on success.
    """
    prespec_path = OUTPUT / "PRESPEC.md"
    if not prespec_path.exists():
        raise PrespecConfirmationError(f"PRESPEC not found at {prespec_path}")
    live_sha = _file_sha256(prespec_path)
    if not confirm_prespec:
        raise PrespecConfirmationError(
            "Refusing to run: pass --confirm-prespec "
            f"{live_sha} after reading the current PRESPEC.md (including "
            "Amendment 1)."
        )
    if confirm_prespec != live_sha:
        raise PrespecConfirmationError(
            f"--confirm-prespec {confirm_prespec!r} does not match the live "
            f"PRESPEC.md hash {live_sha!r}. The pre-spec changed underneath "
            "you -- re-read it before confirming."
        )
    return live_sha


def run_name_step(
    logs: list[str] | None = None,
    *,
    partitions: list[str] | None = None,
    dry_run: bool = False,
    confirm_prespec: str | None = None,
) -> list[dict[str, Any]]:
    require_prespec_confirmation(confirm_prespec)
    model = MODELS["namer_gpt41"]
    cache_dir = OUTPUT / "cache" / "namer_gpt41"
    failures_path = OUTPUT / "failures" / "name_namer_gpt41.jsonl"
    if not dry_run:
        # B5/S6: abort before any real call if this fails; use whichever
        # deployment (primary or fallback) actually answered the probe.
        model = run_startup_smoke_probe(["namer_gpt41"])["namer_gpt41"]
    units = [
        (log, partition)
        for log in (logs or list(LOG_SPECS))
        for partition in (partitions or NAMED_PARTITIONS)
    ]
    breaker = CircuitBreaker(total_units=len(units))
    rows: list[dict[str, Any]] = []
    dry_run_rows: list[dict[str, Any]] = []
    lock = threading.Lock()

    def run_one(unit: tuple[str, str]) -> None:
        log, partition = unit
        breaker.check()
        result = name_log_partition(
            log,
            partition,
            endpoint=model["endpoint"],
            deployment=model["deployment"],
            model_version=model["model_version"],
            expected_model=model["expected_model"],
            max_tokens=model["max_tokens"],
            repair_max_tokens=model["repair_max_tokens"],
            cache_dir=cache_dir,
            dry_run=dry_run,
        )
        if dry_run:
            with lock:
                dry_run_rows.append(result)
            return
        out_dir = OUTPUT / "names" / log
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{partition}.json").write_text(
            json.dumps(
                {
                    "letter_codes": result["letter_codes"],
                    "response": result["response"],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        with lock:
            for entry in result["response"]["clusters"]:
                rows.append(
                    {
                        "log": log,
                        "partition": partition,
                        "cluster_code": entry["cluster_code"],
                        "business_name": entry["business_name"],
                        "definition": entry["definition"],
                        "confidence": entry["confidence"],
                        "cache_hit": result["cache_hit"],
                    }
                )
        print(
            f"[name] {log}/{partition} done (cache_hit={result['cache_hit']})",
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=model["workers"]) as pool:
        futures = {pool.submit(run_one, unit): unit for unit in units}
        for future in as_completed(futures):
            unit = futures[future]
            try:
                future.result()
            except FatalCallError:
                for other in futures:
                    other.cancel()
                raise
            except Exception as error:  # noqa: BLE001 - isolate per unit
                breaker.record(succeeded=False)
                write_failure_record(
                    failures_path,
                    unit={"log": unit[0], "partition": unit[1]},
                    error=error,
                )
                print(
                    f"[name] {unit[0]}/{unit[1]} FAILED: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                breaker.check()
            else:
                breaker.record(succeeded=True)

    if dry_run:
        return dry_run_rows
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT / "names_table.csv", index=False)
    return rows


# ---------------------------------------------------------------------------
# Judging held-out variants
# ---------------------------------------------------------------------------


def sample_heldout_variants(log: str, *, cap: int = 300, seed: int = SEED) -> list[str]:
    split = load_split(log)
    test_ids = sorted(split.loc[split["split"] == "test", "variant_id"])
    if len(test_ids) <= cap:
        return test_ids
    rng = np.random.default_rng(seed)
    positions = rng.choice(len(test_ids), size=cap, replace=False)
    return sorted(np.array(test_ids)[positions].tolist())


def build_judge_batches(
    sample: list[str], *, batch_size: int = 10, seed: int = SEED
) -> list[list[str]]:
    """Split a fixed held-out sample into fixed, seeded, shuffled batches."""
    order = np.random.default_rng(seed).permutation(len(sample))
    shuffled = [sample[i] for i in order]
    return [
        shuffled[i : i + batch_size] for i in range(0, len(shuffled), batch_size)
    ]


def validate_judge_response(
    response: dict[str, Any], *, batch_id: str, slot_ids: set[str], name_codes: set[str]
) -> None:
    validate_response(response, JUDGE_SCHEMA)
    if response["batch_id"] != batch_id:
        raise ValueError("batch_id does not match the request")
    answers = response["answers"]
    seen_slots: set[str] = set()
    allowed_codes = name_codes | {"unclear"}
    for answer in answers:
        slot = answer["slot_id"]
        if slot not in slot_ids:
            raise ValueError(f"Unknown slot_id {slot!r}")
        if slot in seen_slots:
            raise ValueError(f"Duplicate slot_id {slot!r}")
        seen_slots.add(slot)
        if answer["chosen_name_code"] not in allowed_codes:
            raise ValueError(f"Unknown chosen_name_code {answer['chosen_name_code']!r}")
    if seen_slots != slot_ids:
        raise ValueError("Response does not cover every variant slot exactly once")


def _judge_candidate_set(log: str, partition: str, *, dry_run: bool = False) -> dict[str, Any]:
    """Blinded candidate-name set and true-label lookups for one partition.

    In ``--dry-run`` mode, if the namer hasn't produced real output yet,
    synthesizes placeholder names (same K, same letter codes, derived from
    the deterministic cluster profile) so judge prompts can be rendered and
    token/call estimates reported without requiring a prior real ``name``
    run. Never used for anything but token/call estimation -- real `judge`
    runs always require real names on disk.
    """
    names_path = OUTPUT / "names" / log / f"{partition}.json"
    if names_path.exists():
        names = json.loads(names_path.read_text(encoding="utf-8"))
    elif dry_run:
        profile_path = OUTPUT / "profiles" / log / f"{partition}.json"
        profiles = json.loads(profile_path.read_text(encoding="utf-8"))
        letter_codes = assign_letter_codes(list(profiles), log=log, partition=partition)
        names = {
            "letter_codes": letter_codes,
            "response": {
                "clusters": [
                    {
                        "cluster_code": code,
                        "business_name": "Placeholder dry run example business name",
                        "definition": (
                            "Placeholder definition sentence used only to "
                            "estimate prompt size before any real name "
                            "exists."
                        ),
                    }
                    for code in sorted(set(letter_codes.values()))
                ]
            },
        }
    else:
        raise FileNotFoundError(
            f"{names_path} does not exist -- run `name` for {log}/{partition} "
            "before `judge`"
        )
    letter_codes = names["letter_codes"]  # real cluster_label(str) -> letter
    entries = {entry["cluster_code"]: entry for entry in names["response"]["clusters"]}
    letters = sorted(entries)
    name_codes = [f"N{i + 1}" for i in range(len(letters))]
    code_seed = (SEED + zlib.crc32(f"judge:{log}:{partition}".encode())) % (2**32)
    shuffled_letters = np.random.default_rng(code_seed).permutation(letters)
    name_code_for_letter = dict(zip(shuffled_letters, name_codes, strict=True))
    candidate_names = [
        {
            "name_code": name_code_for_letter[letter],
            "business_name": entries[letter]["business_name"],
            "definition": entries[letter]["definition"],
        }
        for letter in shuffled_letters
    ] + [
        {
            "name_code": "unclear",
            "business_name": "unclear",
            "definition": "none of the above plausibly fits",
        }
    ]
    partition_df = load_partition(log, partition)
    label_map = {
        str(row.variant_id): str(row.cluster_label)
        for row in partition_df.itertuples(index=False)
    }
    return {
        "candidate_names": candidate_names,
        "name_codes": name_codes,
        "name_code_for_letter": name_code_for_letter,
        "label_map": label_map,
        "letter_codes": letter_codes,
    }


def judge_one_batch(
    log: str,
    partition: str,
    batch_index: int,
    batch: list[str],
    *,
    judge_key: str,
    endpoint: str,
    deployment: str,
    model_version: str,
    expected_model: str,
    max_tokens: int,
    repair_max_tokens: int,
    cache_dir: Path,
    sequence_map: dict[str, list[str]],
    candidate_set: dict[str, Any],
    dry_run: bool = False,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Run (and cache) a single judge batch. One unit of work for isolation."""
    batch_id = _opaque_code("judge", log, partition, str(batch_index))
    var_order = np.random.default_rng(
        (SEED + zlib.crc32(f"{log}:{partition}:{batch_index}".encode())) % (2**32)
    ).permutation(len(batch))
    shuffled_batch = [batch[i] for i in var_order]
    slot_ids = [f"S{i + 1}" for i in range(len(shuffled_batch))]
    variant_payload = [
        {"slot_id": slot, "activity_sequence": sequence_map[variant_id]}
        for slot, variant_id in zip(slot_ids, shuffled_batch, strict=True)
    ]
    artifact = render_prompt(
        JUDGE_PROMPT,
        JUDGE_SCHEMA,
        {
            "BATCH_ID": batch_id,
            "CANDIDATE_NAMES_JSON": json.dumps(
                candidate_set["candidate_names"], ensure_ascii=False
            ),
            "VARIANT_BATCH_JSON": json.dumps(variant_payload, ensure_ascii=False),
        },
        prompt_version="p9",
    )
    validator = partial(
        validate_judge_response,
        batch_id=batch_id,
        slot_ids=set(slot_ids),
        name_codes=set(candidate_set["name_codes"]),
    )
    if dry_run:
        return {
            "log": log,
            "partition": partition,
            "batch_index": batch_index,
            "prompt_chars": len(artifact.rendered_prompt),
            "estimated_input_tokens": len(artifact.rendered_prompt) // 4,
            "estimated_output_tokens": max_tokens,
        }

    schema = json.loads(JUDGE_SCHEMA.read_text(encoding="utf-8"))
    parameters = GenerationParameters(
        temperature=0.0, seed=None, max_output_tokens=max_tokens
    )
    identity = ModelIdentity(
        provider="azure_openai_entra", model=deployment, model_version=model_version
    )
    cache = StructuredResponseCache(cache_dir)

    def provider_factory(call_parameters: GenerationParameters) -> Any:
        from llm_variant_clustering.azure_openai_provider import (
            AzureOpenAIToolProvider,
        )

        return AzureOpenAIToolProvider(
            endpoint=endpoint,
            deployment=deployment,
            api_version=API_VERSION,
            parameters=call_parameters,
            schema=schema,
            schema_name="judge_variant_match_p9",
            max_attempts=MAX_PROVIDER_ATTEMPTS,
            backoff_seconds=PROVIDER_BACKOFF_SECONDS,
            max_backoff_seconds=PROVIDER_BACKOFF_CAP_SECONDS,
            jitter_seconds=PROVIDER_JITTER_SECONDS,
            honor_retry_after=True,
        )

    response, cache_hit = execute_cached_request(
        cache=cache,
        model=identity,
        parameters=parameters,
        prompt=artifact,
        input_sha256=canonical_json_hash(
            {"candidates": candidate_set["candidate_names"], "variants": variant_payload}
        ),
        generate=partial(
            generate_with_retries_and_repairs,
            provider_factory=provider_factory,
            base_parameters=parameters,
            repair_max_tokens=repair_max_tokens,
            validator=validator,
            expected_model=expected_model,
        ),
        validate=validator,
    )
    answer_by_slot = {a["slot_id"]: a for a in response["answers"]}
    rows: list[dict[str, Any]] = []
    for slot, variant_id in zip(slot_ids, shuffled_batch, strict=True):
        answer = answer_by_slot[slot]
        true_letter = candidate_set["letter_codes"][candidate_set["label_map"][variant_id]]
        true_name_code = candidate_set["name_code_for_letter"][true_letter]
        rows.append(
            {
                "log": log,
                "partition": partition,
                "judge": judge_key,
                "batch_id": batch_id,
                "variant_id": variant_id,
                "true_cluster_label": candidate_set["label_map"][variant_id],
                "true_name_code": true_name_code,
                "chosen_name_code": answer["chosen_name_code"],
                "confidence": answer["confidence"],
                "correct": answer["chosen_name_code"] == true_name_code,
                "cache_hit": cache_hit,
            }
        )
    return rows


def run_judge_step(
    judge_key: str,
    logs: list[str] | None = None,
    workers: int | None = None,
    *,
    partitions: list[str] | None = None,
    dry_run: bool = False,
    allow_incomplete: bool = False,
    confirm_prespec: str | None = None,
) -> dict[str, Any]:
    require_prespec_confirmation(confirm_prespec)
    model = MODELS[judge_key]
    cache_dir = OUTPUT / "cache" / judge_key
    failures_path = OUTPUT / "failures" / f"judge_{judge_key}.jsonl"
    if not dry_run:
        # B5/S6: abort before any real call if this fails; use whichever
        # deployment (primary or fallback) actually answered the probe.
        model = run_startup_smoke_probe([judge_key])[judge_key]
    # B6: never exceed the per-model concurrency cap (DeepSeek <=4, GPT-4o
    # <=2), even if --workers asks for more; only ever narrows it.
    workers = min(workers, model["workers"]) if workers else model["workers"]
    requested_partitions = partitions or list(NAMED_PARTITIONS)
    summary: dict[str, Any] = {"judge": judge_key, "logs": {}}

    for log in logs or list(LOG_SPECS):
        variants = load_variants(log)
        sequence_map = {
            str(row.variant_id): row.activity_sequence
            for row in variants.itertuples(index=False)
        }
        sample = sample_heldout_variants(log)
        batches = build_judge_batches(sample)
        out_dir = OUTPUT / "judgements" / judge_key / log
        out_dir.mkdir(parents=True, exist_ok=True)

        partitions = (
            list(requested_partitions)
            if dry_run
            else [
                p
                for p in requested_partitions
                if (OUTPUT / "names" / log / f"{p}.json").exists()
            ]
        )
        candidate_sets = {
            p: _judge_candidate_set(log, p, dry_run=dry_run) for p in partitions
        }

        # One unit of work = one (partition, batch) pair. Isolated per unit
        # (B6): a batch failure never stops other batches/partitions.
        units = [
            (partition, batch_index, batch)
            for partition in partitions
            for batch_index, batch in enumerate(batches)
        ]
        breaker = CircuitBreaker(total_units=len(units))
        rows_by_partition: dict[str, list[dict[str, Any]]] = {p: [] for p in partitions}
        dry_run_rows: list[dict[str, Any]] = []
        failed_variant_ids: set[str] = set()
        lock = threading.Lock()

        def run_one(unit: tuple[str, int, list[str]]) -> None:
            partition, batch_index, batch = unit
            breaker.check()
            result = judge_one_batch(
                log,
                partition,
                batch_index,
                batch,
                judge_key=judge_key,
                endpoint=model["endpoint"],
                deployment=model["deployment"],
                model_version=model["model_version"],
                expected_model=model["expected_model"],
                max_tokens=model["max_tokens"],
                repair_max_tokens=model["repair_max_tokens"],
                cache_dir=cache_dir,
                sequence_map=sequence_map,
                candidate_set=candidate_sets[partition],
                dry_run=dry_run,
            )
            with lock:
                if dry_run:
                    dry_run_rows.append(result)
                else:
                    rows_by_partition[partition].extend(result)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_one, unit): unit for unit in units}
            for future in as_completed(futures):
                partition, batch_index, batch = futures[future]
                try:
                    future.result()
                except FatalCallError:
                    for other in futures:
                        other.cancel()
                    raise
                except Exception as error:  # noqa: BLE001 - isolate per unit
                    breaker.record(succeeded=False)
                    with lock:
                        failed_variant_ids.update(batch)
                    write_failure_record(
                        failures_path,
                        unit={
                            "log": log,
                            "partition": partition,
                            "batch_index": batch_index,
                            "variant_ids": batch,
                        },
                        error=error,
                    )
                    print(
                        f"[judge:{judge_key}] {log}/{partition} batch "
                        f"{batch_index} FAILED: {type(error).__name__}: {error}",
                        flush=True,
                    )
                    breaker.check()
                else:
                    breaker.record(succeeded=True)

        if dry_run:
            summary["logs"][log] = {"dry_run_batches": dry_run_rows}
            continue

        # Missing-data / paired-exclusion rule (Section 13, B6): if a batch
        # ultimately failed for ANY partition, every variant in it is
        # excluded from ALL partitions of this log+judge, so every
        # partition's judged sample stays identically paired.
        excluded = sorted(failed_variant_ids)
        n_sample = len(sample)
        incomplete_fraction = (len(excluded) / n_sample) if n_sample else 0.0
        incomplete = incomplete_fraction > INCOMPLETE_DATA_THRESHOLD
        if excluded:
            print(
                f"[judge:{judge_key}] {log}: excluding {len(excluded)}/{n_sample} "
                f"({incomplete_fraction:.1%}) variants from every partition "
                f"(paired-exclusion rule)",
                flush=True,
            )
        if incomplete and not allow_incomplete:
            print(
                f"[judge:{judge_key}] {log}: incomplete_fraction "
                f"{incomplete_fraction:.1%} exceeds {INCOMPLETE_DATA_THRESHOLD:.0%} "
                "threshold -- marked incomplete",
                flush=True,
            )
        for partition in partitions:
            frame = pd.DataFrame(rows_by_partition[partition])
            if not frame.empty and excluded:
                frame = frame[~frame["variant_id"].isin(excluded)]
            frame.to_parquet(out_dir / f"{partition}.parquet", index=False)
            print(
                f"[judge:{judge_key}] {log}/{partition}: {len(frame)} judgements, "
                f"{int((~frame['correct']).sum()) if len(frame) else 0} miss",
                flush=True,
            )
        log_summary = {
            "n_sample": n_sample,
            "excluded_variants": excluded,
            "incomplete_fraction": incomplete_fraction,
            "incomplete": incomplete,
        }
        summary["logs"][log] = log_summary
        (out_dir / "_summary.json").write_text(
            json.dumps(log_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return summary


# ---------------------------------------------------------------------------
# Statistics (kappa, paired bootstrap, classification)
# ---------------------------------------------------------------------------


def case_weighted_cohen_kappa(
    true_labels: np.ndarray, pred_labels: np.ndarray, weights: np.ndarray
) -> float:
    """Case-frequency-weighted Cohen's kappa (nominal categories)."""
    return float(
        cohen_kappa_score(true_labels, pred_labels, sample_weight=weights)
    )


def _classify(lower: float, upper: float) -> str:
    if math.isnan(lower) or math.isnan(upper):
        # Every bootstrap resample was dropped (each resample needs >=2
        # classes on every side being compared); the CI is undetermined, not
        # merely "includes zero" -- distinct from a normal null result and
        # never counted as informative (PI review of the analyze crash).
        return "undetermined"
    if lower > 0:
        return "positive"
    if upper < 0:
        return "negative"
    return "includes_zero"


def _percentile_interval(
    point: float,
    samples: np.ndarray,
    confidence: float,
    *,
    dropped_resamples: int = 0,
    requested_resamples: int | None = None,
) -> dict[str, float]:
    tail = (1.0 - confidence) / 2.0
    if samples.size == 0:
        # All resamples were dropped as single-class for this comparison
        # (small n, e.g. RTFM's held-out set). np.percentile cannot reduce
        # an empty array; report an undefined interval instead of crashing,
        # and the 100% dropped fraction below will trip "low_information".
        lower = upper = float("nan")
    else:
        lower, upper = np.percentile(samples, [100 * tail, 100 * (1 - tail)])
    requested = requested_resamples if requested_resamples is not None else (
        samples.size + dropped_resamples
    )
    dropped_fraction = float(dropped_resamples / requested) if requested else 0.0
    return {
        "point_estimate": float(point),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": float(confidence),
        "resamples": int(samples.size),
        "resamples_requested": int(requested),
        "dropped_resamples": int(dropped_resamples),
        "dropped_resamples_fraction": dropped_fraction,
        # Amendment 2: visibility flag, not an exclusion -- the interval
        # still counts towards the decision even when flagged.
        "low_information": bool(dropped_fraction > LOW_INFORMATION_DROPPED_FRACTION),
    }


def resample_indices(n: int, resamples: int, seed: int = SEED) -> np.ndarray:
    """Bootstrap resample indices, shared across every partition of a log."""
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n))


def _kappa_diff_samples(
    true_a: np.ndarray,
    pred_a: np.ndarray,
    true_b: np.ndarray,
    pred_b: np.ndarray,
    weights: np.ndarray,
    indices: np.ndarray,
) -> tuple[float, np.ndarray, int]:
    """Point estimate and per-resample kappa(A) - kappa(B) bootstrap samples.

    Returns ``(point_estimate, samples, dropped_resamples)``. A resample is
    dropped (B7) when either side's resampled subset has fewer than two true
    classes present, in which case kappa is undefined for that draw.
    """
    point = case_weighted_cohen_kappa(true_a, pred_a, weights) - case_weighted_cohen_kappa(
        true_b, pred_b, weights
    )
    samples = []
    dropped = 0
    for subset in indices:
        if np.unique(true_a[subset]).size < 2 or np.unique(true_b[subset]).size < 2:
            dropped += 1
            continue
        samples.append(
            case_weighted_cohen_kappa(true_a[subset], pred_a[subset], weights[subset])
            - case_weighted_cohen_kappa(true_b[subset], pred_b[subset], weights[subset])
        )
    return point, np.asarray(samples), dropped


def paired_kappa_bootstrap(
    true_a: np.ndarray,
    pred_a: np.ndarray,
    true_b: np.ndarray,
    pred_b: np.ndarray,
    weights: np.ndarray,
    *,
    indices: np.ndarray,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Paired bootstrap for kappa(A) minus kappa(B) on identical variants.

    ``true_a``/``pred_a`` and ``true_b``/``pred_b`` must be aligned to the
    same variant order as ``weights``; ``indices`` (from ``resample_indices``)
    is shared across every partition pair of the same log.
    """
    point, sample_array, dropped = _kappa_diff_samples(
        true_a, pred_a, true_b, pred_b, weights, indices
    )
    interval = _percentile_interval(
        point,
        sample_array,
        confidence,
        dropped_resamples=dropped,
        requested_resamples=indices.shape[0],
    )
    interval["class"] = _classify(interval["ci_lower"], interval["ci_upper"])
    if interval["dropped_resamples_fraction"] > DROPPED_RESAMPLE_WARN_FRACTION:
        print(
            f"[analyze] WARNING: dropped {interval['dropped_resamples']}/"
            f"{interval['resamples_requested']} "
            f"({interval['dropped_resamples_fraction']:.1%}) bootstrap resamples "
            "for a single-class subset",
            flush=True,
        )
    return interval


# ---------------------------------------------------------------------------
# Analyze: separability tables and the pre-specified decision
# ---------------------------------------------------------------------------

NOMINAL_CONFIDENCE = 0.95
LOG_FAMILY_SIZE = 5
BONFERRONI_CONFIDENCE = 1.0 - (1.0 - NOMINAL_CONFIDENCE) / LOG_FAMILY_SIZE  # 0.99
JUDGES = ("deepseek_v4_pro", "gpt4o")


def _variant_weights(log: str) -> dict[str, int]:
    variants = load_variants(log)
    return dict(
        zip(variants["variant_id"].astype(str), variants["frequency"].astype(int), strict=True)
    )


def _load_judgement_frame(judge_key: str, log: str, partition: str) -> pd.DataFrame | None:
    path = OUTPUT / "judgements" / judge_key / log / f"{partition}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def _load_bge_embedder() -> Any:
    """Load the frozen BGE embedder once per process (S4).

    ``analyze`` calls ``name_distinctness`` once per (judge, log, partition)
    row; re-downloading/re-loading the model from disk for every one of
    those calls is wasteful. Cached for the lifetime of the process.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        "BAAI/bge-large-en-v1.5",
        revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        cache_folder=str(ROOT / "models/huggingface"),
        device="cpu",
    )


def name_distinctness(names: list[str]) -> float:
    """Mean pairwise cosine similarity between a partition's names.

    Uses the frozen BGE embedder (``BAAI/bge-large-en-v1.5``, revision
    ``d4aa6901d3a41ba39fb536a557fa166f842b0e09``). Lower means more distinct.
    """
    if len(names) < 2:
        return float("nan")
    model = _load_bge_embedder()
    embeddings = np.asarray(
        model.encode(names, normalize_embeddings=True, convert_to_numpy=True)
    )
    similarity = embeddings @ embeddings.T
    mask = ~np.eye(len(names), dtype=bool)
    return float(similarity[mask].mean())



def _kish_effective_sample_size(weights: np.ndarray) -> float:
    """Kish's effective sample size: (sum w)^2 / sum(w^2).

    Frequency-only (no LLM output involved) -- quantifies how many
    "independent" judgements the case-weighted metrics actually rest on
    when a held-out sample is dominated by a few high-frequency variants
    (Amendment 2).
    """
    total = float(weights.sum())
    sq_total = float(np.square(weights).sum())
    if sq_total <= 0:
        return float("nan")
    return (total**2) / sq_total


def _separability_metrics(
    true: np.ndarray, pred: np.ndarray, correct: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    """kappa/accuracy/macro-recall/majority-baseline for one weighting scheme.

    Cohen's kappa is mathematically degenerate (evaluates to exactly 0, not
    "no agreement") when the held-out ``true`` labels have zero variance --
    chance agreement equals observed agreement by construction, so the
    formula collapses regardless of prediction quality. PI review: this must
    be reported as undefined (NaN, ``kappa_defined=False``), not as a real
    kappa of 0.0.
    """
    kappa_defined = bool(np.unique(true).size >= 2)
    kappa = case_weighted_cohen_kappa(true, pred, weights) if kappa_defined else float("nan")
    accuracy = float(np.average(correct.astype(np.float64), weights=weights))
    recalls = []
    for code in sorted(set(true)):
        mask = true == code
        if weights[mask].sum() <= 0:
            continue
        recalls.append(float(np.average((pred[mask] == code).astype(np.float64), weights=weights[mask])))
    macro_recall = float(np.mean(recalls)) if recalls else float("nan")
    cluster_case_totals = {code: float(weights[true == code].sum()) for code in set(true)}
    majority_code = max(cluster_case_totals, key=cluster_case_totals.get)
    majority_baseline_accuracy = float(cluster_case_totals[majority_code] / weights.sum())
    return {
        "kappa": kappa,
        "kappa_defined": kappa_defined,
        "accuracy": accuracy,
        "macro_recall": macro_recall,
        "majority_baseline_accuracy": majority_baseline_accuracy,
    }


def separability_row(judge_key: str, log: str, partition: str) -> dict[str, Any] | None:
    frame = _load_judgement_frame(judge_key, log, partition)
    if frame is None:
        return None
    weights_map = _variant_weights(log)
    weights = frame["variant_id"].astype(str).map(weights_map).to_numpy(dtype=np.float64)
    true = frame["true_name_code"].to_numpy()
    pred = frame["chosen_name_code"].to_numpy()
    correct = frame["correct"].to_numpy()

    # Amendment 2: unweighted variant-level metrics are PRIMARY (the judge's
    # unit is the variant type, one judgement each); case-weighted metrics
    # are secondary/descriptive, reported alongside the Kish effective
    # sample size (ESS) of the case weights so a reader can see how many
    # "independent" judgements the case-weighted numbers actually rest on.
    unweighted = _separability_metrics(true, pred, correct, np.ones_like(weights))
    case_weighted = _separability_metrics(true, pred, correct, weights)

    names_path = OUTPUT / "names" / log / f"{partition}.json"
    names = json.loads(names_path.read_text(encoding="utf-8"))
    business_names = [entry["business_name"] for entry in names["response"]["clusters"]]

    return {
        "judge": judge_key,
        "log": log,
        "partition": partition,
        "kappa": unweighted["kappa"],
        "kappa_defined": unweighted["kappa_defined"],
        "accuracy": unweighted["accuracy"],
        "macro_recall": unweighted["macro_recall"],
        "majority_baseline_accuracy": unweighted["majority_baseline_accuracy"],
        "kappa_case_weighted": case_weighted["kappa"],
        "accuracy_case_weighted": case_weighted["accuracy"],
        "macro_recall_case_weighted": case_weighted["macro_recall"],
        "majority_baseline_accuracy_case_weighted": case_weighted["majority_baseline_accuracy"],
        "ess": _kish_effective_sample_size(weights),
        "name_distinctness": name_distinctness(business_names),
        "n_judgements": int(len(frame)),
        "n_unclear": int((pred == "unclear").sum()),
    }


def _aligned_arrays(
    judge_key: str, log: str, partition: str, weights_map: dict[str, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index] | None:
    frame = _load_judgement_frame(judge_key, log, partition)
    if frame is None:
        return None
    # Sort by variant_id (B1): row order otherwise follows the per-partition
    # judge-batch shuffle, so two partitions covering the identical variant
    # set would never satisfy ``index.equals`` under insertion order alone.
    frame = (
        frame.drop_duplicates("variant_id")
        .set_index("variant_id")
        .sort_index()
    )
    weights = np.array([weights_map[v] for v in frame.index], dtype=np.float64)
    return (
        frame["true_name_code"].to_numpy(),
        frame["chosen_name_code"].to_numpy(),
        weights,
        frame.index,
    )


def _random_mean_diff_samples(
    true_p7: np.ndarray,
    pred_p7: np.ndarray,
    weights: np.ndarray,
    random_frames: list[tuple[np.ndarray, np.ndarray]],
    indices: np.ndarray,
) -> tuple[float, np.ndarray, int]:
    """kappa(P7) minus the mean of kappa(random_i) over i=1..3 (B2).

    Computed per bootstrap resample: for each resample, kappa is recomputed
    for P7 and for every available random partition on that same resample,
    then P7's kappa minus the mean of the randoms' kappas is one sample.
    ``random_frames`` entries are (true, pred) arrays already aligned to the
    same variant order as ``true_p7``/``weights``.
    """
    point = case_weighted_cohen_kappa(true_p7, pred_p7, weights) - float(
        np.mean([case_weighted_cohen_kappa(t, p, weights) for t, p in random_frames])
    )
    samples: list[float] = []
    dropped = 0
    for subset in indices:
        if np.unique(true_p7[subset]).size < 2:
            dropped += 1
            continue
        random_kappas = []
        skip = False
        for true_r, pred_r in random_frames:
            if np.unique(true_r[subset]).size < 2:
                skip = True
                break
            random_kappas.append(
                case_weighted_cohen_kappa(true_r[subset], pred_r[subset], weights[subset])
            )
        if skip:
            dropped += 1
            continue
        kappa_p7_subset = case_weighted_cohen_kappa(
            true_p7[subset], pred_p7[subset], weights[subset]
        )
        samples.append(kappa_p7_subset - float(np.mean(random_kappas)))
    return point, np.asarray(samples), dropped


def _check_incomplete_judgements(allow_incomplete: bool) -> list[dict[str, Any]]:
    """B6: ``analyze`` refuses incomplete judged data unless explicitly allowed."""
    incomplete: list[dict[str, Any]] = []
    for judge_key in JUDGES:
        for log in LOG_SPECS:
            summary_path = OUTPUT / "judgements" / judge_key / log / "_summary.json"
            if not summary_path.exists():
                continue
            log_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if log_summary.get("incomplete"):
                incomplete.append({"judge": judge_key, "log": log, **log_summary})
    if incomplete and not allow_incomplete:
        details = "; ".join(
            f"{row['judge']}/{row['log']} "
            f"({row['incomplete_fraction']:.1%} excluded)"
            for row in incomplete
        )
        raise RuntimeError(
            "analyze refused: incomplete judged data exceeds the "
            f"{INCOMPLETE_DATA_THRESHOLD:.0%} paired-exclusion threshold for "
            f"{details}. Re-run pass with --allow-incomplete to proceed anyway."
        )
    return incomplete


def run_analyze(*, allow_incomplete: bool = False) -> None:
    incomplete_logs = _check_incomplete_judgements(allow_incomplete)
    separability_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []
    judge_agreement_rows: list[dict[str, Any]] = []
    decision: dict[str, Any] = {"h_n1": {}, "h_n2": {}, "h_n3": {}}
    primary_judge_for_ess = "deepseek_v4_pro"
    log_ess: dict[str, float] = {}

    for judge_key in JUDGES:
        for log in LOG_SPECS:
            weights_map = _variant_weights(log)
            available = [
                p
                for p in NAMED_PARTITIONS
                if _load_judgement_frame(judge_key, log, p) is not None
            ]
            for partition in available:
                row = separability_row(judge_key, log, partition)
                if row is not None:
                    separability_rows.append(row)
            if "p7" not in available:
                continue
            true_p7, pred_p7, case_weights, index = _aligned_arrays(
                judge_key, log, "p7", weights_map
            )
            indices = resample_indices(len(case_weights), BOOTSTRAP_RESAMPLES, seed=SEED)
            # Amendment 2: unweighted variant-level kappa (each held-out
            # variant counted once) is PRIMARY for H-N1/H-N2/H-N3; the
            # existing case-weighted metrics are secondary/descriptive,
            # reported alongside each log's Kish ESS (see separability_row).
            unweighted_weights = np.ones_like(case_weights)
            weight_schemes = (
                ("unweighted", unweighted_weights),
                ("case_weighted", case_weights),
            )
            if judge_key == primary_judge_for_ess:
                log_ess[log] = _kish_effective_sample_size(case_weights)

            def _build_interval(
                point: float,
                samples: np.ndarray,
                dropped: int,
                requested: int,
                confidence: float,
                *,
                weighting: str,
                comparison: str,
            ) -> dict[str, Any]:
                interval = _percentile_interval(
                    point,
                    samples,
                    confidence,
                    dropped_resamples=dropped,
                    requested_resamples=requested,
                )
                interval["class"] = _classify(interval["ci_lower"], interval["ci_upper"])
                interval["weighting"] = weighting
                # low_information is already set by _percentile_interval.
                if interval["dropped_resamples_fraction"] > DROPPED_RESAMPLE_WARN_FRACTION:
                    # Identify the exact log x judge x comparison x weighting
                    # cell that hit this, including the fully-undetermined
                    # (100% dropped) case, so it is traceable in the run log.
                    print(
                        f"[analyze] WARNING: {judge_key}/{log}/{comparison} ({weighting}) "
                        f"dropped {interval['dropped_resamples']}/{interval['resamples_requested']} "
                        f"({interval['dropped_resamples_fraction']:.1%}) bootstrap resamples for a "
                        f"single-class subset{' -- UNDETERMINED (0 valid resamples)' if interval['class'] == 'undetermined' else ''}",
                        flush=True,
                    )
                return interval

            def _diff(weights: np.ndarray, weighting: str, other: str, label: str) -> None:
                aligned = _aligned_arrays(judge_key, log, other, weights_map)
                if aligned is None:
                    return
                true_o, pred_o, weights_o, index_o = aligned
                weights_o_scheme = np.ones_like(weights_o) if weighting == "unweighted" else weights_o
                if not index.equals(index_o):
                    # Partitions cover different variant sets (e.g. the
                    # paired-exclusion rule dropped different variants before
                    # this fix, or a partial run). Restrict to the common
                    # variants and pull each side's OWN true/pred codes (B1)
                    # -- never reuse P7's true codes for the other partition.
                    common = index.intersection(index_o)
                    true_common_p7 = pd.Series(true_p7, index=index).loc[common].to_numpy()
                    pred_common_p7 = pd.Series(pred_p7, index=index).loc[common].to_numpy()
                    true_common_o = pd.Series(true_o, index=index_o).loc[common].to_numpy()
                    pred_common_o = pd.Series(pred_o, index=index_o).loc[common].to_numpy()
                    w_common = pd.Series(weights, index=index).loc[common].to_numpy()
                    diff_indices = resample_indices(len(common), BOOTSTRAP_RESAMPLES, seed=SEED)
                    point, samples, dropped = _kappa_diff_samples(
                        true_common_p7, pred_common_p7, true_common_o, pred_common_o,
                        w_common, diff_indices,
                    )
                else:
                    diff_indices = indices
                    point, samples, dropped = _kappa_diff_samples(
                        true_p7, pred_p7, true_o, pred_o, weights, indices
                    )
                nominal = _build_interval(
                    point, samples, dropped, diff_indices.shape[0], NOMINAL_CONFIDENCE,
                    weighting=weighting, comparison=f"p7_minus_{label}",
                )
                bonferroni = _build_interval(
                    point, samples, dropped, diff_indices.shape[0], BONFERRONI_CONFIDENCE,
                    weighting=weighting, comparison=f"p7_minus_{label}",
                )
                difference_rows.append(
                    {
                        "judge": judge_key,
                        "log": log,
                        "comparison": f"p7_minus_{label}",
                        "weighting": weighting,
                        **{f"nominal_{k}": v for k, v in nominal.items()},
                        **{f"bonferroni_{k}": v for k, v in bonferroni.items()},
                    }
                )

            for weighting, scheme_weights in weight_schemes:
                _diff(scheme_weights, weighting, "raw", "raw")
                _diff(scheme_weights, weighting, "a2v", "a2v")

            random_available = [p for p in ("random1", "random2", "random3") if p in available]
            if random_available:
                random_frames: list[tuple[np.ndarray, np.ndarray]] = []
                for random_partition in random_available:
                    aligned = _aligned_arrays(judge_key, log, random_partition, weights_map)
                    if aligned is None:
                        continue
                    true_r, pred_r, weights_r, index_r = aligned
                    if not index.equals(index_r):
                        print(
                            f"[analyze] WARNING: {judge_key}/{log}/{random_partition} "
                            "judged sample does not match p7's; excluded from the "
                            "p7_minus_random mean",
                            flush=True,
                        )
                        continue
                    random_frames.append((true_r, pred_r))
                if random_frames:
                    # Amendment 1b: H-N1 needs all 3 random draws present for
                    # a log; n_random_partitions < 3 is flagged "incomplete"
                    # (not informative/uninformative) downstream.
                    for weighting, scheme_weights in weight_schemes:
                        point, samples, dropped = _random_mean_diff_samples(
                            true_p7, pred_p7, scheme_weights, random_frames, indices
                        )
                        nominal = _build_interval(
                            point, samples, dropped, indices.shape[0], NOMINAL_CONFIDENCE,
                            weighting=weighting, comparison="p7_minus_random",
                        )
                        bonferroni = _build_interval(
                            point, samples, dropped, indices.shape[0], BONFERRONI_CONFIDENCE,
                            weighting=weighting, comparison="p7_minus_random",
                        )
                        difference_rows.append(
                            {
                                "judge": judge_key,
                                "log": log,
                                "comparison": "p7_minus_random",
                                "weighting": weighting,
                                "n_random_partitions": len(random_frames),
                                **{f"nominal_{k}": v for k, v in nominal.items()},
                                **{f"bonferroni_{k}": v for k, v in bonferroni.items()},
                            }
                        )

        # Judge agreement (kappa between the two judges' choices), per log x
        # partition. One-directional only (S5): JUDGES has exactly the
        # primary and secondary judge, so compute each unordered pair once.
        for other_judge in JUDGES:
            if JUDGES.index(other_judge) <= JUDGES.index(judge_key):
                continue
            for log in LOG_SPECS:
                for partition in NAMED_PARTITIONS:
                    a = _load_judgement_frame(judge_key, log, partition)
                    b = _load_judgement_frame(other_judge, log, partition)
                    if a is None or b is None:
                        continue
                    merged = a.set_index("variant_id")[["chosen_name_code"]].join(
                        b.set_index("variant_id")[["chosen_name_code"]],
                        lsuffix="_a",
                        rsuffix="_b",
                        how="inner",
                    )
                    if merged.empty:
                        continue
                    weights_map = _variant_weights(log)
                    weights = np.array(
                        [weights_map[v] for v in merged.index], dtype=np.float64
                    )
                    judge_agreement_rows.append(
                        {
                            "log": log,
                            "partition": partition,
                            "judge_a": judge_key,
                            "judge_b": other_judge,
                            "kappa": case_weighted_cohen_kappa(
                                merged["chosen_name_code_a"].to_numpy(),
                                merged["chosen_name_code_b"].to_numpy(),
                                weights,
                            ),
                        }
                    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(separability_rows).to_csv(OUTPUT / "separability.csv", index=False)
    differences = pd.DataFrame(difference_rows)
    if "n_random_partitions" not in differences.columns:
        # No log in this run produced a p7_minus_random row (e.g. a fixture
        # or a partial run with zero random partitions judged) -- keep the
        # column present so downstream Amendment 1b filtering is safe.
        differences["n_random_partitions"] = 0
    differences.to_csv(OUTPUT / "separability_differences.csv", index=False)
    pd.DataFrame(judge_agreement_rows).to_csv(
        OUTPUT / "judge_agreement.csv", index=False
    )

    # H-N1: P7 vs size-matched random, every log, primary judge. Amendment 2:
    # unweighted variant-level kappa is the primary metric. Amendment 1b: a
    # log needs all 3 random draws present, else it is "incomplete" rather
    # than informative/uninformative.
    primary_judge = "deepseek_v4_pro"
    raw_diffs = differences[
        (differences["judge"] == primary_judge)
        & (differences["comparison"] == "p7_minus_random")
        & (differences["weighting"] == "unweighted")
    ]
    complete_diffs = raw_diffs[raw_diffs["n_random_partitions"] >= 3]
    h_n1_incomplete_logs = sorted(set(LOG_SPECS) - set(complete_diffs["log"]))
    h_n1_logs_passing = sorted(complete_diffs.loc[complete_diffs["nominal_class"] == "positive", "log"])
    # PI review: a 0-valid-resample ("undetermined") log is not a normal
    # null result -- it gets its own bucket, never "uninformative" (which
    # implies a defined-but-non-significant interval) and never "informative".
    h_n1_undetermined_logs = sorted(
        complete_diffs.loc[complete_diffs["nominal_class"] == "undetermined", "log"]
    )
    h_n1_logs_failing = sorted(
        set(complete_diffs["log"]) - set(h_n1_logs_passing) - set(h_n1_undetermined_logs)
    )
    h_n1_low_information_logs = sorted(complete_diffs.loc[complete_diffs["nominal_low_information"], "log"])
    raw_diffs_case_weighted = differences[
        (differences["judge"] == primary_judge)
        & (differences["comparison"] == "p7_minus_random")
        & (differences["weighting"] == "case_weighted")
    ]
    decision["h_n1"] = {
        "primary_metric": "unweighted_variant_level_kappa",
        "informative_logs": h_n1_logs_passing,
        "uninformative_logs": h_n1_logs_failing,
        "undetermined_logs": h_n1_undetermined_logs,
        "incomplete_logs": h_n1_incomplete_logs,
        "low_information_logs": h_n1_low_information_logs,
        "secondary_case_weighted": raw_diffs_case_weighted[
            ["log", "nominal_point_estimate", "nominal_class", "nominal_low_information"]
        ].to_dict(orient="records"),
    }

    # H-N2: P7 vs raw, primary judge Bonferroni-positive on >=3/5 logs, and
    # the secondary judge agrees in sign on those logs. Primary metric is
    # unweighted variant-level kappa (Amendment 2); H-N1's informative-logs
    # gate (now incomplete-aware, Amendment 1b) still applies.
    primary_raw = differences[
        (differences["judge"] == primary_judge)
        & (differences["comparison"] == "p7_minus_raw")
        & (differences["weighting"] == "unweighted")
    ].set_index("log")
    secondary_raw = differences[
        (differences["judge"] != primary_judge)
        & (differences["comparison"] == "p7_minus_raw")
        & (differences["weighting"] == "unweighted")
    ].set_index("log")
    primary_positive_logs = [
        log
        for log in primary_raw.index
        if log in h_n1_logs_passing and primary_raw.loc[log, "bonferroni_class"] == "positive"
    ]
    agreeing_logs = [
        log
        for log in primary_positive_logs
        if log in secondary_raw.index
        and np.sign(secondary_raw.loc[log, "nominal_point_estimate"])
        == np.sign(primary_raw.loc[log, "nominal_point_estimate"])
    ]
    supported = len(primary_positive_logs) >= 3 and len(agreeing_logs) == len(
        primary_positive_logs
    )
    decision["h_n2"] = {
        "primary_metric": "unweighted_variant_level_kappa",
        "primary_positive_logs": primary_positive_logs,
        "secondary_agreeing_logs": agreeing_logs,
        "supported": bool(supported),
        "claim": (
            f"P7 clusters were more identifiable from LLM-generated names than "
            f"raw-label clusters on {len(primary_positive_logs)} of 5 logs."
            if supported
            else "The naming evaluation does not separate P7 from the raw-label baseline."
        ),
    }

    # H-N3: descriptive, P7 vs Activity2Vec. Primary metric is unweighted
    # variant-level kappa (Amendment 2); case-weighted rows kept as
    # secondary/descriptive.
    a2v_rows = differences[
        (differences["comparison"] == "p7_minus_a2v") & (differences["weighting"] == "unweighted")
    ]
    a2v_rows_case_weighted = differences[
        (differences["comparison"] == "p7_minus_a2v") & (differences["weighting"] == "case_weighted")
    ]
    decision["h_n3"] = {
        "primary_metric": "unweighted_variant_level_kappa",
        "per_log": a2v_rows[
            ["judge", "log", "nominal_point_estimate", "nominal_class", "nominal_low_information"]
        ].to_dict(orient="records"),
        "per_log_case_weighted": a2v_rows_case_weighted[
            ["judge", "log", "nominal_point_estimate", "nominal_class", "nominal_low_information"]
        ].to_dict(orient="records"),
    }

    # Amendment 2: Kish effective sample size per log (frequency-only, no
    # LLM output), quantifying how many "independent" judgements the
    # case-weighted/secondary metrics actually rest on.
    decision["ess_by_log"] = log_ess

    decision["data_completeness"] = {
        "allow_incomplete": allow_incomplete,
        "incomplete_logs": incomplete_logs,
    }

    (OUTPUT / "decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


# ---------------------------------------------------------------------------
# Manifest: aggregate call/retry/repair stats straight from the cache
# ---------------------------------------------------------------------------


def _scan_model_cache(cache_dir: Path) -> dict[str, Any]:
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(cache_dir.glob("**/*.json"))
    ]
    returned_models: set[str] = set()
    total_provider_attempts = 0
    total_validation_attempts = 0
    total_retry_after_waits = 0
    repaired_calls = 0
    failed = 0
    prompt_shas: set[str] = set()
    schema_shas: set[str] = set()
    for record in records:
        meta = record.get("provider_metadata", {})
        if meta.get("provider_model"):
            returned_models.add(meta["provider_model"])
        total_provider_attempts += int(meta.get("attempt_count", 1))
        request_metadata = meta.get("request_metadata", {})
        validation_attempts = int(request_metadata.get("validation_attempt_count", 1))
        total_validation_attempts += validation_attempts
        total_retry_after_waits += int(request_metadata.get("retry_after_waits", 0) or 0)
        if validation_attempts > 1:
            repaired_calls += 1
        if record.get("status") != "validated":
            failed += 1
        prompt_shas.add(record.get("prompt_sha256", ""))
        schema_shas.add(record.get("schema_sha256", ""))
    return {
        "call_count": len(records),
        "returned_models": sorted(returned_models),
        "total_provider_attempts": total_provider_attempts,
        "total_validation_attempts": total_validation_attempts,
        "retry_after_waits": total_retry_after_waits,
        "repaired_calls": repaired_calls,
        "failed_calls": failed,
        "prompt_sha256": sorted(prompt_shas),
        "schema_sha256": sorted(schema_shas),
    }


def build_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "prespec_sha256": (
            _file_sha256(OUTPUT / "PRESPEC.md") if (OUTPUT / "PRESPEC.md").exists() else None
        ),
        "seed": SEED,
        "random_partition_seeds": RANDOM_SEEDS,
        "heldout_cap_per_log": 300,
        "judge_batch_size": 10,
        "api_version": API_VERSION,
        "models": {},
    }
    model_cache_keys = {
        "namer_gpt41": "namer_gpt41",
        "deepseek_v4_pro": "deepseek_v4_pro",
        "gpt4o": "gpt4o",
    }
    for model_key, cache_subdir in model_cache_keys.items():
        cache_dir = OUTPUT / "cache" / cache_subdir
        model_cfg = MODELS[model_key]
        entry = {
            "endpoint": model_cfg["endpoint"],
            "deployment": model_cfg["deployment"],
            "expected_model": model_cfg["expected_model"],
        }
        if cache_dir.exists():
            entry.update(_scan_model_cache(cache_dir))
        else:
            entry.update({"call_count": 0, "returned_models": []})
        manifest["models"][model_key] = entry

    # B5: startup smoke-probe results (expected vs. served model per role).
    model_check_path = OUTPUT / _MODEL_CHECK_PATH_NAME
    manifest["model_check"] = (
        json.loads(model_check_path.read_text(encoding="utf-8"))
        if model_check_path.exists()
        else None
    )

    # B6: paired-exclusion / incomplete-data stats per judge+log.
    excluded_variants: dict[str, Any] = {}
    for judge_key in JUDGES:
        for log in LOG_SPECS:
            summary_path = OUTPUT / "judgements" / judge_key / log / "_summary.json"
            if summary_path.exists():
                excluded_variants[f"{judge_key}/{log}"] = json.loads(
                    summary_path.read_text(encoding="utf-8")
                )
    manifest["excluded_variants"] = excluded_variants

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_name_dry_run_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("[dry-run][name] 0 calls (no units matched)")
        return
    total_input = sum(int(r["estimated_input_tokens"]) for r in rows)
    total_output = sum(int(r["estimated_output_tokens"]) for r in rows)
    print(
        f"[dry-run][name] {len(rows)} calls, "
        f"~{total_input:,} estimated input tokens, "
        f"~{total_output:,} estimated output tokens (max_tokens budget)"
    )


def _print_judge_dry_run_summary(summary: dict[str, Any]) -> None:
    all_rows = [
        row
        for log_summary in summary.get("logs", {}).values()
        for row in log_summary.get("dry_run_batches", [])
    ]
    if not all_rows:
        print(f"[dry-run][judge:{summary.get('judge')}] 0 calls (no units matched)")
        return
    total_input = sum(int(r["estimated_input_tokens"]) for r in all_rows)
    total_output = sum(int(r["estimated_output_tokens"]) for r in all_rows)
    print(
        f"[dry-run][judge:{summary.get('judge')}] {len(all_rows)} calls, "
        f"~{total_input:,} estimated input tokens, "
        f"~{total_output:,} estimated output tokens (max_tokens budget). "
        "Note: candidate names are placeholders wherever a real `name` run "
        "hasn't happened yet, so input-token estimates for those "
        "partitions are approximate."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_profile = sub.add_parser("profile", help="Compute deterministic cluster profiles")
    p_profile.add_argument("--logs", nargs="+", choices=list(LOG_SPECS))

    p_pipeline = sub.add_parser(
        "pipeline-check", help="P7 vs frozen ARI check (run before any LLM call)"
    )
    del p_pipeline

    p_name = sub.add_parser("name", help="Generate business names (LLM calls)")
    p_name.add_argument("--logs", nargs="+", choices=list(LOG_SPECS))
    p_name.add_argument(
        "--partitions",
        nargs="+",
        choices=list(NAMED_PARTITIONS),
        help="Restrict to specific partitions (default: all 7); use for pilots",
    )
    p_name.add_argument(
        "--dry-run",
        action="store_true",
        help="Render prompts and estimate tokens/calls without calling the provider",
    )
    p_name.add_argument(
        "--confirm-prespec",
        metavar="SHA256",
        help="SHA-256 of the currently-read PRESPEC.md (B8 guard); required for real calls",
    )

    p_judge = sub.add_parser("judge", help="Judge held-out variants (LLM calls)")
    p_judge.add_argument("--judge", required=True, choices=("deepseek_v4_pro", "gpt4o"))
    p_judge.add_argument("--logs", nargs="+", choices=list(LOG_SPECS))
    p_judge.add_argument(
        "--partitions",
        nargs="+",
        choices=list(NAMED_PARTITIONS),
        help="Restrict to specific partitions (default: all with real names on disk); use for pilots",
    )
    p_judge.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override worker count; capped at the model's B6 concurrency limit",
    )
    p_judge.add_argument(
        "--dry-run",
        action="store_true",
        help="Render prompts and estimate tokens/calls without calling the provider",
    )
    p_judge.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Proceed even if paired-exclusion drops > 5%% of a log's judged sample",
    )
    p_judge.add_argument(
        "--confirm-prespec",
        metavar="SHA256",
        help="SHA-256 of the currently-read PRESPEC.md (B8 guard); required for real calls",
    )

    p_analyze = sub.add_parser("analyze", help="Compute separability metrics and decision.json")
    p_analyze.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Proceed even if any judge/log was marked incomplete during judging",
    )
    sub.add_parser("manifest", help="Rebuild manifest.json from the on-disk cache")

    args = parser.parse_args()
    if args.command == "pipeline-check":
        frame = run_pipeline_check()
        print(frame.to_string(index=False))
    elif args.command == "profile":
        run_profile_step(args.logs)
    elif args.command == "name":
        result = run_name_step(
            args.logs,
            partitions=args.partitions,
            dry_run=args.dry_run,
            confirm_prespec=args.confirm_prespec,
        )
        if args.dry_run:
            _print_name_dry_run_summary(result)
    elif args.command == "judge":
        result = run_judge_step(
            args.judge,
            args.logs,
            workers=args.workers,
            partitions=args.partitions,
            dry_run=args.dry_run,
            allow_incomplete=args.allow_incomplete,
            confirm_prespec=args.confirm_prespec,
        )
        if args.dry_run:
            _print_judge_dry_run_summary(result)
    elif args.command == "analyze":
        run_analyze(allow_incomplete=args.allow_incomplete)
    elif args.command == "manifest":
        build_manifest()
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
