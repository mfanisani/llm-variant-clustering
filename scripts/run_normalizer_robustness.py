"""Normalizer and embedder robustness for the paper-1 P7 minus raw contrasts.

Pre-specification: ``evaluation/normalizer_robustness_v1/PRESPEC.md``.

``normalize`` re-runs the frozen P7 prompt and schema through one alternative
normalizing model on every activity label of the five public logs. ``analyze``
rebuilds the ordered P7 pyramid from each model's descriptors, clusters at the
matched K, and re-computes the paired P7 minus raw silhouette bootstrap with
the identical resamples used in ``statistical_uncertainty_v2``. It also swaps
the embedding model for both sides. Frozen artifacts are read only.
"""

from __future__ import annotations

import os

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    pairwise_distances,
)

from llm_variant_clustering.clustering import weighted_kmeans
from llm_variant_clustering.evaluation import evaluate_partition
from llm_variant_clustering.features import parse_sequences
from llm_variant_clustering.llm_cache import (
    GenerationParameters,
    ModelIdentity,
    StructuredResponseCache,
    canonical_json_hash,
    execute_cached_request,
    request_cache_key,
)
from llm_variant_clustering.prompts import render_prompt
from llm_variant_clustering.semantic_compiler import (
    canonical_activity_text,
    ordered_semantic_pyramid,
    validate_activity_semantics,
)
from llm_variant_clustering.uncertainty import paired_silhouette_difference
from run_representation_source_benchmark import _compact_activity_space, _embed
from run_statistical_uncertainty import DATASET_FOLDERS

ROOT = Path(__file__).parents[1]
PROMPT = ROOT / "configs/frozen/p7_activity_gpt41_v1/prompt.txt"
SCHEMA = ROOT / "configs/frozen/p7_activity_gpt41_v1/schema.json"
OUTPUT = ROOT / "evaluation/normalizer_robustness_v1"
ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
API_VERSION = "2025-01-01-preview"
CONFIDENCE = 0.95
RESAMPLES = 400
CAP = 2500
FIELDS = (
    "business_object",
    "action",
    "process_stage",
    "explicit_state",
    "explicit_role",
)

# dataset, matched K, frozen partition suffix, has paired raw contrast
LOGS = (
    ("bpi2020", 9, "dynamic", False),
    ("bpi2017", 2, "dynamic", True),
    ("bpi2019", 6, "dynamic", True),
    ("sepsis", 8, "k8", True),
    ("rtfm", 2, "dynamic", True),
)
FROZEN_CLASS = {
    "bpi2017": "positive",
    "bpi2019": "includes_zero",
    "sepsis": "includes_zero",
    "rtfm": "positive",
}

NORMALIZERS = {
    "gpt41_rerun": {
        "kind": "azure",
        "deployment": "gpt-4.1",
        "model_version": "2025-04-14",
        "expected_model": "gpt-4.1-2025-04-14",
        "decision_rule": True,
    },
    "gpt41_mini": {
        "kind": "azure",
        "deployment": "gpt-4.1-mini",
        "model_version": "2025-04-14",
        "expected_model": "gpt-4.1-mini-2025-04-14",
        "decision_rule": True,
    },
    "gpt4o": {
        "kind": "azure",
        "deployment": "gpt-4o-2024-08-06-global",
        "model_version": "2024-08-06",
        "expected_model": "gpt-4o-2024-08-06",
        "decision_rule": True,
    },
    "qwen25_7b": {
        "kind": "local",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "model_version": "a09a35458c702b33eeacc393d103063234e8bc28",
        "expected_model": "Qwen/Qwen2.5-7B-Instruct",
        "decision_rule": True,
    },
    "deepseek_v4_pro": {
        "kind": "azure",
        "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT_SECONDARY") or os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        "deployment": "deepseek-v4-pro",
        "model_version": "2026-04-23",
        "expected_model": "deepseek-v4-pro",
        "decision_rule": True,
    },
    "gpt5_minimal": {
        "kind": "azure_reasoning",
        "deployment": "gpt-5",
        "model_version": "2025-08-07;reasoning_effort=minimal",
        "expected_model": "gpt-5-2025-08-07",
        "reasoning_effort": "minimal",
        "decision_rule": False,
    },
}

EMBEDDERS = {
    "te3_small": {"kind": "azure", "deployment": "text-embedding-3-small"},
    "mpnet": {
        "kind": "local",
        "model_id": "sentence-transformers/all-mpnet-base-v2",
        "revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
    },
}


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


def _all_labels() -> list[tuple[str, str]]:
    rows = []
    for dataset, *_ in LOGS:
        index = pd.read_parquet(
            ROOT / "evaluation/full_benchmark" / dataset / "activity_embedding_index.parquet"
        )
        rows.extend((dataset, str(label)) for label in index["activity_label"])
    return rows


def _reasoning_shim(provider: object, reasoning_effort: str) -> None:
    """Adapt the strict-tool provider to a reasoning model.

    Reasoning deployments reject ``temperature`` and ``max_tokens``. Only the
    request keywords change; parsing and validation are the provider's own.
    """
    create = provider.client.chat.completions.create  # type: ignore[attr-defined]

    def patched(**kwargs: object) -> object:
        kwargs.pop("temperature", None)
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        kwargs["reasoning_effort"] = reasoning_effort
        return create(**kwargs)

    provider.client = SimpleNamespace(  # type: ignore[attr-defined]
        chat=SimpleNamespace(completions=SimpleNamespace(create=patched))
    )


def _local_prompt(artifact: object) -> object:
    schema_text = SCHEMA.read_text(encoding="utf-8")
    rendered = (
        artifact.rendered_prompt  # type: ignore[attr-defined]
        + "\n\nThe output must exactly match this JSON Schema:\n"
        + schema_text
    )
    return replace(
        artifact,
        rendered_prompt=rendered,
        prompt_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    )


def normalize(key: str, workers: int) -> None:
    spec = NORMALIZERS[key]
    out_dir = OUTPUT / "normalizers" / key
    cache = StructuredResponseCache(out_dir / "cache")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    kind = spec["kind"]

    if kind == "local":
        from llm_variant_clustering.local_provider import LocalTransformersProvider
        from run_second_llm_robustness import _generate_with_repairs as repair_loop

        parameters = GenerationParameters(temperature=0.0, seed=None, max_output_tokens=400)
        identity = ModelIdentity(
            provider="local_transformers_robustness",
            model=spec["model_id"],
            model_version=spec["model_version"],
        )
        shared = LocalTransformersProvider(
            model_id=spec["model_id"],
            model_revision=spec["model_version"],
            parameters=parameters,
            cache_dir=ROOT / "models/huggingface",
        )
        make_provider = lambda: shared  # noqa: E731
        workers = 1
    else:
        from llm_variant_clustering.azure_openai_provider import AzureOpenAIToolProvider
        from run_azure_prompt_pilot import generate_with_validation_repairs as repair_loop

        max_tokens = 4000 if kind == "azure_reasoning" else 1000
        parameters = GenerationParameters(temperature=0.0, seed=None, max_output_tokens=max_tokens)
        identity = ModelIdentity(
            provider="azure_openai_entra",
            model=spec["deployment"],
            model_version=spec["model_version"],
        )

        def make_provider() -> object:
            provider = AzureOpenAIToolProvider(
                endpoint=spec.get("endpoint", ENDPOINT),
                deployment=spec["deployment"],
                api_version=API_VERSION,
                parameters=parameters,
                schema=schema,
                schema_name="activity_semantics_p7",
                max_attempts=6,
            )
            if kind == "azure_reasoning":
                _reasoning_shim(provider, spec["reasoning_effort"])
            return provider

    def run_one(item: tuple[str, str]) -> dict[str, object]:
        dataset, label = item
        artifact = render_prompt(
            PROMPT, SCHEMA, {"ACTIVITY_LABEL": label}, prompt_version="p7_activity"
        )
        if kind == "local":
            artifact = _local_prompt(artifact)
        validator = partial(
            validate_activity_semantics,
            schema_path=SCHEMA,
            expected_activity_label=label,
        )
        input_sha = canonical_json_hash(label)
        row: dict[str, object] = {"dataset": dataset, "activity_label": label}
        try:
            response, hit = execute_cached_request(
                cache=cache,
                model=identity,
                parameters=parameters,
                prompt=artifact,
                input_sha256=input_sha,
                generate=partial(repair_loop, provider=make_provider(), validator=validator),
                validate=validator,
            )
        except Exception as error:  # recorded, not swallowed: see status column
            row.update(
                status="failed",
                response_json=None,
                provider_model=None,
                validation_attempts=None,
                error=f"{type(error).__name__}: {str(error)[:400]}",
            )
            return row
        record = cache.load(
            request_cache_key(identity, parameters, artifact, input_sha256=input_sha)
        )
        meta = (record or {}).get("provider_metadata", {})
        returned = meta.get("provider_model") or spec["expected_model"]
        if not str(returned).startswith(spec["expected_model"]):
            raise RuntimeError(
                f"{key}: deployment returned {returned!r}, expected {spec['expected_model']!r}"
            )
        row.update(
            status="validated",
            cache_hit=hit,
            response_json=json.dumps(response, ensure_ascii=False, separators=(",", ":")),
            provider_model=returned,
            validation_attempts=(meta.get("request_metadata") or {}).get(
                "validation_attempt_count"
            ),
            error=None,
        )
        return row

    items = _all_labels()
    print(f"[{key}] normalizing {len(items)} labels with {workers} worker(s)", flush=True)
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, row in enumerate(pool.map(run_one, items), start=1):
            rows.append(row)
            if count % 10 == 0 or count == len(items):
                failed = sum(r["status"] == "failed" for r in rows)
                print(f"[{key}] {count}/{len(items)} done, {failed} failed", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(out_dir / "activity_semantics.parquet", index=False)
    (out_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "key": key,
                "spec": spec,
                "parameters": parameters.__dict__,
                "prompt_sha256": hashlib.sha256(PROMPT.read_bytes()).hexdigest(),
                "schema_sha256": hashlib.sha256(SCHEMA.read_bytes()).hexdigest(),
                "label_count": len(frame),
                "validated": int(frame["status"].eq("validated").sum()),
                "returned_models": sorted(
                    {str(m) for m in frame["provider_model"].dropna()}
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(frame["status"].value_counts().to_string())


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def _log_frame(dataset: str) -> tuple[pd.DataFrame, Path]:
    bench = ROOT / "evaluation/full_benchmark" / dataset
    variants = pd.read_parquet(
        ROOT / "datasets/processed" / DATASET_FOLDERS[dataset] / "variants.parquet"
    )
    index_path = bench / "embedding_index.parquet"
    if index_path.exists():
        order = pd.read_parquet(index_path)["variant_id"].astype(str)
        variants = (
            variants.assign(variant_id=variants["variant_id"].astype(str))
            .set_index("variant_id")
            .loc[order]
            .reset_index()
        )
    return variants.reset_index(drop=True), bench


def _partition(bench: Path, method: str, suffix: str, order: list[str]) -> np.ndarray:
    frame = pd.read_parquet(bench / f"assignments_{method}_{suffix}.parquet")
    lookup = frame.set_index(frame["variant_id"].astype(str))["cluster_label"]
    return lookup.loc[order].to_numpy()


def _fields(descriptor: str) -> dict[str, str]:
    """Parse ``object: x; action: y; stage: z; state: w; role: v``."""
    names = dict(zip(("object", "action", "stage", "state", "role"), FIELDS, strict=True))
    parsed = {}
    for part in descriptor.split(";"):
        name, _, value = part.strip().partition(":")
        parsed[names[name.strip()]] = value.strip().lower()
    return parsed


def _classify(lower: float, upper: float) -> str:
    if lower > 0:
        return "positive"
    if upper < 0:
        return "negative"
    return "includes_zero"


def _pyramid(sequences: list, labels: list[str], vectors: np.ndarray) -> np.ndarray:
    compact, _ = _compact_activity_space(vectors)
    return ordered_semantic_pyramid(sequences, dict(zip(labels, compact, strict=True)))


def _cluster(matrix: np.ndarray, weights: np.ndarray, k: int) -> tuple[np.ndarray, dict]:
    from scipy.sparse import csr_matrix

    distances = pairwise_distances(matrix, metric="cosine")
    sparse = csr_matrix(matrix)  # same container as embedding_feature_set
    result = weighted_kmeans(sparse, weights, n_clusters=k)
    metrics = evaluate_partition(
        sparse, result.labels, weights, metric="cosine", distances=distances
    )
    return result.labels, {
        "case_weighted_silhouette": metrics["case_weighted_silhouette"],
        "stability_ari": result.mean_seed_stability_ari,
        "distances": distances,
    }


def _paired(p7_dist, p7_labels, raw_dist, raw_labels, weights) -> dict[str, object]:
    sample = CAP if len(weights) > CAP else None
    diff = paired_silhouette_difference(
        p7_dist, p7_labels, raw_dist, raw_labels, weights,
        resamples=RESAMPLES, sample_size=sample,
    )
    adjusted = diff.adjusted_interval(1.0 - (1.0 - CONFIDENCE) / len(FROZEN_CLASS))
    row = diff.as_dict("difference_")
    row.update(
        bonferroni_ci_lower=adjusted.lower,
        bonferroni_ci_upper=adjusted.upper,
        bonferroni_class=_classify(adjusted.lower, adjusted.upper),
        bootstrap_sample_size=sample or len(weights),
    )
    return row


class _Embedders:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}
        self.cache_dir = OUTPUT / "embedders" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._local = {}
        self._azure = None
        self._records_path = self.cache_dir / "records.json"
        if self._records_path.exists():
            self.records = json.loads(self._records_path.read_text(encoding="utf-8"))

    def bge(self, texts: list[str]) -> np.ndarray:
        return _embed(texts, ROOT / "models/huggingface")

    def __call__(self, key: str, texts: list[str]) -> np.ndarray:
        digest = hashlib.sha256(json.dumps([key, texts]).encode("utf-8")).hexdigest()
        path = self.cache_dir / f"{key}_{digest[:24]}.npy"
        spec = EMBEDDERS[key]
        if key not in self.records:
            # Record the served model even when every vector comes from cache.
            if spec["kind"] == "azure":
                self._azure_embed(spec["deployment"], ["probe"], key)
            else:
                self.records[key] = {"model_id": spec["model_id"], "revision": spec["revision"]}
            self._records_path.write_text(json.dumps(self.records, indent=2) + "\n", encoding="utf-8")
        if path.exists():
            return np.load(path)
        if spec["kind"] == "azure":
            vectors = self._azure_embed(spec["deployment"], texts, key)
        else:
            vectors = self._local_embed(spec["model_id"], spec["revision"], texts)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors.astype(np.float32)
        np.save(path, vectors)
        return vectors

    def _azure_embed(self, deployment: str, texts: list[str], key: str) -> np.ndarray:
        if self._azure is None:
            from azure.identity import AzureCliCredential, get_bearer_token_provider
            from openai import AzureOpenAI

            self._azure = AzureOpenAI(
                azure_endpoint=ENDPOINT,
                azure_ad_token_provider=get_bearer_token_provider(
                    AzureCliCredential(process_timeout=60),
                    "https://cognitiveservices.azure.com/.default",
                ),
                api_version=API_VERSION,
                timeout=120.0,
            )
        response = self._azure.embeddings.create(model=deployment, input=texts)
        self.records[key] = {"returned_model": response.model, "deployment": deployment}
        ordered = sorted(response.data, key=lambda item: item.index)
        return np.asarray([item.embedding for item in ordered], dtype=np.float64)

    def _local_embed(self, model_id: str, revision: str, texts: list[str]) -> np.ndarray:
        if model_id not in self._local:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                model_id,
                revision=revision,
                cache_folder=str(ROOT / "models/huggingface"),
                device="cpu",
                trust_remote_code=False,
            )
            self._local[model_id] = model
        model = self._local[model_id]
        lengths = model.tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]
        if max(len(ids) for ids in lengths) > model.max_seq_length:
            raise ValueError(f"{model_id}: an activity text would require truncation")
        return model.encode(texts, batch_size=16, convert_to_numpy=True).astype(np.float64)


def _semantics_map(key: str) -> dict[str, dict[str, object]]:
    path = OUTPUT / "normalizers" / key / "activity_semantics.parquet"
    frame = pd.read_parquet(path)
    ok = frame.loc[frame["status"].eq("validated")]
    return {
        (row.dataset, row.activity_label): json.loads(row.response_json)
        for row in ok.itertuples()
    }


def analyze(keys: list[str]) -> None:
    embed = _Embedders()
    v2 = pd.read_csv(
        ROOT / "evaluation/statistical_uncertainty_v2/paired_silhouette_differences.csv"
    ).set_index("dataset")
    norm_rows: list[dict[str, object]] = []
    field_rows: list[dict[str, object]] = []
    emb_rows: list[dict[str, object]] = []
    check_rows: list[dict[str, object]] = []
    semantics = {key: _semantics_map(key) for key in keys}

    for dataset, k, suffix, paired in LOGS:
        variants, bench = _log_frame(dataset)
        order = variants["variant_id"].astype(str).tolist()
        sequences = parse_sequences(variants)
        weights = variants["frequency"].to_numpy(dtype=np.float64)
        index = pd.read_parquet(bench / "activity_embedding_index.parquet")
        labels = index["activity_label"].astype(str).tolist()
        frozen_desc = dict(zip(labels, index["canonical_descriptor"].astype(str), strict=True))
        frozen_p7 = _partition(bench, "p7_ordered_pyramid_1_2_4_bge", suffix, order)

        # Pipeline check on the frozen descriptors.
        ref_matrix = _pyramid(sequences, labels, embed.bge([frozen_desc[x] for x in labels]))
        frozen_npy = np.load(
            bench / ("p7_ordered_embeddings.npy" if paired else "ordered_pyramid_embeddings.npy")
        )
        ref_labels, ref_metrics = _cluster(ref_matrix, weights, k)
        distance_error = float(
            np.abs(ref_metrics["distances"] - pairwise_distances(frozen_npy, metric="cosine")).max()
        )
        check = {
            "dataset": dataset,
            "frozen_dims": int(frozen_npy.shape[1]),
            "rebuilt_dims": int(ref_matrix.shape[1]),
            "max_cosine_distance_abs_error": distance_error,
            # PRESPEC amendment 1: compare geometry, not coordinates.
            "embeddings_allclose": distance_error <= 1e-5,
            "ari_rebuilt_vs_frozen": float(adjusted_rand_score(frozen_p7, ref_labels)),
        }
        raw_dist = raw_labels = None
        if paired:
            raw_npy = np.load(bench / "raw_ordered_embeddings.npy")
            raw_dist = pairwise_distances(raw_npy, metric="cosine")
            raw_labels = _partition(bench, "raw_activity_label_pyramid_bge", suffix, order)
            ref_row = _paired(
                pairwise_distances(frozen_npy, metric="cosine"), frozen_p7,
                raw_dist, raw_labels, weights,
            )
            check["paired_point_abs_error"] = abs(
                ref_row["difference_point_estimate"] - v2.loc[dataset, "difference_point_estimate"]
            )
            check["bonferroni_bounds_abs_error"] = max(
                abs(ref_row["bonferroni_ci_lower"] - v2.loc[dataset, "bonferroni_ci_lower"]),
                abs(ref_row["bonferroni_ci_upper"] - v2.loc[dataset, "bonferroni_ci_upper"]),
            )
            check["frozen_class"] = ref_row["bonferroni_class"]
        check_rows.append(check)
        print(f"[check] {check}", flush=True)
        if not check["embeddings_allclose"] or check["ari_rebuilt_vs_frozen"] != 1.0 or (
            paired and (check["paired_point_abs_error"] > 1e-6
                        or check["bonferroni_bounds_abs_error"] > 1e-6)
        ):
            pd.DataFrame(check_rows).to_csv(OUTPUT / "pipeline_check.csv", index=False)
            raise SystemExit(f"Pipeline check failed on {dataset}; see pipeline_check.csv")

        # Factor 1: normalizers.
        for key in keys:
            sem = semantics[key]
            validated = [x for x in labels if (dataset, x) in sem]
            descriptors = [
                canonical_activity_text(sem[(dataset, x)]) if (dataset, x) in sem
                else frozen_desc[x]
                for x in labels
            ]
            for x, desc in zip(labels, descriptors, strict=True):
                field_rows.append(
                    {
                        "normalizer": key,
                        "dataset": dataset,
                        "activity_label": x,
                        "validated": (dataset, x) in sem,
                        "frozen_descriptor": frozen_desc[x],
                        "model_descriptor": desc,
                        "identical": desc == frozen_desc[x],
                        **{
                            f"{field}_agrees": (
                                _fields(desc)[field] == _fields(frozen_desc[x])[field]
                            ) if (dataset, x) in sem else None
                            for field in FIELDS
                        },
                    }
                )
            matrix = _pyramid(sequences, labels, embed.bge(descriptors))
            model_labels, metrics = _cluster(matrix, weights, k)
            row: dict[str, object] = {
                "normalizer": key,
                "dataset": dataset,
                "status": "complete" if len(validated) == len(labels) else "partial_substitution",
                "activity_count": len(labels),
                "validated_count": len(validated),
                "descriptor_identity_rate": float(
                    np.mean([d == frozen_desc[x] for x, d in zip(labels, descriptors)])
                ),
                "matched_k": k,
                "ari_vs_frozen_p7": float(adjusted_rand_score(frozen_p7, model_labels)),
                "nmi_vs_frozen_p7": float(normalized_mutual_info_score(frozen_p7, model_labels)),
                "case_weighted_silhouette": metrics["case_weighted_silhouette"],
                "stability_ari": metrics["stability_ari"],
            }
            if paired:
                row.update(_paired(metrics["distances"], model_labels, raw_dist, raw_labels, weights))
                row["frozen_class"] = FROZEN_CLASS[dataset]
                row["preserves_frozen_class"] = row["bonferroni_class"] == FROZEN_CLASS[dataset]
            norm_rows.append(row)
            print(f"[{dataset}] {key} ari={row['ari_vs_frozen_p7']:.3f}", flush=True)

        # Factor 2: embedders (frozen GPT-4.1 descriptors and raw labels).
        if paired:
            raw_k = int(np.unique(raw_labels).size)
            for key in EMBEDDERS:
                p7_matrix = _pyramid(sequences, labels, embed(key, [frozen_desc[x] for x in labels]))
                raw_matrix = _pyramid(sequences, labels, embed(key, labels))
                p7_lab, p7_m = _cluster(p7_matrix, weights, k)
                raw_lab, raw_m = _cluster(raw_matrix, weights, raw_k)
                row = {
                    "embedder": key,
                    "dataset": dataset,
                    "p7_k": k,
                    "raw_k": raw_k,
                    "p7_silhouette": p7_m["case_weighted_silhouette"],
                    "raw_silhouette": raw_m["case_weighted_silhouette"],
                    "p7_ari_vs_frozen_p7": float(adjusted_rand_score(frozen_p7, p7_lab)),
                    **_paired(p7_m["distances"], p7_lab, raw_m["distances"], raw_lab, weights),
                    "frozen_class": FROZEN_CLASS[dataset],
                }
                row["preserves_frozen_class"] = row["bonferroni_class"] == FROZEN_CLASS[dataset]
                emb_rows.append(row)
                print(f"[{dataset}] embedder {key} diff={row['difference_point_estimate']:.3f}", flush=True)

    pd.DataFrame(check_rows).to_csv(OUTPUT / "pipeline_check.csv", index=False)
    norm = pd.DataFrame(norm_rows)
    norm.to_csv(OUTPUT / "normalizer_robustness.csv", index=False)
    pd.DataFrame(field_rows).to_csv(OUTPUT / "descriptor_agreement.csv", index=False)
    emb = pd.DataFrame(emb_rows)
    emb.to_csv(OUTPUT / "embedder_robustness.csv", index=False)

    decision = _decide(norm, emb)
    (OUTPUT / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "ancillary_sensitivity_after_primary_artifact_lock",
                "prespecification": "PRESPEC.md",
                "prespecification_sha256": hashlib.sha256(
                    (OUTPUT / "PRESPEC.md").read_bytes()
                ).hexdigest(),
                "normalizers": {k: NORMALIZERS[k] for k in keys},
                "embedders": embed.records,
                "resamples": RESAMPLES,
                "max_bootstrap_variants": CAP,
                "bootstrap_seed": 20240601,
                "bonferroni_family_size": len(FROZEN_CLASS),
                "frozen_inputs_read_only": [
                    "configs/frozen/p7_activity_gpt41_v1",
                    "evaluation/full_benchmark/*",
                    "evaluation/statistical_uncertainty_v2/paired_silhouette_differences.csv",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2))


def _decide(norm: pd.DataFrame, emb: pd.DataFrame) -> dict[str, object]:
    positive_logs = [d for d, c in FROZEN_CLASS.items() if c == "positive"]
    rule_models = [k for k, s in NORMALIZERS.items() if s["decision_rule"]]
    paired = norm.dropna(subset=["bonferroni_class"]) if "bonferroni_class" in norm else norm.iloc[:0]
    missing = [k for k in rule_models if k not in set(norm["normalizer"])]
    incomplete = sorted(
        set(norm.loc[norm["status"].ne("complete") & norm["normalizer"].isin(rule_models), "normalizer"])
    )
    failures = paired.loc[
        paired["normalizer"].isin(rule_models)
        & paired["dataset"].isin(positive_logs)
        & paired["status"].eq("complete")
        & ~paired["preserves_frozen_class"].astype(bool),
        ["normalizer", "dataset", "bonferroni_class"],
    ].to_dict("records")
    emb_failures = emb.loc[
        emb["dataset"].isin(positive_logs) & ~emb["preserves_frozen_class"].astype(bool),
        ["embedder", "dataset", "bonferroni_class"],
    ].to_dict("records")
    return {
        "rule": "PRESPEC.md, Decision rule",
        "normalizer_models_in_rule": rule_models,
        "normalizer_models_missing": missing,
        "normalizer_models_incomplete_coverage": incomplete,
        "normalizer_failures_on_positive_logs": failures,
        # PRESPEC amendment 2: decide over the rule models present; the
        # verdict is provisional while any rule model is missing.
        "normalizer_verdict_status": "provisional" if missing else "final",
        "normalizer_verdict": (
            "robust_to_normalizer" if not failures else "normalizer_dependent"
        ),
        "embedder_failures_on_positive_logs": emb_failures,
        "embedder_verdict": "robust_to_embedder" if not emb_failures else "embedder_dependent",
        "null_logs_class_changes": paired.loc[
            ~paired["dataset"].isin(positive_logs) & ~paired["preserves_frozen_class"].astype(bool),
            ["normalizer", "dataset", "bonferroni_class"],
        ].to_dict("records")
        + emb.loc[
            ~emb["dataset"].isin(positive_logs) & ~emb["preserves_frozen_class"].astype(bool),
            ["embedder", "dataset", "bonferroni_class"],
        ].to_dict("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    n = sub.add_parser("normalize")
    n.add_argument("key", choices=sorted(NORMALIZERS))
    n.add_argument("--workers", type=int, default=4)
    a = sub.add_parser("analyze")
    a.add_argument("--keys", nargs="*", default=sorted(NORMALIZERS))
    args = parser.parse_args()
    if args.command == "normalize":
        normalize(args.key, args.workers)
    else:
        analyze(args.keys)


if __name__ == "__main__":
    main()
