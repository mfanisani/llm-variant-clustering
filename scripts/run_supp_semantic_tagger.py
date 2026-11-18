"""Non-LLM semantic-role tagger as the P7 activity normalizer (supplementary).

Pre-specification: ``evaluation/supp_semantic_tagger_v1/PRESPEC.md``.

The GPT-4.1 activity normalizer is replaced by the BERT semantic-role tagger of
Rebmann & van der Aa (Information Systems 2022). ``tag`` cleans every activity
label with the authors' preprocessing, runs the authors' trained tagger in the
separate Python 3.9 environment ``.venv_tagger`` (subcommand ``_worker``), and
maps the tagged spans onto the frozen P7 schema with fixed, deterministic
rules. ``analyze`` rebuilds the ordered P7 pyramid from these descriptors and
re-computes the paired P7 minus raw silhouette bootstrap exactly as
``run_normalizer_robustness.py`` does. ``heldout`` evaluates the resulting
partitions on the frozen held-out variant split. Frozen artifacts are read
only; no LLM is called.

This file must stay importable by Python 3.9 (the tagger environment): heavy
and main-environment imports live inside functions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evaluation/supp_semantic_tagger_v1"
SCHEMA = ROOT / "configs/frozen/p7_activity_gpt41_v1/schema.json"
TAGGER_ROOT = ROOT / ".venv_tagger"
TAGGER_PYTHON = TAGGER_ROOT / "env/Scripts/python.exe"
TAGGER_SRC = TAGGER_ROOT / "src/semantic-annotation"
TAGGER_MODEL = TAGGER_SRC / ".model/main"
TAGGER_COMMIT = "045eb3e3b893d335d08b9e40de88be71a549e7cc"
TAGGER_MODEL_ZIP_SHA256 = "33565523585cf8f184be84c77d1a263456e9c8adfabd833dab9bc3254353933a"
TAGGER_MODEL_ZIP_BYTES = 809_770_103
HASH_SEED = "0"
AMBIGUITY = "not assessed (non-LLM tagger)"
NO_ROLE = "none explicit"
UNKNOWN = "unknown"

# Authors' preprocessing (preprocessing/preprocessor.py), verbatim patterns.
CAMEL_PATTERN_1 = re.compile("(.)([A-Z][a-z]+)")
CAMEL_PATTERN_2 = re.compile("([a-z0-9])([A-Z])")
NON_ALPHANUM = re.compile("[^a-zA-Z]")
MULTI_SPACE = re.compile(" +")
DIGITS = re.compile(r"\d+")

# Authors' const.TERMS_FOR_MISSING, verbatim.
TERMS_FOR_MISSING = [
    'MISSING', 'UNDEFINED', 'undefined', 'missing', 'none', 'nan', 'empty', 'empties', 'unknown', 'other', 'others', 'na',
    'nil', 'null', '', "", ' ', '<unknown>', "0;n/a", "NIL", 'undefined', 'missing', 'none', 'nan', 'empty', 'empties', 'unknown', 'other', 'others', 'na',
    'nil', 'null', '', ' ',
]
BO_NAME = "BO"  # const.ConceptType.BO_NAME.value
OTHER_TAG = "X"  # const.ConceptType.OTHER.value
IGNORED_TAGS = ("[PAD]", "O")

STATE_ENUM = (
    "created", "submitted", "approved", "final_approved", "rejected", "saved",
    "started", "ended", "requested", "handled", "reminded", "cancelled", "completed",
)
STATE_LEXICON = {value.replace("_", " "): value for value in STATE_ENUM}
STATE_LEXICON["canceled"] = "cancelled"
STAGE_BY_STATE = {
    "created": "initiation", "submitted": "initiation", "saved": "initiation",
    "requested": "initiation", "approved": "decision", "final_approved": "decision",
    "rejected": "decision", "started": "execution", "handled": "settlement",
    "ended": "closure", "cancelled": "closure", "completed": "closure",
    "reminded": "notification",
}


# --------------------------------------------------------------------------
# Pure mapping (stdlib only; shared by tests and both environments)
# --------------------------------------------------------------------------


def clean_label(label: str) -> str:
    """The authors' ``filter_col`` applied to one value."""
    text = CAMEL_PATTERN_1.sub(r"\1 \2", str(label))
    text = CAMEL_PATTERN_2.sub(r"\1 \2", text)
    text = text.lower()
    text = NON_ALPHANUM.sub(" ", text)
    text = MULTI_SPACE.sub(" ", text)
    text = text.strip()
    text = DIGITS.sub("", text)
    return text.strip()


def check_tok_for_object_type(split: list[str], pred: list[str]) -> list[str]:
    """The authors' post-processing, verbatim except that the unreachable
    proper-noun branch (it needs spaCy's large model) raises instead."""
    new_pred = []
    i = 1
    for tok, lab in zip(split, pred):
        if len(split) > i and tok == BO_NAME and split[i] != BO_NAME:
            raise AssertionError("check_propn branch reached; not replicated")
        new_pred.append(lab)
        i += 1
    return new_pred


def find_role(split: list[str], tags: list[str], role: str) -> list[str]:
    """The authors' ``model.augmented_log.find_role``, verbatim."""
    roles = []
    curr = ''
    for tok, r in zip(split, tags):
        if r == role:
            curr = " ".join([curr, tok])
        if r != role and curr != '':
            roles.append(curr.strip())
            curr = ''
    if curr != '':
        roles.append(curr.strip())
    return roles


def tag_runs(split: list[str], tags: list[str]) -> list[tuple[str, str]]:
    """All maximal same-tag runs in label order as ``(tag, text)``."""
    runs: list[tuple[str, list[str]]] = []
    for tok, tag in zip(split, tags):
        if runs and runs[-1][0] == tag:
            runs[-1][1].append(tok)
        else:
            runs.append((tag, [tok]))
    return [(tag, " ".join(words)) for tag, words in runs]


def normalize_tags(tags: list[str]) -> list[str]:
    return [OTHER_TAG if tag in IGNORED_TAGS else tag for tag in tags]


def map_to_p7(activity_label: str, split: list[str], tags: list[str]) -> dict[str, str]:
    """PRESPEC mapping from tagger spans to a frozen-schema P7 descriptor."""
    if len(split) != len(tags):
        raise ValueError(f"{activity_label!r}: {len(split)} tokens but {len(tags)} tags")
    tags = normalize_tags(check_tok_for_object_type(split, tags))
    action = " ".join(find_role(split, tags, "A")) or UNKNOWN
    business_object = " ".join(find_role(split, tags, "BO")) or UNKNOWN
    state_spans = [text for tag, text in tag_runs(split, tags) if tag in ("ASTATE", "BOSTATE")]
    if not state_spans:
        state = UNKNOWN
    else:
        state = next((STATE_LEXICON[s] for s in state_spans if s in STATE_LEXICON), "other")
    roles = [
        span
        for role in ("ACTOR", "REC")
        for span in find_role(split, tags, role)
        if span not in TERMS_FOR_MISSING
    ]
    return {
        "schema_version": "p7",
        "activity_label": activity_label,
        "action": action,
        "business_object": business_object,
        "process_stage": STAGE_BY_STATE.get(state, UNKNOWN),
        "explicit_state": state,
        "explicit_role": ", ".join(roles) or NO_ROLE,
        "ambiguity": AMBIGUITY,
    }


def token_jaccard(first: str, second: str) -> float:
    a, b = set(first.split()), set(second.split())
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Tagger worker (runs inside .venv_tagger, Python 3.9)
# --------------------------------------------------------------------------


def _worker(input_path: Path, output_path: Path) -> None:
    sys.path.insert(0, str(TAGGER_SRC))
    # torch/tensorflow must be imported before pandas in this environment
    # (a pandas-first import order crashes the interpreter on Windows).
    from instancelabeling.bert_tagger import BertForLabelParsing, BertWrapper
    import tensorflow
    import torch
    import transformers

    torch.manual_seed(0)
    torch.set_num_threads(2)
    labels = json.loads(input_path.read_text(encoding="utf-8"))["labels"]
    if any(not label.split() for label in labels):
        raise ValueError("Worker received an empty cleaned label")
    model = BertWrapper.load_serialized(str(TAGGER_MODEL), BertForLabelParsing)
    preds, confidence = model.predict([label.split() for label in labels])
    parameters = json.loads((TAGGER_MODEL / "sec_parameters.json").read_text(encoding="utf-8"))
    _write_json(
        output_path,
        {
            "labels": labels,
            "tags": [list(map(str, pred)) for pred in preds],
            "confidence_mean_max_logit": [float(value) for value in confidence],
            "environment": {
                "python": sys.version,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "tensorflow": tensorflow.__version__,
                "threads": torch.get_num_threads(),
            },
            "model_parameters": parameters,
        },
    )


# --------------------------------------------------------------------------
# tag (main environment)
# --------------------------------------------------------------------------


def _study_labels() -> dict[str, list[str]]:
    import pandas as pd
    from run_normalizer_robustness import LOGS

    return {
        dataset: pd.read_parquet(
            ROOT / "evaluation/full_benchmark" / dataset / "activity_embedding_index.parquet"
        )["activity_label"].astype(str).tolist()
        for dataset, *_ in LOGS
    }


def tag() -> None:
    import pandas as pd
    from llm_variant_clustering.semantic_compiler import (
        canonical_activity_text,
        validate_activity_semantics,
    )

    prespec = OUTPUT / "PRESPEC.md"
    if not prespec.exists():
        raise SystemExit("PRESPEC.md must exist before tagging")
    out_dir = OUTPUT / "tagger"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = _study_labels()
    cleaned = {(d, x): clean_label(x) for d, xs in labels.items() for x in xs}
    unique = sorted({c for c in cleaned.values() if c.split()})
    request, response = out_dir / "tagger_input.json", out_dir / "tagger_output.json"
    _write_json(request, {"labels": unique})
    tmp = TAGGER_ROOT / "cache/tmp"
    env = {
        **os.environ,
        "PYTHONHASHSEED": HASH_SEED,
        "PYTHONPATH": "",
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "TMP": str(tmp),
        "TEMP": str(tmp),
    }
    started = time.time()
    subprocess.run(
        [str(TAGGER_PYTHON), str(Path(__file__).resolve()), "_worker", str(request), str(response)],
        check=True,
        env=env,
        cwd=str(TAGGER_SRC),
    )
    seconds = time.time() - started
    result = json.loads(response.read_text(encoding="utf-8"))
    if result["labels"] != unique:
        raise SystemExit("Tagger output does not echo the requested labels")
    tagged = {
        label: (tags, conf)
        for label, tags, conf in zip(
            result["labels"], result["tags"], result["confidence_mean_max_logit"]
        )
    }

    rows = []
    for (dataset, label), clean in cleaned.items():
        split = clean.split()
        tags, conf = tagged.get(clean, ([], None))
        status, response_json, canonical, error = "validated", None, None, None
        try:
            semantics = map_to_p7(label, split, list(tags))
            validate_activity_semantics(
                semantics, schema_path=SCHEMA, expected_activity_label=label
            )
            response_json = json.dumps(semantics, sort_keys=True)
            canonical = canonical_activity_text(semantics)
        except (ValueError, AssertionError) as exc:
            status, error = "failed", str(exc)
        rows.append(
            {
                "dataset": dataset,
                "activity_label": label,
                "cleaned_label": clean,
                "tokens_json": json.dumps(split),
                "tags_json": json.dumps(list(tags)),
                "confidence_mean_max_logit": conf,
                "status": status,
                "error": error,
                "response_json": response_json,
                "canonical_descriptor": canonical,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_parquet(out_dir / "activity_semantics.parquet", index=False)
    frame.drop(columns=["response_json"]).to_csv(out_dir / "activity_semantics.csv", index=False)
    _write_json(
        out_dir / "run_manifest.json",
        {
            "normalizer": "rebmann_vanderaa_2022_bert_tagger",
            "repository": "https://github.com/a-rebmann/semantic-annotation",
            "commit": TAGGER_COMMIT,
            "model_zip_sha256": TAGGER_MODEL_ZIP_SHA256,
            "model_zip_bytes": TAGGER_MODEL_ZIP_BYTES,
            "model_dir": ".model/main",
            "environment": result["environment"],
            "model_parameters": result["model_parameters"],
            "schema_sha256": _sha256(SCHEMA),
            "prespecification_sha256": _sha256(prespec),
            "label_count": len(frame),
            "unique_cleaned_labels_tagged": len(unique),
            "validated": int(frame["status"].eq("validated").sum()),
            "tagger_seconds": round(seconds, 1),
            "tagger_input_sha256": _sha256(request),
            "tagger_output_sha256": _sha256(response),
        },
    )
    print(frame["status"].value_counts().to_string())


# --------------------------------------------------------------------------
# analyze (main environment)
# --------------------------------------------------------------------------


def _validity(frame, dataset: str) -> dict[str, object]:
    import numpy as np

    ok = frame.loc[frame["status"].eq("validated")]
    sem = [json.loads(x) for x in ok["response_json"]]
    tags = [t for x in frame["tags_json"] for t in normalize_tags(json.loads(x))]
    collisions = int(
        frame.groupby("cleaned_label")["activity_label"].nunique().gt(1).sum()
    )
    row: dict[str, object] = {
        "dataset": dataset,
        "labels": len(frame),
        "tagged_and_mapped": len(ok),
        "label_coverage": len(ok) / len(frame),
        "schema_validation_rate": float(frame["status"].eq("validated").mean()),
        "empty_cleaned_labels": int(frame["cleaned_label"].str.split().str.len().fillna(0).eq(0).sum()),
        "cleaned_label_collisions": collisions,
        "distinct_canonical_descriptors": int(ok["canonical_descriptor"].nunique()),
        "empty_descriptor_rate": float(np.mean(
            [s["action"] == UNKNOWN and s["business_object"] == UNKNOWN for s in sem]
        )),
        "action_unknown_rate": float(np.mean([s["action"] == UNKNOWN for s in sem])),
        "object_unknown_rate": float(np.mean([s["business_object"] == UNKNOWN for s in sem])),
        "state_unknown_rate": float(np.mean([s["explicit_state"] == UNKNOWN for s in sem])),
        "state_other_rate": float(np.mean([s["explicit_state"] == "other" for s in sem])),
        "stage_unknown_rate": float(np.mean([s["process_stage"] == UNKNOWN for s in sem])),
        "role_none_rate": float(np.mean([s["explicit_role"] == NO_ROLE for s in sem])),
        "mean_confidence_max_logit": float(frame["confidence_mean_max_logit"].dropna().mean()),
        "min_confidence_max_logit": float(frame["confidence_mean_max_logit"].dropna().min()),
        "tokens": len(tags),
    }
    for name in ("A", "BO", "ACTOR", "REC", "ASTATE", "BOSTATE", "X"):
        row[f"tag_share_{name}"] = tags.count(name) / max(len(tags), 1)
    return row


def _bpi2020_secondary(embed, labels, frozen_desc, tagger_desc, check_rows, rows) -> None:
    """Replicates run_representation_uncertainty.py for the BPI 2020 contrast."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import pairwise_distances

    from llm_variant_clustering.clustering import weighted_kmeans
    from llm_variant_clustering.features import parse_sequences
    from llm_variant_clustering.semantic_compiler import ordered_semantic_pyramid
    from llm_variant_clustering.uncertainty import paired_silhouette_difference
    from run_representation_source_benchmark import _compact_activity_space
    from run_statistical_uncertainty import DATASET_FOLDERS

    dataset, k, resamples = "bpi2020", 9, 300
    bench = ROOT / "evaluation/full_benchmark" / dataset
    variants = pd.read_parquet(ROOT / "datasets/processed" / DATASET_FOLDERS[dataset] / "variants.parquet")
    index_path = bench / "embedding_index.parquet"
    if index_path.exists():
        keep = frozenset(pd.read_parquet(index_path)["variant_id"].astype(str))
        variants = variants.loc[variants["variant_id"].astype(str).isin(keep)]
    variants = variants.reset_index(drop=True)
    sequences = parse_sequences(variants)
    weights = variants["frequency"].to_numpy(dtype=np.float64)
    sample = 2500 if len(variants) > 2500 else None

    def computed(texts):
        compact, _ = _compact_activity_space(embed(texts))
        matrix = ordered_semantic_pyramid(sequences, dict(zip(labels, compact)))
        return (
            pairwise_distances(matrix, metric="cosine"),
            weighted_kmeans(matrix, weights, n_clusters=k).labels,
        )

    p7 = computed([frozen_desc[x] for x in labels])
    raw = computed(labels)
    tagger = computed([tagger_desc[x] for x in labels])

    def paired(first, second):
        return paired_silhouette_difference(
            *first, *second, weights, resamples=resamples, sample_size=sample
        ).as_dict("difference_")

    reference = pd.read_csv(ROOT / "evaluation/representation_sources/source_paired_differences.csv")
    reference = reference.loc[
        reference["dataset"].eq(dataset)
        & reference["first"].eq("p7_llm_semantic_bge")
        & reference["second"].eq("raw_activity_label_bge")
    ].iloc[0]
    ref = paired(p7, raw)
    error = max(
        abs(ref[f"difference_{name}"] - reference[f"difference_{name}"])
        for name in ("point_estimate", "ci_lower", "ci_upper")
    )
    check_rows.append(
        {"dataset": "bpi2020_secondary", "table2_point_and_bounds_abs_error": error}
    )
    print(f"[check] bpi2020 secondary error={error:.2e}", flush=True)
    if error > 1e-6:
        raise SystemExit("Pipeline check failed on the BPI 2020 secondary contrast")
    for contrast, first, second, frozen in (
        ("tagger_minus_raw", tagger, raw, ref),
        ("tagger_minus_gpt41", tagger, p7, None),
    ):
        row = {"dataset": dataset, "contrast": contrast, "resamples": resamples, "bonferroni": False}
        row.update(paired(first, second))
        if frozen is not None:
            row.update(
                gpt41_point=frozen["difference_point_estimate"],
                gpt41_ci_lower=frozen["difference_ci_lower"],
                gpt41_ci_upper=frozen["difference_ci_upper"],
            )
        rows.append(row)


def analyze() -> None:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        pairwise_distances,
    )

    from llm_variant_clustering.features import parse_sequences
    from llm_variant_clustering.semantic_compiler import _normalize
    from run_normalizer_robustness import (
        FIELDS,
        FROZEN_CLASS,
        LOGS,
        _classify,
        _cluster,
        _fields,
        _log_frame,
        _paired,
        _partition,
        _pyramid,
    )
    from run_representation_source_benchmark import _embed

    cache: dict[str, np.ndarray] = {}

    def embed(texts: list[str]) -> np.ndarray:
        key = hashlib.sha256(json.dumps(texts).encode("utf-8")).hexdigest()
        if key not in cache:
            cache[key] = _embed(texts, ROOT / "models/huggingface")
        return cache[key]

    tagged = pd.read_parquet(OUTPUT / "tagger/activity_semantics.parquet")
    v2 = pd.read_csv(
        ROOT / "evaluation/statistical_uncertainty_v2/paired_silhouette_differences.csv"
    ).set_index("dataset")
    check_rows, contrast_rows, field_rows, validity_rows, partition_rows = [], [], [], [], []
    assignments_dir = OUTPUT / "assignments"
    assignments_dir.mkdir(parents=True, exist_ok=True)

    for dataset, k, suffix, paired in LOGS:
        frame = tagged.loc[tagged["dataset"].eq(dataset)]
        validity = _validity(frame, dataset)
        validity_rows.append(validity)
        if validity["label_coverage"] != 1.0:
            pd.DataFrame(validity_rows).to_csv(OUTPUT / "validity.csv", index=False)
            raise SystemExit(f"{dataset}: label coverage below 100%; see validity.csv")
        variants, bench = _log_frame(dataset)
        order = variants["variant_id"].astype(str).tolist()
        sequences = parse_sequences(variants)
        weights = variants["frequency"].to_numpy(dtype=np.float64)
        index = pd.read_parquet(bench / "activity_embedding_index.parquet")
        labels = index["activity_label"].astype(str).tolist()
        frozen_desc = dict(zip(labels, index["canonical_descriptor"].astype(str)))
        tagger_desc = dict(zip(frame["activity_label"], frame["canonical_descriptor"]))
        frozen_p7 = _partition(bench, "p7_ordered_pyramid_1_2_4_bge", suffix, order)

        # Pipeline check on the frozen descriptors (template, unchanged).
        ref_matrix = _pyramid(sequences, labels, embed([frozen_desc[x] for x in labels]))
        frozen_npy = np.load(
            bench / ("p7_ordered_embeddings.npy" if paired else "ordered_pyramid_embeddings.npy")
        )
        frozen_dist = pairwise_distances(frozen_npy, metric="cosine")
        ref_labels, ref_metrics = _cluster(ref_matrix, weights, k)
        distance_error = float(np.abs(ref_metrics["distances"] - frozen_dist).max())
        check = {
            "dataset": dataset,
            "max_cosine_distance_abs_error": distance_error,
            "embeddings_allclose": distance_error <= 1e-5,
            "ari_rebuilt_vs_frozen": float(adjusted_rand_score(frozen_p7, ref_labels)),
        }
        raw_dist = raw_labels = ref_row = None
        if paired:
            raw_dist = pairwise_distances(np.load(bench / "raw_ordered_embeddings.npy"), metric="cosine")
            raw_labels = _partition(bench, "raw_activity_label_pyramid_bge", suffix, order)
            ref_row = _paired(frozen_dist, frozen_p7, raw_dist, raw_labels, weights)
            check["paired_point_abs_error"] = abs(
                ref_row["difference_point_estimate"] - v2.loc[dataset, "difference_point_estimate"]
            )
            check["bonferroni_bounds_abs_error"] = max(
                abs(ref_row["bonferroni_ci_lower"] - v2.loc[dataset, "bonferroni_ci_lower"]),
                abs(ref_row["bonferroni_ci_upper"] - v2.loc[dataset, "bonferroni_ci_upper"]),
            )
        check_rows.append(check)
        print(f"[check] {check}", flush=True)
        if not check["embeddings_allclose"] or check["ari_rebuilt_vs_frozen"] != 1.0 or (
            paired and (check["paired_point_abs_error"] > 1e-6
                        or check["bonferroni_bounds_abs_error"] > 1e-6)
        ):
            pd.DataFrame(check_rows).to_csv(OUTPUT / "pipeline_check.csv", index=False)
            raise SystemExit(f"Pipeline check failed on {dataset}; see pipeline_check.csv")

        # Descriptor agreement with the frozen GPT-4.1 descriptors.
        for x in labels:
            mine, frozen = _fields(tagger_desc[x]), _fields(frozen_desc[x])
            field_rows.append(
                {
                    "dataset": dataset,
                    "activity_label": x,
                    "gpt41_descriptor": frozen_desc[x],
                    "tagger_descriptor": tagger_desc[x],
                    "identical": tagger_desc[x] == frozen_desc[x],
                    **{f"{f}_agrees": _normalize(mine[f]) == _normalize(frozen[f]) for f in FIELDS},
                    **{
                        f"{f}_token_jaccard": token_jaccard(_normalize(mine[f]), _normalize(frozen[f]))
                        for f in ("business_object", "action", "explicit_role")
                    },
                }
            )

        if validity["distinct_canonical_descriptors"] <= 1:
            contrast_rows.append({"dataset": dataset, "status": "degenerate"})
            continue
        descriptors = [tagger_desc[x] for x in labels]
        matrix = _pyramid(sequences, labels, embed(descriptors))
        tagger_labels, metrics = _cluster(matrix, weights, k)
        pd.DataFrame(
            {
                "variant_id": order,
                "cluster_label": tagger_labels.astype("int32"),
                "cluster_id": [f"tagger_c{int(c)}" for c in tagger_labels],
            }
        ).to_parquet(assignments_dir / f"assignments_p7_tagger_{dataset}_k{k}.parquet", index=False)
        partition = {
            "dataset": dataset,
            "matched_k": k,
            "tagger_silhouette": metrics["case_weighted_silhouette"],
            "tagger_stability_ari": metrics["stability_ari"],
            "gpt41_silhouette": ref_metrics["case_weighted_silhouette"],
            "ari_vs_frozen_p7": float(adjusted_rand_score(frozen_p7, tagger_labels)),
            "nmi_vs_frozen_p7": float(normalized_mutual_info_score(frozen_p7, tagger_labels)),
            "occupied_clusters": int(np.unique(tagger_labels).size),
        }
        if paired:
            partition["ari_vs_frozen_raw"] = float(adjusted_rand_score(raw_labels, tagger_labels))
            partition["nmi_vs_frozen_raw"] = float(normalized_mutual_info_score(raw_labels, tagger_labels))
            row = {"dataset": dataset, "contrast": "tagger_minus_raw", "resamples": 400, "bonferroni": True}
            row.update(_paired(metrics["distances"], tagger_labels, raw_dist, raw_labels, weights))
            row.update(
                gpt41_point=ref_row["difference_point_estimate"],
                gpt41_ci_lower=ref_row["difference_ci_lower"],
                gpt41_ci_upper=ref_row["difference_ci_upper"],
                gpt41_bonferroni_ci_lower=ref_row["bonferroni_ci_lower"],
                gpt41_bonferroni_ci_upper=ref_row["bonferroni_ci_upper"],
                gpt41_class=FROZEN_CLASS[dataset],
            )
            row["matches_gpt41_class"] = row["bonferroni_class"] == FROZEN_CLASS[dataset]
            contrast_rows.append(row)
            row = {"dataset": dataset, "contrast": "tagger_minus_gpt41", "resamples": 400, "bonferroni": True}
            row.update(_paired(metrics["distances"], tagger_labels, frozen_dist, frozen_p7, weights))
            contrast_rows.append(row)
        else:
            _bpi2020_secondary(embed, labels, frozen_desc, tagger_desc, check_rows, contrast_rows)
        partition_rows.append(partition)
        print(f"[{dataset}] tagger silhouette={partition['tagger_silhouette']:.4f} "
              f"ari_vs_p7={partition['ari_vs_frozen_p7']:.3f}", flush=True)

    pd.DataFrame(check_rows).to_csv(OUTPUT / "pipeline_check.csv", index=False)
    pd.DataFrame(validity_rows).to_csv(OUTPUT / "validity.csv", index=False)
    fields = pd.DataFrame(field_rows)
    fields.to_csv(OUTPUT / "descriptor_agreement.csv", index=False)
    agree_cols = ["identical"] + [f"{f}_agrees" for f in FIELDS] + [
        c for c in fields.columns if c.endswith("_token_jaccard")
    ]
    summary = fields.groupby("dataset", sort=False)[agree_cols].mean()
    summary.loc["all_labels"] = fields[agree_cols].mean()
    summary.reset_index().to_csv(OUTPUT / "descriptor_agreement_summary.csv", index=False)
    pd.DataFrame(partition_rows).to_csv(OUTPUT / "partition_metrics.csv", index=False)
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(OUTPUT / "silhouette_contrasts.csv", index=False)

    primary = contrasts.loc[contrasts["contrast"].eq("tagger_minus_raw") & contrasts["bonferroni"].eq(True)]
    positive = [d for d in ("bpi2017", "rtfm")
                if (primary.loc[primary["dataset"].eq(d), "bonferroni_class"] == "positive").any()]
    statement = {2: "reproduces", 1: "partially reproduces", 0: "does not reproduce"}[len(positive)]
    _write_json(
        OUTPUT / "summary.json",
        {
            "prespecified_statement": f"The tagger {statement} the GPT-4.1 advantage "
                                      "(Bonferroni-positive on BPI 2017 and RTFM).",
            "bonferroni_positive_logs_among_bpi2017_rtfm": positive,
            "class_matches": {
                r.dataset: bool(r.matches_gpt41_class) for r in primary.itertuples()
            },
            "note": "Descriptive; inclusion in the paper is the author's decision.",
        },
    )
    print(contrasts.round(4).to_string(index=False))


# --------------------------------------------------------------------------
# held-out process-model F1
# --------------------------------------------------------------------------

HELDOUT_LOGS = ("bpi2020", "bpi2017", "bpi2019", "rtfm")
FROZEN_P7_METHOD = "p7_ordered_pyramid_1_2_4_bge_dynamic"
TAGGER_METHOD = "p7_tagger"
AGGREGATED = (
    "fitness", "precision", "fitness_precision_f1", "simplicity_arc_degree",
    "places", "transitions", "silent_transitions", "arcs",
)


def coverage_problems(assignments, split) -> list[str]:
    """Clusters lacking a train or a test variant on the given split."""
    merged = assignments[["variant_id", "cluster_id"]].astype(str).merge(
        split[["variant_id", "split"]].astype(str), on="variant_id", how="left"
    )
    if merged["split"].isna().any():
        return ["variants missing from the frozen split"]
    problems = []
    for cluster, group in merged.groupby("cluster_id", sort=True):
        present = set(group["split"])
        for needed in ("train", "test"):
            if needed not in present:
                problems.append(f"{cluster} has no {needed} variant")
    return problems


def _aggregate(metrics):
    import pandas as pd

    rows = []
    for method, group in metrics.groupby("method", sort=True):
        weights = group["test_cases"]
        rows.append(
            {
                "method": method,
                "scope": "test_case_weighted_aggregate",
                "test_cases": int(weights.sum()),
                "clusters": len(group),
                **{m: float((group[m] * weights).sum() / weights.sum()) for m in AGGREGATED},
            }
        )
    return pd.DataFrame(rows)


def _heldout_worker(dataset: str) -> None:
    import pandas as pd
    import pm4py

    from llm_variant_clustering.process_evaluation import evaluate_process_partition_holdout
    from run_heldout_hashseed_sensitivity import _case_map, _xes
    from run_normalizer_robustness import LOGS

    if os.environ.get("PYTHONHASHSEED") != HASH_SEED:
        raise SystemExit("Held-out worker must run with PYTHONHASHSEED=0")
    k = {d: kk for d, kk, *_ in LOGS}[dataset]
    bench = ROOT / "evaluation/full_benchmark" / dataset
    out_dir = OUTPUT / "heldout"
    split = pd.read_parquet(bench / "process_model_metrics_dynamic_heldout_variant_split.parquet")
    frames = {
        TAGGER_METHOD: pd.read_parquet(OUTPUT / "assignments" / f"assignments_p7_tagger_{dataset}_k{k}.parquet"),
        FROZEN_P7_METHOD: pd.read_parquet(bench / f"assignments_{FROZEN_P7_METHOD}.parquet"),
    }
    status_path = out_dir / f"{dataset}_status.json"
    variant_sets = {m: frozenset(f["variant_id"].astype(str)) for m, f in frames.items()}
    split_set = frozenset(split["variant_id"].astype(str))
    problems = coverage_problems(frames[TAGGER_METHOD], split)
    if variant_sets[TAGGER_METHOD] != split_set or variant_sets[FROZEN_P7_METHOD] != split_set:
        problems.append("assignment variant sets differ from the frozen split")
    if problems:
        _write_json(status_path, {"dataset": dataset, "status": "infeasible_on_frozen_split",
                                  "problems": problems})
        print(f"[{dataset}] infeasible: {problems}", flush=True)
        return
    started = time.time()
    case_map = pd.read_parquet(_case_map(dataset))
    case_map = case_map[case_map["variant_id"].astype(str).isin(split_set)].copy()
    events = pm4py.read_xes(str(_xes(dataset)), return_legacy_log_object=False, show_progress_bar=False)
    events = events[events["case:concept:name"].astype(str).isin(set(case_map["case_id"].astype(str)))].copy()
    parts = [
        evaluate_process_partition_holdout(
            events, case_map, frame, split, method=method, noise_threshold=0.2
        )
        for method, frame in frames.items()
    ]
    metrics = pd.concat(parts, ignore_index=True)
    metrics.to_csv(out_dir / f"{dataset}_clusters_hashseed{HASH_SEED}.csv", index=False)
    aggregate = _aggregate(metrics)
    aggregate.insert(0, "dataset", dataset)
    aggregate.to_csv(out_dir / f"{dataset}_aggregate_hashseed{HASH_SEED}.csv", index=False)
    _write_json(status_path, {"dataset": dataset, "status": "completed", "hash_seed": HASH_SEED,
                              "seconds": round(time.time() - started, 1)})
    print(aggregate.to_string(index=False), flush=True)


def heldout(datasets: list[str]) -> None:
    import pandas as pd

    out_dir = OUTPUT / "heldout"
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        if not (out_dir / f"{dataset}_status.json").exists():
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "_heldout_worker", dataset],
                check=True,
                env={**os.environ, "PYTHONHASHSEED": HASH_SEED},
            )
    table3 = pd.read_csv(ROOT / "evaluation/paper_tables_v1/table3_heldout_f1.csv").set_index("log")
    rows = []
    for dataset in HELDOUT_LOGS:
        status_path = out_dir / f"{dataset}_status.json"
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        row = {"dataset": dataset, "status": status["status"],
               "frozen_table3_whole_log": table3.loc[dataset, "whole_log"],
               "frozen_table3_raw": table3.loc[dataset, "raw_bge"],
               "frozen_table3_p7": table3.loc[dataset, "p7"]}
        if status["status"] == "completed":
            agg = pd.read_csv(out_dir / f"{dataset}_aggregate_hashseed{HASH_SEED}.csv").set_index("method")
            row.update(
                tagger_f1=agg.loc[TAGGER_METHOD, "fitness_precision_f1"],
                p7_f1_same_seed=agg.loc[FROZEN_P7_METHOD, "fitness_precision_f1"],
                tagger_minus_p7=agg.loc[TAGGER_METHOD, "fitness_precision_f1"]
                - agg.loc[FROZEN_P7_METHOD, "fitness_precision_f1"],
                tagger_fitness=agg.loc[TAGGER_METHOD, "fitness"],
                tagger_precision=agg.loc[TAGGER_METHOD, "precision"],
                hash_seed=HASH_SEED,
            )
            row["tagger_minus_frozen_raw"] = row["tagger_f1"] - row["frozen_table3_raw"]
        else:
            row["problems"] = "; ".join(status.get("problems", []))
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "heldout_f1.csv", index=False)
    print(frame.round(4).to_string(index=False))


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


def manifest() -> None:
    files = sorted(
        p for p in OUTPUT.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    )
    script = Path(__file__).resolve()
    tests = ROOT / "tests/test_supp_semantic_tagger.py"
    _write_json(
        OUTPUT / "manifest.json",
        {
            "schema_version": 1,
            "status": "supplementary_after_primary_artifact_lock",
            "prespecification": "PRESPEC.md",
            "prespecification_sha256": _sha256(OUTPUT / "PRESPEC.md"),
            "tagger": {
                "reference": "Rebmann & van der Aa, Information Systems 2022, doi:10.1016/j.is.2022.102111",
                "repository": "https://github.com/a-rebmann/semantic-annotation",
                "commit": TAGGER_COMMIT,
                "model_zip_sha256": TAGGER_MODEL_ZIP_SHA256,
                "model_zip_bytes": TAGGER_MODEL_ZIP_BYTES,
                "sec_parameters_sha256": _sha256(TAGGER_MODEL / "sec_parameters.json"),
            },
            "embedding_model": "BAAI/bge-large-en-v1.5@d4aa6901d3a41ba39fb536a557fa166f842b0e09",
            "bootstrap": {"resamples": 400, "cap": 2500, "seed": 20240601, "bonferroni_family": 4,
                          "bpi2020_secondary_resamples": 300},
            "heldout_hash_seed": HASH_SEED,
            "llm_api_calls": 0,
            "frozen_inputs_read_only": [
                "configs/frozen/p7_activity_gpt41_v1",
                "evaluation/full_benchmark",
                "evaluation/statistical_uncertainty_v2/paired_silhouette_differences.csv",
                "evaluation/representation_sources/source_paired_differences.csv",
                "evaluation/paper_tables_v1/table3_heldout_f1.csv",
            ],
            "code": {
                "scripts/run_supp_semantic_tagger.py": _sha256(script),
                "tests/test_supp_semantic_tagger.py": _sha256(tests) if tests.exists() else None,
            },
            "files": {str(p.relative_to(OUTPUT)).replace("\\", "/"): _sha256(p) for p in files},
        },
    )
    print(OUTPUT / "manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("_worker")
    worker.add_argument("input", type=Path)
    worker.add_argument("output", type=Path)
    sub.add_parser("tag")
    sub.add_parser("analyze")
    held = sub.add_parser("heldout")
    held.add_argument("--datasets", nargs="+", default=list(HELDOUT_LOGS))
    held_worker = sub.add_parser("_heldout_worker")
    held_worker.add_argument("dataset")
    sub.add_parser("manifest")
    args = parser.parse_args()
    if args.command == "_worker":
        _worker(args.input, args.output)
    elif args.command == "tag":
        tag()
    elif args.command == "analyze":
        analyze()
    elif args.command == "heldout":
        heldout(args.datasets)
    elif args.command == "_heldout_worker":
        _heldout_worker(args.dataset)
    else:
        manifest()


if __name__ == "__main__":
    main()
