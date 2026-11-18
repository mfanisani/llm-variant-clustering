# Non-LLM semantic tagger as P7 normalizer (supplementary, v1): pre-specification

Written 2026-11-07, **before any of the study's activity labels were tagged and
before any embedding or clustering in this directory.** Supplementary analysis
for paper 1; not confirmatory. Frozen artifacts (`configs/frozen/*`,
`evaluation/full_benchmark/*`, `evaluation/statistical_uncertainty_v2/*`,
`evaluation/representation_sources/*`, `evaluation/paper_tables_v1/*`) are read
only. All outputs go to `evaluation/supp_semantic_tagger_v1/`. Code:
`scripts/run_supp_semantic_tagger.py`. No LLM API call is made.

## Question

Does a non-LLM semantic-role tagger, used in place of the GPT-4.1 activity
normalizer, give the same P7 − raw-label silhouette contrast as the frozen
GPT-4.1 descriptors?

Frozen GPT-4.1 reference (`statistical_uncertainty_v2`, 400 resamples):

| Log | Point | 95% | Bonferroni 98.75% | Class |
|---|---|---|---|---|
| BPI 2017 | +0.109 | [+0.085, +0.118] | [+0.080, +0.120] | positive |
| BPI 2019 | +0.034 | [−0.061, +0.111] | [−0.082, +0.153] | includes zero |
| Sepsis | −0.006 | [−0.020, +0.012] | [−0.022, +0.017] | includes zero |
| RTFM | +0.083 | [+0.012, +0.134] | [+0.008, +0.161] | positive |

BPI 2020 (paper Table 2, `representation_sources`, 300 resamples, 95% only):
+0.033 [−0.028, +0.077].

## Tagger (fixed)

- Rebmann & van der Aa, "Enabling semantics-aware process mining through the
  automatic annotation of event logs", Information Systems 2022,
  doi:10.1016/j.is.2022.102111. Code `github.com/a-rebmann/semantic-annotation`
  at commit `045eb3e3b893d335d08b9e40de88be71a549e7cc`; trained model
  `.model.zip` (Git LFS, 809,770,103 bytes, SHA-256
  `33565523585cf8f184be84c77d1a263456e9c8adfabd833dab9bc3254353933a`),
  directory `.model/main` (BERT-base-uncased token classifier; tags
  A, BO, ACTOR, REC, ASTATE, BOSTATE, X).
- Separate environment `.venv_tagger/env` (Python 3.9.21, torch 1.9.0+cpu,
  transformers 4.9.2, tensorflow-cpu 2.7.0). The main `.venv` is not modified.
- Installation reproduction check (done before this file was written, on the
  authors' own `input/logs/sample_log.csv`, not on study data): for all 8
  distinct labels, the predicted A/BO/ACTOR/REC spans equal the
  `action:name`, `object:name`, `org:actor:name` and `org:passive:name`
  columns of the authors' published `output/logs/sample_log_augmented.csv`.
- Inference: CPU, eval mode, `torch.manual_seed(0)`, 2 threads, the authors'
  `BertWrapper.predict` (batch 100, max length 64), unique cleaned labels in
  sorted order. The tagger is deterministic given its input.

## Input labels (fixed)

The activity labels of `evaluation/full_benchmark/<log>/activity_embedding_index.parquet`
(the same label set the frozen GPT-4.1 descriptors cover): BPI 2020 34,
BPI 2017 24, BPI 2019 42, Sepsis 16, RTFM 11 (127 total).

**Decision:** the authors' pipeline first classifies event attributes and tags
only "rich text" columns; we bypass attribute classification and tag each
`concept:name` label directly, because only activity labels enter P7.

Cleaning is the authors' `preprocessor.filter_col`, replicated verbatim:
camel-case split `(.)([A-Z][a-z]+)` → `\1 \2` and `([a-z0-9])([A-Z])` → `\1 \2`;
lowercase; `[^a-zA-Z]` → space; collapse spaces; strip; remove digits; strip.
Tokens are the whitespace split. The authors' post-processing
`check_tok_for_object_type` is applied verbatim; on lowercased tokens its
condition (`token == "BO"`) cannot hold, so it is the identity (asserted).
Predicted `[PAD]` or `O` tags are treated as X. Spans are the authors'
`find_role` (maximal runs of consecutive tokens with the same tag), in label
order.

## Mapping to the frozen P7 schema (fixed)

| P7 field | Rule | Fill when empty |
|---|---|---|
| `action` | A spans joined by one space (surface forms, no lemmatization) | `unknown` |
| `business_object` | BO spans joined by one space | `unknown` |
| `explicit_state` | ASTATE and BOSTATE spans in label order; the first span equal to an enum value written with `_` → space (`final approved` → `final_approved`), or `canceled` → `cancelled`, gives that value; if spans exist but none matches → `other` | `unknown` |
| `process_stage` | the frozen prompt's deterministic convention applied to `explicit_state`: created/submitted/saved/requested → initiation; approved/final_approved/rejected → decision; started → execution; handled → settlement; ended/cancelled/completed → closure; reminded → notification | `unknown` (also for `other`/`unknown` state) |
| `explicit_role` | ACTOR spans then REC spans, joined by `, `, after dropping spans whose text is in the authors' `TERMS_FOR_MISSING` list (e.g. `missing`, `undefined`, `none`, `unknown`) | `none explicit` |
| `ambiguity` | constant `not assessed (non-LLM tagger)` (not part of the canonical text) | — |
| `activity_label` | the original label; `schema_version` = `p7` | — |

No other word source is used: words tagged X are dropped, A spans that look
like states are *not* moved to `explicit_state`, and no field is copied from
the GPT-4.1 descriptors. Every mapped descriptor is validated against the
frozen `schema.json` with `validate_activity_semantics`; canonical text is
`canonical_activity_text` (identical to the frozen pipeline).

## Fixed analysis (identical to `run_normalizer_robustness.py`)

BGE-large-en-v1.5 at the frozen revision, rank-preserving compact projection,
ordered pyramid levels 1-2-4, case-weighted K-means (sparse container, seeds as
frozen) at matched K: BPI 2020 9, BPI 2017 2, BPI 2019 6, Sepsis 8, RTFM 2.
Raw side for the four paired logs: frozen raw-label pyramid embeddings and
frozen raw partition. Bootstrap: 400 resamples, m-out-of-n cap 2500, seed
20240601 (identical resamples to `statistical_uncertainty_v2`), 95% percentile
and Bonferroni 98.75% (family 4).

BPI 2020 (secondary, 95% only): replicates `run_representation_uncertainty.py`
exactly — `variants.parquet` order filtered to the embedding index, raw labels
BGE-embedded and compacted, dense matrix, weighted K-means K = 9,
`paired_silhouette_difference` with 300 resamples, seed 20240601, no cap.

## Metrics

Per log:
1. **Primary:** paired P7(tagger) − raw case-weighted silhouette difference,
   95% and (paired logs) Bonferroni intervals and class, next to the frozen
   GPT-4.1 values.
2. Paired P7(tagger) − P7(GPT-4.1) difference, same resamples.
3. Case-weighted silhouette; ARI and NMI of the tagger partition vs the
   frozen P7 partition and vs the frozen raw partition.
4. Descriptor agreement with frozen GPT-4.1: exact canonical identity rate and
   per-field exact agreement (object, action, stage, state, role, compared
   after the frozen `_normalize`); secondary descriptive token overlap
   (Jaccard of word sets) for object, action and role.
5. Held-out process-model F1 (below).

## Validity checks (reported with the results)

- Label coverage: every label tagged and mapped (required = 100%; otherwise
  the run stops and the gap is reported).
- Schema validation rate (required = 100%; a failure is a mapping bug, the
  run stops).
- Empty-descriptor rate (action and object both `unknown`), per-field
  `unknown`/`none explicit` rates, tag distribution, number of distinct
  canonical descriptors vs labels, cleaned-label collisions (two labels that
  clean to the same token string), mean tagger confidence (the authors'
  per-label mean of max logits; not a probability).
- A log whose tagger descriptors are all identical is reported as
  `degenerate` and not clustered.

## Pipeline check (must pass before results are read)

Rebuilding from the frozen GPT-4.1 descriptors with this script must
reproduce, as in the template: max abs error of pairwise cosine distances vs
the frozen P7 embeddings ≤ 1e-5, ARI = 1 vs the frozen P7 partition, and the
`statistical_uncertainty_v2` paired point and Bonferroni bounds within 1e-6;
for BPI 2020 the `representation_sources` P7 − raw point and 95% bounds within
1e-6. Otherwise the run stops and the discrepancy is reported.

## Held-out process-model F1 (if feasible within budget)

Logs BPI 2020, BPI 2017, BPI 2019, RTFM (as paper Table 3). Frozen variant
split `full_benchmark/<log>/process_model_metrics_dynamic_heldout_variant_split.parquet`
(no re-split), `evaluate_process_partition_holdout`, noise threshold 0.2,
test-case-weighted aggregate as in `run_heldout_process_evaluation.py`. The
tagger partition and the frozen P7 dynamic partition are evaluated in the same
subprocess with `PYTHONHASHSEED=0` (Inductive Miner tie-breaking); the
reported contrast is tagger − P7 at that seed, alongside the frozen Table 3
values. If any tagger cluster lacks training or test variants on the frozen
split, that log is reported `infeasible_on_frozen_split` (no re-split, no
partial aggregate). No interval is attached to F1, as in the paper.

## Descriptive summary statement (fixed now; not a decision on inclusion)

Per paired log, the tagger *matches* the GPT-4.1 class if its Bonferroni
class equals the frozen class. Overall: the tagger **reproduces** the
GPT-4.1 advantage if it is Bonferroni-positive on both BPI 2017 and RTFM,
**partially** if on one, **does not** if on neither. Inclusion in the paper is
the author's decision.

## Limits stated in advance

- The tagger was trained on labels from process-model collections and event
  logs that may overlap the public logs studied here; no claim of
  out-of-distribution evaluation is made.
- The tagger provides no stage or ambiguity semantics; `process_stage` is
  derived only from tagged states by the frozen prompt's convention, so stage
  agreement with GPT-4.1 is expected to be low by construction.
- Surface forms are not lemmatized (GPT-4.1 gives base verbs), which lowers
  exact action agreement without necessarily changing embeddings much.
- One tagger, one mapping. No human ratings are involved or simulated.
