# Cluster naming and semantic separability (v1): pre-specification

Written 2026-11-07, **before any LLM naming or judging call in this
directory.** This directory adds a post-hoc business-naming step on top of
the already-frozen P7 clustering (paper 1) and tests whether the resulting
names let a held-out judge recover cluster membership better than they could
for baseline partitions at the same K. It is a machine-only evaluation. It
does not alter any frozen artifact: `evaluation/full_benchmark/`,
`evaluation/representation_sources/`, `evaluation/statistical_uncertainty_v2/`,
`configs/frozen/`, and `datasets/` are read only. All outputs go to
`evaluation/cluster_naming_v1/`.

## Central risk: circularity

An LLM can produce distinct, plausible names for any partition, including a
random one. Distinct names alone prove nothing. The evaluation therefore:

1. tests names on **held-out variants** the namer never saw;
2. uses a **judge from a different vendor** than the namer;
3. compares P7 against the **baseline partitions** and a **random partition**
   under the identical procedure.

**No simulated human data.** This is a machine-only evaluation. The LLM judge
is not described as a "human proxy" and does not establish human
interpretability. Human validation is deferred to the paper 2 human study
(`evaluation/human_study/v2/`).

## Claim under test

*P7 clusters are more nameable and separable than baseline partitions at the
same K.* A null or negative result is reported as found.

## Inputs (fixed, read only)

- Logs, matched K, roles: BPI 2020 (K=9, primary), BPI 2017 (K=2, primary),
  BPI 2019 (K=6, primary), Sepsis (K=8, exploratory, stress test), RTFM (K=2,
  stress test).
- Partitions named per log: `p7`, `raw`, `a2v` (rank matched to P7, not
  `d16`), `tfidf` (own automatic K, descriptive only), `random` × 3 draws
  (seeds 20240601 / 20240602 / 20240603, P7 cluster sizes preserved).
- Train/held-out split: the existing 80/20 `process_model_metrics_dynamic_
  heldout_variant_split.parquet` for BPI 2020/2017/2019/RTFM. For Sepsis, a
  newly created split under the same rule (20% uniform random sample of
  unique variants, seed 20240601), saved to
  `evaluation/cluster_naming_v1/splits/sepsis_variant_split.parquet`.
- BPI 2017 and BPI 2019 are restricted to the variants in
  `evaluation/full_benchmark/<log>/embedding_index.parquet` (excludes the 45
  calibration variants); the representation-source assignment files are
  already restricted to this set.
- Seed for every random element: **20240601**.

## Pipeline check (must pass before results are read)

`assignments_p7_llm_semantic_bge_k{K}` must agree with the frozen
`evaluation/full_benchmark/<log>/assignments_p7_ordered_pyramid_1_2_4_bge_
{dynamic|k8}.parquet` at case-weighted ARI = 1 on all five logs (Sepsis uses
`_k8`). If any log fails, the run stops and the discrepancy is reported
instead of results. (Reported in `pipeline_check.csv`.)

## Blinding

- The namer sees randomized, seeded letter codes for clusters, not the
  partition's identity, and the same instructions for every partition.
- The namer sees no dataset name, domain description, or log metadata; names
  must come from activity labels alone.
- The judge sees only names and definitions, in a random per-call order, with
  no cluster codes, sizes, or partition source.

## Models

| Role | Model | Endpoint | Deployment | Expected returned model |
|---|---|---|---|---|
| Namer | GPT-4.1 | `<primary Azure OpenAI endpoint>` | `gpt-4.1-standard` | `gpt-4.1-2025-04-14` |
| Primary judge | DeepSeek-V4-Pro | `<secondary Azure OpenAI endpoint>` | `deepseek-v4-pro` | `deepseek-v4-pro` |
| Secondary judge | GPT-4o | `<secondary Azure OpenAI endpoint>` | `gpt-4o-2024-08-06-global` | `gpt-4o-2024-08-06` |

See Amendment 1 below (corrects the namer and GPT-4o deployments from an
earlier, invalid configuration that was never used for a real call).

Temperature 0, strict JSON-schema tool call, api_version
`2025-01-01-preview`, per-model concurrency caps (namer 2, DeepSeek-V4-Pro 4,
GPT-4o 2 — see Amendment 1). Authentication is Entra ID
via the Azure CLI (`AzureOpenAIToolProvider`); no key or token is written to
any file. Every response is cached with `StructuredResponseCache`, and the
model string returned by the service is checked against the table above; a
mismatch aborts the run. Optional, only if time allows: a crossed run with
DeepSeek as namer and GPT-4o as judge, reported as exploratory.

## Hypotheses

- **H-N1 (sanity):** P7 names identify held-out variants better than the
  size-matched random partition, on every log with all 3 random draws
  present (Amendment 1b; a log with fewer than 3 draws is "incomplete").
- **H-N2 (primary):** P7 names identify held-out variants better than
  raw-label names, at the same K.
- **H-N3 (secondary):** P7 versus Activity2Vec names, at the same K.
- **Descriptive:** TF-IDF at its own K.

## Metrics and statistics

**Primary metric: unweighted variant-level Cohen's κ** (Amendment 2; was
case-weighted κ) between the true cluster and the judge's choice on the
held-out sample — each held-out variant counts once, because the judge's
unit of evaluation is the variant type, not the case. κ corrects for K and
for cluster-size imbalance, which matters because chance accuracy is 50%
when K = 2.

**Secondary/descriptive metrics:**
- case-weighted κ, case-weighted accuracy, and the case-weighted majority
  baseline (Amendment 2 demotes these from primary to secondary; weights
  are case frequencies), shown alongside each log's Kish effective sample
  size (ESS, frequency-only) per Amendment 2;
- unweighted (macro per-cluster) recall;
- for comparison, the accuracy of always choosing the largest cluster;
- name distinctness: mean pairwise cosine similarity between a partition's
  names, using the frozen BGE embedder (`BAAI/bge-large-en-v1.5`, revision
  `d4aa6901d3a41ba39fb536a557fa166f842b0e09`) — lower means more distinct;
- evidence validity: the share of names that pass validation on the first
  attempt;
- agreement between the two judges (κ between their choices).

**Intervals.**
- Paired bootstrap over sampled held-out variants: 2,000 resamples (Amendment
  1; was 400), seed 20240601. The same resampled variant indices are applied
  to every partition of a log (paired design). Resamples dropped because a
  resampled subset contained a single class are recorded alongside every
  interval.
- Report differences in κ: P7 − raw, P7 − a2v and P7 − random (mean over the
  3 draws).
- Bonferroni across the 5 logs for H-N2 (99% two-sided intervals, family of
  5). Nominal 95% for everything else, labelled descriptive.
- Each difference is classified as *positive*, *negative*, or *includes zero*
  using the same rule as `_classify` in `run_normalizer_robustness.py`: an
  interval is *positive* if its lower bound is > 0, *negative* if its upper
  bound is < 0, and *includes zero* otherwise.

## Pre-specified claim language (fixed now)

- **H-N1:** if it fails on any log, the naming step is not informative there.
  That log's H-N2 result is reported but not interpreted as a separability
  claim.
- **H-N2 supported:** if the primary judge's P7 − raw interval is positive on
  ≥ 3 of 5 logs and the secondary judge agrees in sign on those logs, the
  paper may say: *"P7 clusters were more identifiable from LLM-generated
  names than raw-label clusters on N of 5 logs."*
- **Otherwise:** the paper says the naming evaluation **does not separate P7
  from the raw-label baseline**.
- **In every case, the paper does not claim** that names are correct, useful
  to analysts, or human-interpretable.

## Validation and repair (naming)

Up to 3 repair attempts via `generate_with_validation_repairs`:
- schema validity;
- exactly K entries, one per letter code;
- unique names and a word count of 3–8;
- every cited evidence ID exists in **that cluster's** profile;
- the banned-words filter (no "Cluster 1", "miscellaneous", "standard cases",
  and no success, failure, compliance, speed, cost, or ownership claims
  unless an activity label states them).

## Known limitations (stated in advance)

- BPI 2017 and RTFM have K = 2, so naming separability there is weak
  evidence.
- Sepsis uses an exploratory K = 8, because the dynamic-K selector abstains
  on it.
- Partitions were fitted on all variants (transductive). Held-out variants
  were unseen by the namer, but not by the clustering.
- The namer and one judge are OpenAI models. DeepSeek is the only other
  vendor. Claude is unavailable (expired CAPI certificate).
- Activity labels of public logs may be in the models' pretraining data.

## Amendments

### Amendment 1 (2026-11-07)

Recorded before any LLM naming or judging call was made under it, following
an implementation review (`reports/task_cluster_naming_separability.md`,
Section 13) that found the pairing, prompt-leak, model-configuration, and
exception-handling bugs below in the pre-review script. No results existed
yet for this evaluation, so nothing is being reinterpreted; these are
corrections to the planned procedure itself.

**Pairing and statistics**
- The P7-vs-other-partition diff was previously computed on frames whose row
  order (not just content) had to match; two partitions covering the
  identical held-out variant set in a different row order were silently
  treated as "misaligned", and the misaligned fallback path reused P7's own
  true labels for the other partition instead of that partition's own.
  Fixed: alignment is now by `variant_id` (indices are sorted before
  comparison), and the misaligned fallback (kept as defense in depth) pulls
  each side's own true/predicted codes.
- P7 − random is now the mean of κ over every available random1–3 draw,
  computed per bootstrap resample (previously it used only `random1` and
  discarded the computed per-partition means).
- Bootstrap resamples raised from 400 to **2,000**; the count of resamples
  dropped because a resampled subset contained a single class is now
  recorded on every interval, with a console warning above 1%.

**Blinding**
- Partition and log identity no longer appear in any LLM-visible prompt
  text (previously `Partition code: <log>:<partition>` and a `log:partition:
  batch_index` batch ID were sent verbatim). Both are now opaque
  SHA-256-derived codes with no reversible mapping sent to the model.
- Evidence IDs (e.g. `A1`, `T1`) are prefixed with their cluster's own letter
  code before being shown to the namer, so they are globally unique within a
  partition; previously two different clusters could both expose an `A1`.

**Models**
- The namer's deployment was corrected from `gpt-4.1` to `gpt-4.1-standard`
  on the primary endpoint (the former does not exist there).
- The GPT-4o secondary judge moves from the primary endpoint/`gpt-4o` (no such
  deployment exists on the primary endpoint) to **the secondary endpoint**, deployment
  `gpt-4o-2024-08-06-global`, expected served model `gpt-4o-2024-08-06`.
- Every call now asserts `response.model` starts with the expected model
  string and aborts the run on a mismatch (previously unchecked). A startup
  smoke probe (one minimal call per model) runs before the first production
  call of `name`/`judge` for that model and also aborts the run on failure
  or mismatch.
- A documented, amendment-gated fallback deployment exists for the namer
  only (`gpt-4.1` plain on the primary endpoint, 1 worker), used only if the smoke probe
  or a production call against the primary namer deployment fails with a
  not-found (404) error.

**Exception handling (approximately 2,100 calls)**
- Provider-level retries: 6 attempts, jittered exponential backoff capped at
  60s, honouring a `Retry-After` response header when present.
- The repair loop now wraps `provider.generate` itself (previously a
  `ValueError` from a length-truncated or tool-call-less response escaped
  the repair loop uncaught); the first repair attempt also raises
  `max_tokens`.
- Authentication failures (401/403), not-found (404), content-filter
  rejections, and model mismatches are fatal and abort the whole run
  immediately rather than being retried or isolated per unit.
- Every other per-unit failure (one partition for naming; one
  (partition, batch) pair for judging) is caught, written to
  `failures/*.jsonl`, and does not stop the rest of the run
  (`ThreadPoolExecutor.submit`/`as_completed`, not `pool.map`).
- Missing-data rule (pre-specified here, before any judge call): if a batch
  fails for a log+judge in any partition, every variant in it is excluded
  from **every** partition of that log+judge (paired design preserved). If
  the excluded fraction exceeds 5% of the held-out sample, that log+judge is
  marked incomplete; `analyze` refuses incomplete data unless invoked with
  `--allow-incomplete`.
- A circuit breaker aborts a step after 10 consecutive failures or a 5%
  overall failure fraction.
- Concurrency caps: DeepSeek-V4-Pro ≤ 4 workers, GPT-4o ≤ 2 workers, namer ≤
  2 workers (unchanged from the original spec, now enforced in code).
- Outputs are resumable (the existing structured-response cache keys on
  prompt/schema/parameters content); `name`/`judge` honour a
  `--confirm-prespec <sha256-of-this-file>` guard so a run cannot
  accidentally start against a stale pre-specification.

**Verification order:** B1–B8 and S1–S7 implemented; full test suite green;
`--dry-run` executed and call counts/token estimates reported; a small
pilot (RTFM, partitions p7 and random1, namer + DeepSeek judge only) run and
reviewed before any further production call.

### Amendment 1b (2026-11-07)

Recorded before the full production run, following pilot review. Analysis
rule only; no LLM call is affected.

- **H-N1 data-completeness rule:** H-N1 requires all 3 random draws
  (`random1`, `random2`, `random3`) to be judged for a log. With fewer than
  3 draws present, that log is classified **"incomplete"** for H-N1 (not
  "informative" or "uninformative"); it is still reported, but it does not
  count towards either classification.

### Amendment 2 (2026-11-07)

Recorded before the full production run, triggered by inspection of the
RTFM pilot results and justified by the frequency-only Kish effective
sample size (ESS) table below. Analysis rule only; no LLM call is affected,
and the pilot's cached naming/judging calls are reused unchanged.

**Why.** Case-weighted κ (the original primary metric) weights each
held-out variant by its case frequency in the full log. On several logs the
held-out sample is dominated by one or two high-frequency variants, so the
case-weighted statistic is effectively driven by only a handful of
"judgements" even though dozens of variants were sampled and judged. Kish's
effective sample size, `ESS = (Σw)² / Σw²`, quantifies this directly from
the held-out sample's case-frequency weights alone — no LLM output is
involved:

| Log | n (held-out variants) | ESS | Top-1 variant share |
|---|---|---|---|
| RTFM | 46 | 2.1 | 0.67 |
| BPI2019 | 300 | 2.4 | 0.58 |
| BPI2020 | 170 | 8.0 | — |
| BPI2017 | 300 | 35 | — |
| Sepsis | 169 | 93 | — |

On three of the five logs, case-weighted κ rests on an effective sample of
2–8 judgements, so its bootstrap interval cannot support a decision rule.
The RTFM pilot's case-weighted κ for P7 (−0.13; P7 − random1 = −0.03) was
driven by a single test-only variant (`Create Fine > Send Fine`, 20,385
cases) — a held-out variant, so it was absent from the training-only
profile the namer saw.

**Change.**
- The **primary metric** for H-N1, H-N2 and H-N3 is now **unweighted
  variant-level Cohen's κ**: each held-out variant counts as exactly one
  judgement (weight = 1), because the judge's unit of evaluation is the
  variant type, not the case.
- Case-weighted κ, case-weighted accuracy, macro-recall, and the majority
  baseline accuracy are retained and reported as **secondary/descriptive**
  statistics, shown alongside each log's ESS so a reader can see how many
  "independent" judgements they rest on. The paper reports both the primary
  (unweighted) and secondary (case-weighted) numbers.
- The bootstrap is unchanged: 2,000 resamples, paired across partitions,
  seed 20240601 (Amendment 1). The dropped-resample rate continues to be
  recorded for every interval.
- **Low-information flag:** if more than 5% of resamples are dropped for a
  log × comparison (either weighting), that interval is flagged
  `"low_information": true`. The flag is visible in `separability_differences.csv`
  and in `decision.json`; it does not exclude the interval from the
  decision, it only marks it for cautious reading.
- "Unclear" judge answers continue to be scored as incorrect (never equal
  to the true code) under both weighting schemes — unchanged from the
  original pre-specification.

