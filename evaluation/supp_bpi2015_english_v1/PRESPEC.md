# PRESPEC: BPI 2015 Municipality 1 with English activity names (supp_bpi2015_english_v1)

Status: new, separately pre-specified, review-motivated follow-up. Written on 2026-11-08,
BEFORE any LLM call on English labels and BEFORE any clustering on English labels.
Its SHA-256 is recorded in `PRESPEC.sha256`. This file is never edited after that;
any deviation is logged as a dated entry in `AMENDMENTS.md`.

## 0. Motivation and relation to Supp C

The advisor review noted that BPI 2015 contains an `activityNameEN` attribute that the
original protocol deliberately excluded. In Supp C (`evaluation/supp_extra_logs_v1`,
log `bpi2015_m1`) activity text was the `concept:name` code (e.g. `01_HOOFD_010`); the
frozen GPT-4.1 normaliser returned all-`unknown` descriptors for all 398 codes and P7
collapsed. That result shows a limit of the code-only input condition. It does not show
that the log has no usable text. This follow-up compares the code-only input with the
documented English-name input. P7, the raw-label embedding and the non-LLM tagger all
receive the same text.

The Supp C result stays as it is. Nothing under `evaluation/supp_extra_logs_v1` is
modified. This study is labelled as new work and is not a re-analysis that replaces
Supp C.

## 1. Questions

- **Q1 (does readable text help?):** raw English-label embedding (raw-En) vs raw code
  embedding (raw-code), same pipeline.
- **Q2 (does the LLM add anything beyond embedding the English labels?):**
  P7-English (P7-En) minus raw-En. Paired bootstrap at matched K = 8.
- **Q3:** tagger-English (tagger-En) minus raw-En. Paired bootstrap at matched K = 8.

## 2. Data preparation (fixed before results)

Data:
- Source: `datasets/raw/bpi2015_municipality1/BPIC15_1.xes`.
- Inputs: the same processed variant table and case map as Supp C
  (`datasets/processed/bpi2015_municipality1/variants.parquet`, `case_variant_map.parquet`).
- Variant set: the same 1,170 variants (from `supp_extra_logs_v1/bpi2015_m1/embedding_index.parquet`).
- Size: 1,199 cases, 52,217 events (all lifecycle `complete`), 398 codes.

Rules:
- **Activity identity is unchanged.** The event classifier stays `concept:name`, so the
  variants, the case-to-variant map, the case weights and the held-out replay
  (discovery and replay on codes) are identical to Supp C. Only the text that the
  text-based methods receive changes.
- **Text per code.** Each code gets the event-modal value of `activityNameEN` over all
  events with that code. Ties are broken by lexicographically smallest string.
  - Observed before any analysis: 21 codes carry more than one English name over time
    (the same 21 also vary in `activityNameNL`).
  - One code has a tie at the mode: `05_EIND_060`, 5 vs 5, which resolves to
    "cancel BAG objects research".
  - The number and share of events whose English name differs from the modal name is reported.
- **Codes that share an English name.** These are kept as separate activities. They
  receive identical text and therefore identical raw-En, P7-En and tagger-En vectors.
  There is no merge, and variant identity is unchanged.
  - Observed before analysis: 278 distinct English texts for 398 codes; 85 texts are
    shared by more than one code, covering 205 codes.
  - This collapse of text distinctions is part of the "English-label input condition"
    and is reported.
- **Missing or blank English name.** The text falls back to the code string, flagged.
  Observed before analysis: 0 such events.
- The English text is passed verbatim (stripped of surrounding whitespace) to the
  normaliser and to the raw embedder. The tagger receives its own authors' cleaning
  (`clean_label`), exactly as in the tagger study.
- The normaliser is called once per distinct English text. Its cache key is the label
  text, so there are about 278 calls.

## 3. Methods (all frozen settings reused)

All variant representations use the same composer: positional pyramid 1-2-4 mean pooling
(`ordered_semantic_pyramid`), cosine distance, case-weighted K-means.

| Method id | Per-activity vector |
|---|---|
| `p7_english_pyramid_bge` (P7-En) | Frozen GPT-4.1 normaliser (configs/frozen/p7_activity_gpt41_v1, prompt SHA-256 `0312fc1b…4631a`, schema `f134d0a8…479a`; prompt NOT changed). Settings: Azure deployment `gpt-4.1`, model version `2025-04-14`, api-version `2025-01-01-preview`, temperature 0, same cache mechanism. Input: each distinct English text in isolation. Output: canonical descriptor (`canonical_activity_text`), embedded with BGE-large-en-v1.5 at revision `d4aa6901d3a41ba39fb536a557fa166f842b0e09`, then compact geometry-preserving projection. |
| `raw_english_label_pyramid_bge` (raw-En) | BGE embedding of the English text itself. |
| `tagger_english_pyramid_bge` (tagger-En) | Rebmann & van der Aa BERT tagger (same commit, model, worker and frozen span-to-schema mapping as `evaluation/supp_semantic_tagger_v1`) applied to the cleaned English text, then canonical descriptor, then BGE. |
| `raw_code_label_pyramid_bge` (raw-code) | BGE embedding of the code string (the Supp C raw method), recomputed. |
| `activity2vec_log_context_d16` (A2V) | Log-context Activity2Vec, window 2, d16. Does not depend on text. Recomputed and verified against Supp C. |
| `ngram_tfidf_1_3` (n-gram) | Activity n-gram TF-IDF on code sequences. Does not depend on text. Recomputed and verified against Supp C. |

Further rules:
- **Abstention rules (same as the main and tagger studies).**
  - P7-En abstains if any distinct English text fails schema validation after the
    normaliser's built-in retries.
  - Tagger-En abstains if coverage is below 100% (any text fails cleaning, tagging or mapping).
- **Degeneracy rule (new, fixed now).** A text representation is *degenerate* if its
  activity space has rank < 2 or its variant feature matrix has fewer than 8 distinct
  rows. A degenerate method's Q2/Q3 contrast is reported as
  `degenerate_not_interpretable`, not as support.
- **Supp C verification.** The raw-code and n-gram partitions are compared to the
  stored Supp C assignments (ARI). The raw-code and A2V K = 8 silhouettes are compared
  to the stored Supp C values.
- **Silhouette.** All silhouettes use the current fixed `frequency_weighted_silhouette`
  (a one-case cluster scores 0). Supp C values were computed before that fix; any
  difference is reported.

## 4. K

- **Dynamic selector.** Frozen `configs/clustering/dynamic_k_v1.json` (SHA-256
  `e277c4ac…9400`), K = 2..12, applied to every method. Reported: selected K or
  abstention, plus the eligible K set.
- **Primary matched K = 8**, as in Supp C, for all methods.
- **Secondary K.** If P7-En's selector picks a K ≠ 8, all paired contrasts are repeated
  at that K. These are descriptive, at the unadjusted 95% level.
- **Held-out replay K.** "P7's K" is P7-En's selected K if it selects one, otherwise 8.

## 5. Estimation and replicates

- **Matched K fits.** `weighted_kmeans` (seeds 17, 29, 43, 59, 71; n_init 10). Distances
  are computed in row chunks of 256 (each matrix is 1,170 × 1,170).
- **Silhouette intervals.** Case-weighted silhouette in each method's own space, with a
  full-size bootstrap: 400 resamples, seed 20240601 (1,170 < 2,500, so no subsampling).
- **Paired contrasts** (`paired_silhouette_difference`, 400 resamples, seed 20240601,
  same resampled variants for both members):
  - Q2: P7-En − raw-En at K = 8, each in its own space (main-study convention).
  - Q3: tagger-En − raw-En at K = 8, own spaces.
  - Bonferroni family = {Q2, Q3} at K = 8. The size is fixed at 2 even if a member is
    not estimable, so the adjusted level is 97.5%. It is computed with
    `PairedDifference.adjusted_interval` on the stored replicates.
  - Descriptive own-space contrasts: raw-En − raw-code (labelled not comparable across
    spaces), P7-En − raw-code, P7-En − A2V, P7-En − n-gram, P7-En − tagger-En.
- **Reference geometries (label-independent).** Every method's partition is also scored
  in two fixed geometries that do not depend on label text: the A2V d16 pyramid
  distances and the n-gram TF-IDF distances. Paired differences (400 resamples) in the
  same geometry:
  - Q1: raw-En − raw-code (primary evidence for Q1);
  - P7-En − raw-En and tagger-En − raw-En (descriptive companions to Q2 and Q3).
- **Agreement.** ARI and NMI between all method pairs at K = 8, and at P7's K if different.
- **Seed stability.** 12 seeds (17 … 151), `KMeans(n_init=10, lloyd)` with case weights,
  mean pairwise ARI, seed-level bootstrap with 4,000 resamples. Paired seed-stability
  differences for Q1, Q2 and Q3. Descriptive.
- **Descriptor quality** for P7-En and tagger-En:
  - Per-field fill rate: share not `unknown` for action, business_object,
    process_stage and explicit_state; share not `none explicit` (and not `unknown`) for
    explicit_role. Also reported: `other` rates for process_stage and explicit_state,
    the share with ambiguity = `none`, and the share with both action and object `unknown`.
  - Each is computed over distinct English texts and over the 398 codes (code-weighted).
  - Counts: number of distinct canonical descriptors vs distinct English texts (278)
    vs codes (398), compared with Supp C P7 on codes (1 distinct descriptor).
- **Cost.** Tokens and provider attempts from the cache metadata. Approximate USD at
  list price ($2 per 1M input tokens, $8 per 1M output tokens), labelled approximate.

## 6. Held-out replay (descriptive, transductive)

- **Split.** The frozen Supp C split, regenerated deterministically as Supp C's baseline
  stage would have produced it: `select_variant_holdout_split(case_map,
  {Supp C n-gram dynamic K2, Supp C raw dynamic K8}, test_fraction=0.2,
  max_seed=10000, case_fraction_tolerance=0.05)`. Supp C never wrote it to disk because
  its replay was stopped.
- **What is replayed.** Partitions at P7's K for P7-En, raw-En, raw-code and tagger-En,
  plus the whole log.
  - Inductive miner with noise 0.2 on train cases; token-based fitness and precision on
    test cases; F1 per cluster; case-weighted aggregate.
  - Discovery and replay use codes, so F1 is comparable across text conditions.
  - PYTHONHASHSEED = 0.
- **Infeasible partitions.**
  - A partition with a single-variant cluster is structurally infeasible and abstains
    from replay. K is not changed.
  - If any remaining replayed partition lacks a train or test variant in some cluster on
    the frozen split, one fallback shared split is chosen with the same function over
    the remaining replayed partitions. All methods and the whole log then use it,
    flagged as `fallback_shared_split_over_replayed_partitions`.
- **Time cap: 4 hours wall clock** for all replay. Workers run in parallel, one per
  method plus the whole log, with per-cluster checkpoints.
  - A method's held-out F1 is reported only if all its clusters finish within the cap.
    Otherwise it is "not completed" (partial clusters are listed but not aggregated).
- **F1 intervals.** Descriptive cluster-level bootstrap (`bootstrap_cluster_metric`,
  2,000 resamples, weights = test cases). F1 differences are point estimates.

## 7. What counts as support

**Q2 and Q3** (K = 8, own-space paired difference first − raw-En):
- `higher`: the 95% interval lies above 0.
- `higher_robust`: the Bonferroni 97.5% interval also lies above 0.
- `lower` / `lower_robust`: symmetric, below 0.
- `not_separated`: the 95% interval includes 0. This is not evidence of equivalence.
- `degenerate_not_interpretable`: the first method meets the degeneracy rule.
- Abstention or non-estimability is reported as such.

Own-space silhouette rewards compact representations and is not a task outcome. Q2 and
Q3 are therefore always read together with the reference-geometry differences, the ARI
with raw-En, descriptor quality and held-out F1. These companions are descriptive and
cannot turn a `not_separated` result into support.

**Q1** (descriptive; own-space silhouettes are not comparable across text spaces):
- `consistent_help` requires both:
  - held-out F1 (raw-En − raw-code, same K and split) ≥ +0.01; and
  - the raw-En − raw-code 95% intervals lie above 0 in BOTH reference geometries
    (A2V d16 and n-gram).
- `consistent_against`: the reversed pattern (F1 difference ≤ −0.01 and both intervals below 0).
- Otherwise: `mixed_or_not_separated`.
- If held-out F1 is not completed, Q1 is reported as `partial_*` from the reference
  geometries alone.
- Also reported: the selected K or abstention for both conditions, and the ARI between
  raw-En and raw-code.

## 8. Not run (stated now)

- The tagger on codes.
- Dutch names.
- Merging codes that share an English name.
- Other K values.
- Other BPI 2015 municipalities.
- Any prompt change.

## 9. Frozen inputs at PRESPEC time (SHA-256)

| File | SHA-256 |
|---|---|
| `datasets/raw/bpi2015_municipality1/BPIC15_1.xes` | `7591f70d4a4d536ead76a56c2fa493fbe22bf2ec7e06c8ddd29da79bd666e516` |
| `datasets/processed/bpi2015_municipality1/variants.parquet` | `7c6a9835edd3c60bf3f339db7fcf47f6a9b1ea82b13880e97cd69569ed0748e4` |
| `datasets/processed/bpi2015_municipality1/case_variant_map.parquet` | `f7207d52ccc151b3a6c5c9b1c35fbd08cfef4d7b3c009a96b7f0ca3bb437b4e8` |
| `configs/frozen/p7_activity_gpt41_v1/prompt.txt` | `0312fc1bd366d0d30a4e83bc356d7b54aebbdd488db8c58f31f476dd5cf4631a` |
| `configs/frozen/p7_activity_gpt41_v1/schema.json` | `f134d0a85fc8459472b24c3588606d23eae1574f9603054adc42e17e54f9479a` |
| `configs/frozen/p7_activity_gpt41_v1/manifest.json` | `20293d23f795e5b01faedcc30c62b40feb347b35e6e88ebe755db692a67439ec` |
| `configs/clustering/dynamic_k_v1.json` | `e277c4acf6210c9fe27977d86f1beb2513e0325704d414a0328cef5433db9400` |
| `evaluation/supp_extra_logs_v1/bpi2015_m1/assignments_ngram_tfidf_1_3_dynamic.parquet` | `9b6e5afed9233d4a07e924489c12c938df1c298bb2ef4af6f7aef6dda4266c56` |
| `evaluation/supp_extra_logs_v1/bpi2015_m1/assignments_raw_activity_label_pyramid_bge_dynamic.parquet` | `ef73be6e23071960be3a0c607b270ad86a5c30836dae8619ac1bc0d60d33d43b` |
| `evaluation/supp_extra_logs_v1/bpi2015_m1/embedding_index.parquet` | `4c23cabc5dcb4b4a0cf704bfb5cf6927ff35d39b0fffdc273de9c49d02874407` |
| `evaluation/supp_extra_logs_v1/PRESPEC.md` | `66788e64a4ce73ff4a3dc6424cdf97c1c4fb5d841a1b00790492d62ef08d5ee4` |
| `scripts/run_supp_semantic_tagger.py` | `f99f6c3002568f88566387403a6b20755f391bbfa134f62bcabca4d81f225030` |
| `scripts/run_p7_activity_normalization.py` | `31f70668304681f835ff81dac9ce5ebc0b13be08e17e801519c82540515f687e` |
| `scripts/run_supp_bpi2015_english.py` (this study, at PRESPEC time) | `9601d94b21470838fea9a19be50eef813f673ac69ec2639ef0fe230cf688c064` |

Before this file was written, the script was smoke-tested only on code-only inputs in a
scratch folder outside `evaluation/`: an identity code→text map, the stored Supp C
code-only P7 descriptors, a forced tagger abstention and a 5-minute replay cap. No
English text was normalised, tagged, embedded or clustered.
