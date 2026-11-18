# Pre-specification: Supplementary study C, frozen P7 on additional BPI Challenge logs (supp_extra_logs_v1)

Status: written after the logs were downloaded and variant tables were extracted, and
before any activity normalisation, embedding, clustering or metric was computed on the
new logs. Its SHA-256 is recorded in `PRESPEC.sha256`. Later deviations will be
appended under "Amendments" with reasons, and the original text will not be edited.

## 1. Purpose and status

- Apply the frozen P7 pipeline and the main-study baselines, unchanged, to additional
  public BPI Challenge logs used in the trace-clustering literature.
- The new logs are **out of calibration**: neither the frozen prompt and schema nor
  the dynamic-K selector saw them.
- The study is supplementary and independent of the paper text. The PI decides
  later whether to include it. Results are reported regardless of direction,
  including abstention.

## 2. Log selection (decided before any result)

A primary-source check of 11 trace-clustering and representation-learning papers
(full text or the authors' own code or data where available) gave this picture:

- **BPI 2015 is the only candidate log confirmed in clustering evaluations**:
  - De Koninck, vanden Broucke and De Weerdt 2018, act2vec/trace2vec, BPM.
    DOI 10.1007/978-3-319-98648-7_18. Supplementary archive contains `BPIC15GroundTruth.xes`.
  - Luettgen, Seeliger, Nolle and Mühlhäuser 2021, Case2vec, ICPM Workshops, LNBIP 406.
    DOI 10.1007/978-3-030-72693-5_13. Cites BPIC 2015 and BPIC 2019.
  - De Koninck et al. 2021, expert-driven trace clustering, KAIS 63.
    DOI 10.1007/s10115-021-01548-6. MUNICIP = pre-processed BPIC 2015.
- **BPI 2012, 2013, 2014 and 2018 were not confirmed in any checked paper.**
  - Papers with no BPI log at all:
    - ActiTraC, De Weerdt et al. 2013, TKDE, DOI 10.1109/TKDE.2013.64 (private logs).
    - Song et al. 2013, ESWA, DOI 10.1016/j.eswa.2012.12.078 (hospital log).
    - Tavares et al. 2023, EAAI, DOI 10.1016/j.engappai.2023.107028 (synthetic logs).
    - Zandkarimi et al. 2020, ICPM, DOI 10.1109/ICPM49681.2020.00034 (no empirical data).
  - TraCluSI (De Koninck and De Weerdt 2019, ICPM, DOI 10.1109/ICPM.2019.00014)
    uses "four BPIC datasets" of unverified years.
  - Unverified because full text was inaccessible:
    - Chatain et al. 2017, ER.
    - Hompes et al. 2015, BNAIC.

Chosen logs (2):

| Study id | Log | 4TU DOI | Licence | Reason |
|---|---|---|---|---|
| `bpi2015_m1` | BPI Challenge 2015, Municipality 1 (`BPIC15_1.xes`) | 10.4121/uuid:a0addfda-2044-4541-a450-fdcc9fe16d17 (collection 10.4121/uuid:31a308ef-c844-48da-948c-305d167a0ec1) | 4TU General Terms of Use | Strongest confirmed clustering-literature use; municipality 1 is the smallest feasible single-municipality log |
| `bpi2012` | BPI Challenge 2012 (`BPI_Challenge_2012.xes.gz`) | 10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f | 4TU General Terms of Use | Requested by the PI; very widely used in process mining generally (though not confirmed in the checked clustering papers) |

Downloaded but excluded:

- **BPI 2013 incidents** (10.4121/uuid:500573e6-accc-4b0c-9576-aa5468b10cee).
  - Not confirmed in the literature.
  - The frozen preprocessing (`concept:name`, lifecycle `auto`) leaves only **4 activity
    labels** (Accepted/Queued/Completed/Unmatched).
  - A 4-label alphabet makes activity normalisation nearly vacuous, and A2V d16 cannot
    be meaningfully dimension-matched.
- **BPI 2014** (Rabobank, collection 10.4121/uuid:c3e5d162-0cfd-4bb0-bd82-af5268819c35).
  - Not confirmed in the literature.
  - Distributed as CSV only. Conversion to XES would be a new preprocessing step
    absent from the main study.
- **BPI 2018** was not checked further (not confirmed in the literature; CC0).

Overlap note: BPI 2012 comes from the same Dutch financial institution and loan
application process as BPI 2017, which is a main-study calibration log. The two differ
in period (2011–12 vs 2016–17) and IT system, and share part of the label vocabulary
(`A_`, `O_`, `W_` prefixes; BPI 2012 has Dutch `W_` labels). **BPI 2012 is therefore
only partially out-of-calibration and will be reported as such.**

Raw files (SHA-256, downloaded from 4TU; the 4TU-supplied MD5 matched for each):

```
5cd9cc16b9bcb20bd4aae45666a5d87479ddbf47e6371618b6ad217174cecdf3 *datasets/raw/bpi2012/BPI_Challenge_2012.xes.gz
7591f70d4a4d536ead76a56c2fa493fbe22bf2ec7e06c8ddd29da79bd666e516 *datasets/raw/bpi2015_municipality1/BPIC15_1.xes
6ec2136c607648608655802486df97a866f351668751b7ecbca51e1f8715d522 *datasets/raw/bpi2013_incidents/BPI_Challenge_2013_incidents.xes.gz   (excluded)
```

## 3. Preprocessing (identical to the main study)

- Command: `python -m llm_variant_clustering.cli <xes> datasets/processed/<folder> --lifecycle-policy auto`.
  - Activity is `concept:name`.
  - Events are filtered to lifecycle `complete` when such values exist.
  - Events are ordered by case, timestamp, then source order.
  - The variant counts are cross-checked against pm4py.
- All variants are used, with no calibration subset to exclude. Case frequency is the
  analysis weight.
- **BPI 2015 activity labels are opaque codes** (for example `01_HOOFD_010`). The
  English `activityNameEN` attribute is **not** used, because the main-study preprocessing
  uses `concept:name` only.
  - This is a planned stress condition for the label-only normaliser, whose prompt
    instructs it to return "unknown" for opaque labels.
  - No `activityNameEN` sensitivity run is planned.

| Log | Lifecycle applied | Events | Cases | Activities | Variants |
|---|---|---:|---:|---:|---:|
| bpi2012 | complete | 164,506 | 13,087 | 23 | 4,336 |
| bpi2015_m1 | complete | 52,217 | 1,199 | 398 | 1,170 |

Processed tables (SHA-256):

```
11d0a697f3b4fbb5cc9b47277fa83003075a765978b2fcdcc44669a934bea8b3 datasets/processed/bpi2012/variants.parquet
f84026fe23f9299cbc097fd4bde05fb3fb93d66f11ba2e8f0566d16150e15360 datasets/processed/bpi2012/case_variant_map.parquet
7c6a9835edd3c60bf3f339db7fcf47f6a9b1ea82b13880e97cd69569ed0748e4 datasets/processed/bpi2015_municipality1/variants.parquet
f7207d52ccc151b3a6c5c9b1c35fbd08cfef4d7b3c009a96b7f0ca3bb437b4e8 datasets/processed/bpi2015_municipality1/case_variant_map.parquet
```

## 4. Frozen configuration (unchanged)

```
20293d23f795e5b01faedcc30c62b40feb347b35e6e88ebe755db692a67439ec configs/frozen/p7_activity_gpt41_v1/manifest.json
0312fc1bd366d0d30a4e83bc356d7b54aebbdd488db8c58f31f476dd5cf4631a configs/frozen/p7_activity_gpt41_v1/prompt.txt
f134d0a85fc8459472b24c3588606d23eae1574f9603054adc42e17e54f9479a configs/frozen/p7_activity_gpt41_v1/schema.json
03a0b8991a5c04c6752f66368b764b2039c5aba99937232a58fcecd4a17ba0c1 configs/frozen/p7_representation_v1/manifest.json
e277c4acf6210c9fe27977d86f1beb2513e0325704d414a0328cef5433db9400 configs/clustering/dynamic_k_v1.json
```

**Activity normalisation**

- Tool: `scripts/run_p7_activity_normalization.py`, using the frozen prompt and schema
  above (byte-identical to the defaults).
- Model: Azure OpenAI `gpt-4.1`, model version `2025-04-14`, Entra authentication.
- Generation: temperature 0, max output tokens 1000, with validation repairs.
- Calls: one call per distinct label.
- Cache location: `evaluation/supp_extra_logs_v1/<log>/azure_activity_cache`.
- Every label must validate. Any label that fails after the provider's repairs is
  reported and the log is not run (abstention), rather than patched by hand.

**Representation and clustering**

- Tool: `scripts/run_p7_scalable_benchmark.py` with default arguments and `--clusters 8`.
- Embedding: BGE-large-en-v1.5 revision `d4aa6901…`, CPU, no truncation.
- Pyramid: geometry-preserving SVD projection, then an ordered 1-2-4 pyramid with
  cosine distance.
- Clustering: frequency-weighted K-means.
- Automatic K: frozen selector `dynamic_k_v1`, K 2..12, with these gates:
  - stability ARI ≥ 0.8;
  - entropy ≥ 0.75;
  - minimum cluster case fraction 0.02;
  - minimum 2 variants per cluster.
- If no K passes the gates, the method **abstains**. No gate is relaxed.

## 5. Compared sources and matched K

The comparison follows the main-study Table 2.

**Matched K:** P7's automatic K. If P7 abstains, K = 8, marked **exploratory**.

Sources at matched K (activity vector changes; composer, weights, K-means identical):

- **P7** (`p7_llm_semantic_bge`): BGE of the frozen canonical descriptor.
- **Raw label** (`raw_activity_label_bge`): BGE of the original label.
- **Activity2Vec d16** (`activity2vec_log_context_d16`): log-context embeddings,
  window 2.
- **Activity2Vec d{rank}**: Activity2Vec at P7's semantic rank; secondary,
  descriptive.

N-gram TF-IDF (1–3) is reported at **its own** automatic K, or at K = 8 if it abstains.
It is descriptive only, because it uses a different geometry and K.

Implementation: `scripts/run_supp_extra_logs.py`, which imports
`_compact_activity_space`, `_embed` and `_evaluate` from
`run_representation_source_benchmark.py` and the uncertainty functions from
`llm_variant_clustering.uncertainty`. It reimplements only the dataset loop, since the
existing scripts hard-code the main-study logs.

## 6. Metrics, uncertainty and decision wording (as in the main study)

**Case-weighted silhouette**

- Each source's silhouette gets a percentile bootstrap over unique variants, with case
  frequency as weight. Embeddings, K and partitions are held fixed.
- **400** resamples, seed 20240601, nominal 95%.
- For logs with more than 2,500 variants, each replicate is an unrescaled draw of
  m = 2,500 variants. This applies to BPI 2012.

**Paired differences**

- Paired percentile bootstrap of P7 minus each matched-K source, with identical
  variant draws for both partitions and the same settings as above.

**Bonferroni family**

- The family is {P7 − raw, P7 − Activity2Vec d16} × {bpi2012, bpi2015_m1}: **4
  contrasts, 98.75%**.
- Adjusted intervals re-read the same stored replicates (`adjusted_interval`).
- P7 − A2V d{rank} is not in the family.

**Decision wording** (identical to the paper):

- An interval that excludes zero is reported as "P7 higher/lower than X; the paired
  interval excludes zero."
- An interval that includes zero is reported as "not separated". This is an absence of
  separation, not evidence of equivalence; no equivalence margin is specified.
- A contrast is called robust only if it also excludes zero after Bonferroni
  adjustment.
- When P7 abstains, all matched-K results are exploratory.
- Comparisons with n-gram are descriptive.

**Realized K:** the automatic K, or abstention, for P7, raw-label pyramid and n-gram
(from `metrics_dynamic_k.csv`).

**Seed stability**

- Mean pairwise ARI over the 12 main-study seeds, using
  `KMeans(n_init=10, algorithm="lloyd")` with case weights.
- Computed at matched K and at K = 8.
- Intervals come from a seed-level U-statistic bootstrap (`bootstrap_seed_agreement`,
  4,000 resamples).
- Paired P7-minus-source differences use the same seed draws.

**Held-out replay**

- Tool: `scripts/run_heldout_process_evaluation.py`, under `PYTHONHASHSEED=0`, with
  defaults:
  - 20% of unique variants held out, case-fraction tolerance 0.05;
  - shared SHA-256 split with cross-method cluster coverage, maximum 10,000 seeds;
  - Inductive Miner noise 0.2.
- Methods: raw-label pyramid, n-gram and P7 automatic-K assignments. Any method that
  abstains uses its K = 8 assignment, marked exploratory.
- If no shared split covers every cluster of every method, the evaluation
  **abstains**. K is not changed.
- Reported: whole-log, and test-case-weighted aggregate fitness, precision and F1.
- A descriptive cluster-level bootstrap of F1 (`bootstrap_cluster_metric`, test-case
  weights, 2,000 resamples) is given where a method has at least 2 clusters.
- Held-out results are conditional on a transductive partition and are descriptive.

**LLM cost:** cache entries, provider attempts, prompt and completion tokens, read from
cache `provider_metadata.usage`.

**Cluster-naming separability: not run.** It needs two judge models and a separate
pre-specification (`cluster_naming_v1`), so it is not cheap within this study's budget.

## 7. Outputs

All outputs go to `evaluation/supp_extra_logs_v1/`: one folder per log (benchmark
outputs plus `sources/`), aggregate CSVs, and `README.md`. No existing file outside
this folder is modified.

## 8. Code state at pre-specification

```
4ed6569faed715fb737341e57e18409b4c64b38d396979b74952eab7b6a611ac scripts/run_supp_extra_logs.py
```

## Amendments

(none yet)
