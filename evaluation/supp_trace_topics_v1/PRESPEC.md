# Pre-specification: Semantic Trace Topics supplementary baseline (supp_trace_topics_v1)

Status: written before any topic encoding was computed on any study log.
Date: 2026-11-07. Author of run: independent experiment agent (Task B).
Its SHA-256 is recorded in `manifest.json` and in the PI handoff. Any later
deviation is listed in `README.md` under "Deviations from PRESPEC".

## 1. Method being reproduced

Andrienko, Andrienko, Resinas, van den Elzen, Weber, "Semantic Trace Topics:
Text-Based Encodings for Interpretable Multi-Faceted Process Exploration",
EuroVis Workshop VIPRA 2026, doi 10.2312/vipra.20261000, and its follow-up
TracePrism (arXiv 2610.28532). Each trace is a text document over a domain
vocabulary, which is weighted with TF-IDF and factorised with NMF. Each trace
becomes a vector of topic weights.

Sources used to fix the settings:
- The TracePrism paper text.
- The public TracePrism workbench (`compute_worker.js`).
- Its design-decision document, which records how the JS port differs from
  the original VIPRA Python pipeline.

According to that document, the original pipeline used:
- scikit-learn TF-IDF defaults (`idf = ln((1+n)/(1+df)) + 1`, raw term
  frequency, L2-normalised rows);
- NMF with plain NNDSVD initialisation and scikit-learn's default
  coordinate-descent solver.

The JS port replaced the NMF settings with NNDSVDa initialisation and
multiplicative updates fixed at 50 iterations.

## 2. Encoding (primary: `stt_nmf`)

- **Documents.** One document per unique variant (activity sequence) of the
  frozen study variant set. The variant set is
  `datasets/processed/<folder>/variants.parquet`, filtered by
  `evaluation/full_benchmark/<log>/embedding_index.parquet` where that file
  exists, in that row order. Case frequency is not used to build the
  vocabulary or IDF. It enters only as the K-means sample weight and the
  silhouette weight, exactly like `ngram_tfidf_1_3`.
- **Vocabulary (control flow only).**
  - `act:<a>`: one token per event occurrence.
  - `2g:<a>-><b>`: one token per consecutive pair.
  - `3g:<a>-><b>-><c>`: one token per consecutive triple.
  - No lagged-follow, timing, season or attribute tokens. The comparison is
    restricted to control flow plus labels.
  - No min_df or max_df pruning, and no stop tokens.
- **Term weights.** `TfidfTransformer(norm="l2", use_idf=True,
  smooth_idf=True, sublinear_tf=False)` on raw token counts.
- **NMF (original Python pipeline settings).**
  `sklearn.decomposition.NMF(n_components=T, init="nndsvd", solver="cd",
  beta_loss="frobenius", max_iter=200, tol=1e-4, alpha_W=0, alpha_H="same",
  l1_ratio=0, random_state=0)`. Where the original value is not documented,
  the scikit-learn default is used.
- **Representation.** The variant's row of `W` (T non-negative topic weights,
  not normalised). As in TracePrism, K-means uses Euclidean distance on raw
  `W`.

## 3. Topic-count rule (fixed in advance; uses no evaluation metric)

- **Candidate grid.** T in {5, ..., 21}, which is the TracePrism default
  topic-sweep range. T is truncated to `T <= min(n_variants, vocabulary) - 1`.
- **Error curve.** For each T, fit the NMF above and record the relative
  reconstruction error `e(T) = ||V - W H||_F / ||V||_F`.
- **Knee rule (maximum distance below the chord).**
  - Normalise `x = (T - T_min) / (T_max - T_min)` and
    `y = (e(T_min) - e(T)) / (e(T_min) - e(T_max))`.
  - Choose `T* = argmax_T (y - x)`. Ties go to the smaller T.
- **Fallback.** If `max(y - x) <= 0`, or `e(T_min) <= e(T_max)` (no
  decreasing curve), use `T* = 10`.

T* is chosen once per log and per NMF configuration. Silhouette, F1,
stability, the comparison methods and the frozen P7 results are never
consulted.

## 4. Clustering

- **Matched K.** Use `llm_variant_clustering.clustering.weighted_kmeans`
  unchanged: Lloyd K-means, seeds (17, 29, 43, 59, 71), `n_init=10`, case
  frequency as `sample_weight`, and the lowest-inertia run is kept. It runs
  at the frozen matched K: bpi2020 9, bpi2017 2, bpi2019 6, sepsis 8, rtfm 2.
- **Automatic K.** Use `select_dynamic_k` with the frozen
  `configs/clustering/dynamic_k_v1.json`. This mirrors the "dynamic"
  partitions that Table 3 evaluates. If no K is eligible, this is reported as
  `no_eligible_k`.

## 5. Metrics and comparisons

Comparators:
- **P7.** `p7_llm_semantic_bge` / `p7_ordered_pyramid_1_2_4_bge` (ordered
  pyramid of BGE descriptor embeddings, cosine, matched K), recomputed with
  the existing code. Before use, the labels and silhouette must reproduce the
  frozen values (bpi2020 0.5714, bpi2017 0.6281, bpi2019 0.4689, sepsis
  0.3164, rtfm 0.8823; ARI = 1 with the frozen matched-K assignments).
- **n-gram.**
  - (a) `ngram_tfidf_1_3` recomputed at matched K (cosine).
  - (b) The frozen n-gram own-K partition used in Table 2:
    `full_benchmark/<log>/assignments_ngram_tfidf_1_3_dynamic.parquet`, or
    `_k8` for sepsis.
- **Raw and A2V.** Frozen point values from `paper_tables_v1` are shown for
  context only.

**M1. Case-weighted silhouette** (`evaluate_partition`), computed in each
method's native space. This is the main-study convention: topic weights are
Euclidean, P7 and n-gram are cosine. The topic CI uses `bootstrap_silhouette`
with 400 resamples, seed 20240601 and 95% confidence. For logs with more than
2500 variants (bpi2017, bpi2019), the bootstrap is an unrescaled m-out-of-n
with m = 2500, as in `statistical_uncertainty_v2`.

**M2. Paired silhouette differences.** `paired_silhouette_difference`, with
the first method always the topic encoding (difference = topic - comparator).
Comparators are P7 matched K, n-gram matched K and n-gram frozen own K. Each
side uses its own native distances and the same variant draws. Same settings
as M1. Point estimate plus 95% percentile CI, with no multiplicity adjustment
(Bonferroni over 3 comparisons per log is also reported from the same
samples).

**M3. Held-out process-model F1** (bpi2020, bpi2017, bpi2019, rtfm; not
sepsis, where the frozen held-out split abstained).
- **Split.** Use the frozen variant split
  `full_benchmark/<log>/process_model_metrics_dynamic_heldout_variant_split.parquet`.
  First check that every topic cluster has at least one train and one test
  variant (`_clusters_cover_both_splits`).
- **Evaluation.** `evaluate_process_partition_holdout` with noise 0.2,
  inductive miner, run under `PYTHONHASHSEED=0`. The same run also
  re-evaluates the frozen dynamic assignments of P7, n-gram and raw (as
  listed in the frozen held-out manifest) plus the whole-log model. Those
  same-environment values are the comparators. The frozen Table 3 values are
  shown next to them as a reproduction check.
- **Partitions evaluated.** Primary: the topic encoding's automatic-K
  partition, for parity with Table 3. Secondary: its matched-K partition.
- **Aggregate.** Test-case-weighted mean over clusters. The CI is
  `bootstrap_cluster_metric` over cluster rows weighted by `test_cases`, with
  400 resamples, seed 20240601. Differences against P7 and n-gram are
  reported as point differences only. Clusters are not paired across
  methods, so no paired F1 bootstrap exists. This limitation is stated.
- **Coverage fallback.** If the frozen split does not cover a topic
  partition, that cell is reported as `infeasible_on_frozen_split`. As a
  secondary analysis, a fresh split is selected with
  `select_variant_holdout_split` over the topic partition plus the frozen
  comparator partitions (test fraction 0.2, tolerance 0.05). All methods are
  then re-evaluated on that split and labelled as such.

**M4. Seed stability.**
- (a) 5-seed mean pairwise ARI from `weighted_kmeans` at matched K (as in the
  source benchmark).
- (b) 12-seed ARI matrix with `SEEDS` from `run_reseeding_stability`, using
  `KMeans(n_init=10, lloyd, sample_weight)` exactly as in
  `run_seed_level_stability`, and `bootstrap_seed_agreement` with 4000
  resamples:
  - at K = 8 for bpi2020, bpi2017, sepsis and rtfm, paired against the
    frozen P7 matrices in `statistical_uncertainty_v2/seed_stability/ari_matrices.npz`
    (the P7 matrices are recomputed and must match);
  - at matched K for all five logs, paired against P7 and n-gram recomputed
    at matched K (bpi2020 K = 9 is also checked against the frozen
    `bpi2020_matched_k9` P7 matrix).
  - The difference is topic - comparator.

## 6. Secondary (sensitivity) analyses, declared now

Each of S1–S3 reports M1, M2 (P7 and n-gram matched K) and M4(a) only:
- **S1 `stt_nmf_traceprism`.** The NMF of the TracePrism JS port:
  `init="nndsvda", solver="mu", beta_loss="frobenius", max_iter=50, tol=0`.
  T* is chosen by the same rule on its own error curve.
- **S2 `stt_nmf_cosine`.** Primary `W` with L2-normalised rows; K-means on
  the normalised rows and cosine silhouette. This is the dense-representation
  convention of the main study.
- **S3 fixed T.** T in {5, 10, 15, 20} with primary settings. Point
  estimates only (M1 and M4(a)); no CIs.

## 7. Reporting

- All results are reported neutrally, whatever their direction.
- No result is used to change T*, the vocabulary or any setting above.
- No frozen artifact or existing script is modified.
- All outputs go under `evaluation/supp_trace_topics_v1/`.
- No LLM API calls are made. The local BGE model is used only to rebuild the
  frozen P7 comparator.
