# PRESPEC — C4: 5,000-replicate paired bootstrap tails (P7 − raw silhouette)

Written 2026-11-07, before any result of this analysis was computed. Not to be
edited after results are seen; changes go in an "Amendments" section with the
reason.

## Source of the published intervals

`scripts/run_statistical_uncertainty.py` (400 replicates, cap 2,500) calling
`llm_variant_clustering.uncertainty.paired_silhouette_difference` with its
default RNG seed `20240601`; output
`evaluation/statistical_uncertainty_v2/paired_silhouette_differences.csv`;
Bonferroni read from the same replicates (`PairedDifference.adjusted_interval`).

**Family.** In that code the paired P7-vs-raw contrast exists for BPI 2017,
BPI 2019, Sepsis (K=8) and RTFM. BPI 2020 has no frozen raw-label embedding
matrix in `evaluation/full_benchmark/bpi2020/` (`raw_file=None`), so it is not
in the published family; the paper text (workshop draft §5) also names "BPI
2017, BPI 2019, Sepsis, RTFM". The review request listed BPI 2020 instead of
Sepsis; this analysis re-runs the **published family** (BPI 2017, BPI 2019,
Sepsis, RTFM; family size 4, 98.75%). BPI 2020 is reported as not computable
from frozen artifacts.

## Procedure

- Identical inputs to the published run: `_load_variants`, `_labels`,
  `DATASETS` imported from `scripts/run_statistical_uncertainty.py`; cosine
  distances via `sklearn.metrics.pairwise_distances`; frozen partitions
  (`dynamic`; Sepsis `k8`); case frequencies as weights.
- Paired bootstrap exactly as `paired_silhouette_difference`: draws from
  `uncertainty._resample_indices(n, 5000, seed=20240601, sample_size)`; each
  replicate evaluates both partitions on the identical index draw with
  `uncertainty._silhouette_on_subset`; replicates where either partition has
  <2 clusters are skipped (counted). Replicates are computed in parallel
  chunks; the result must equal a serial call of `paired_silhouette_difference`
  (checked on RTFM and Sepsis, exact equality required).
- RNG seed: 20240601 (the published default). Because NumPy draws fill the
  `(resamples, m)` array row-major, replicates 1–400 are the published
  replicates; reproducing the published 95% and 98.75% intervals from the
  first 400 replicates is a pre-specified sanity check.
- Settings: (a) cap m = 2,500 (applies to BPI 2017 n=5,578 and BPI 2019
  n=11,928; Sepsis n=846 and RTFM n=231 are below the cap and use full n);
  (b) full n (no cap) for BPI 2017 and BPI 2019. BPI 2019 full n: time a pilot
  of replicates; if the projected wall-clock with the available parallelism
  exceeds 60 min it is not run, and the cap result plus the existing
  `cap_sensitivity/m5000` result are reported instead.

## Reported per contrast and setting

Point estimate; valid replicates; nominal 95% percentile interval; Bonferroni
98.75% percentile interval (`np.percentile`, same as the code); share of
replicates with difference > 0; number of replicates strictly below the
98.75% lower endpoint and strictly above the upper endpoint; whether each
interval excludes zero. Output: `paired_bootstrap_5000.csv`,
`replicates_<log>_<setting>.npy` (replicate differences), `manifest.json`,
`README.md`.

## Decision wording (fixed in advance)

- A contrast is "separated after Bonferroni adjustment" if its 5,000-replicate
  98.75% interval excludes zero, in the setting reported in the paper (cap
  2,500). If the cap and full-n settings disagree on exclusion, both are
  reported and the contrast is described as "sensitive to the m-out-of-n cap".
- The published statement "BPI 2017 and RTFM remain above zero after
  Bonferroni adjustment" is "confirmed with 5,000 replicates" if both
  98.75% intervals (cap setting) exclude zero, otherwise "not confirmed".
- The published 400-replicate endpoints are compared with the 5,000-replicate
  endpoints; changes are reported numerically without reinterpretation.

## Amendments (appended after results were partly seen)

- **A1 — first-400 extraction with skipped draws.** On RTFM two of the 5,000
  draws are skipped (a partition had <2 clusters in the draw). The "first 400"
  sanity check therefore uses the valid replicates among draws 1–400 (the same
  set the published 400-draw run kept), via a draw-index mapping rather than
  the first 400 valid values.
- **A2 — BPI 2017 full-n re-run.** The first BPI 2017 full-n attempt (8
  workers) crashed with a NumPy `_ArrayMemoryError` due to memory pressure
  from concurrent jobs on the shared machine; it was re-run unchanged with 4
  workers. Nothing from the crashed attempt was used.
- **A3 — BPI 2019 full n not run.** Pilot: 5.97 s per replicate single
  process; projected 62 min at 8 workers; 12 workers (~41 min) were not
  feasible in memory (~3 GB per worker on a shared machine). Per the
  pre-specified >60 min rule, BPI 2019 is reported at the cap plus the existing
  `cap_sensitivity` m=5,000 result (400 replicates).
- **A4 — point estimates** differ from the published CSV by ~1e-8 (float /
  BLAS threading in the distance computation); this does not affect any
  reported digit.
