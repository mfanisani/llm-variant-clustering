# Normalizer and embedder robustness (v1): pre-specification

Written 2026-11-06, **before any generation or embedding in this directory.**
It is a sensitivity analysis for paper 1. It is not confirmatory. It does not
alter any frozen artifact: `configs/frozen/p7_activity_gpt41_v1`,
`evaluation/full_benchmark/*`, `evaluation/statistical_uncertainty_v2/*` are
read only. All outputs go to `evaluation/normalizer_robustness_v1/`.

## Question

Paper 1 (Table 2) reports the paired case-weighted silhouette difference
P7 − raw-label pyramid, with a Bonferroni-adjusted (family of 4) bootstrap
interval:

| Log | Frozen GPT-4.1 point | Bonferroni 98.75% interval | Class |
|---|---|---|---|
| BPI 2017 | +0.109 | [+0.080, +0.120] | positive |
| BPI 2019 | +0.034 | [−0.082, +0.153] | includes zero |
| Sepsis | −0.006 | [−0.022, +0.017] | includes zero |
| RTFM | +0.083 | [+0.008, +0.161] | positive |

Does this pattern depend on the specific normalizing LLM (GPT-4.1) or the
specific embedding model (BGE-large-en-v1.5)?

## Fixed (identical to the frozen pipeline)

- Logs and variant sets (`embedding_index.parquet` where present).
- Frozen prompt and schema: SHA-256 `0312fc1b…` and `f134d0a8…`.
- Canonical descriptor text, rank-preserving compact projection, ordered
  pyramid levels 1-2-4, weighted K-means at matched K (BPI 2020 9, BPI 2017 2,
  BPI 2019 6, Sepsis 8, RTFM 2).
- Raw side for the normalizer analysis: the frozen raw-label pyramid embeddings
  and frozen raw partition, exactly as in `statistical_uncertainty_v2`.
- Bootstrap: 400 resamples, m-out-of-n cap 2500, seed 20240601 (identical
  resample indices for every model), 95% percentile and Bonferroni 98.75%.

## Normalizers (factor 1)

| Key | Model (as returned by the API) | Access | Decoding |
|---|---|---|---|
| `gpt41_rerun` | gpt-4.1-2025-04-14 | Azure OpenAI, Entra | temperature 0, max 1000 tokens, strict tool call |
| `gpt41_mini` | gpt-4.1-mini-2025-04-14 | Azure OpenAI, Entra | same |
| `gpt4o` | gpt-4o-2024-08-06 | Azure OpenAI, Entra | same |
| `qwen25_7b` | Qwen/Qwen2.5-7B-Instruct @ `a09a3545…` | local CPU, open weights | greedy, max 400 new tokens, schema appended to prompt (local-model protocol used for the earlier 1.5B run) |
| `gpt5_minimal` (exploratory) | gpt-5-2025-08-07 | Azure OpenAI, Entra | **protocol deviation:** reasoning model, temperature not settable, `reasoning_effort=minimal`, max 4000 completion tokens |

All use the same three-attempt validation-repair loop as the frozen run. The
returned model string is checked against the table above; a mismatch aborts
the run. The `gpt41_rerun` row measures re-generation reproducibility of the
frozen normalizer itself.

A label that still fails validation after three attempts falls back to the
frozen GPT-4.1 descriptor. Coverage is reported, and a model with coverage
below 100% is labelled `partial_substitution`.

## Embedders (factor 2, applied to the frozen GPT-4.1 descriptors)

| Key | Model | Access |
|---|---|---|
| `te3_small` | text-embedding-3-small | Azure OpenAI, Entra |
| `mpnet` | sentence-transformers/all-mpnet-base-v2 (revision pinned at download) | local, open weights |

For each embedder, both sides are rebuilt with that embedder: P7 descriptors at
the matched K, raw labels at the K of the frozen raw partition. The same
bootstrap is applied.

## Metrics

Per model × log: validated coverage; exact canonical-descriptor identity with
the frozen GPT-4.1 descriptor; per-field agreement; ARI and NMI of the
matched-K partition against the frozen P7 partition; case-weighted silhouette;
paired P7 − raw difference with 95% and Bonferroni intervals. BPI 2020 is
descriptive only (coverage, identity, ARI, silhouette), as in the paper, which
has no paired raw contrast for it.

## Decision rule (fixed now)

Each paired interval is classified as *positive*, *includes zero* or
*negative* by its Bonferroni interval. A model **preserves** a log when its
class equals the frozen class in the table above.

- The paper may say the BPI 2017 and RTFM advantages are **robust to the
  normalizer** only if every complete-coverage model among `gpt41_rerun`,
  `gpt41_mini`, `gpt4o` and `qwen25_7b` preserves *positive* on both logs.
- Otherwise the paper says the advantage is **normalizer-dependent** and names
  the models and logs that fail.
- The same rule applied to `te3_small` and `mpnet` decides whether the paper
  may say the result is **robust to the embedder**.
- `gpt5_minimal` is reported but does not enter either rule.
- The null results on BPI 2019 and Sepsis are reported; a change of class
  there is reported as such and does not rescue or weaken the two positive
  claims.

## Pipeline check (must pass before results are read)

Rebuilding from the frozen GPT-4.1 descriptors with this script must
reproduce the frozen P7 embeddings (`allclose`, atol 1e-5), the frozen P7
partition (ARI = 1), and the `statistical_uncertainty_v2` paired estimates and
Bonferroni bounds (absolute difference < 1e-6). If not, the run stops and the
discrepancy is reported instead of results.

## Limits stated in advance

- All proprietary normalizers are OpenAI models; Qwen2.5-7B is the only other
  vendor. Claude, Mistral and DeepSeek through CAPI were not accessible
  (expired client certificate) and are not part of this analysis.
- Temperature 0 on Azure OpenAI is not bit-deterministic; the rerun quantifies
  this.
- No human ratings are involved or simulated.
- Credentials: Entra ID through the Azure CLI. No key, token or endpoint
  secret is written to any file.

## Amendment 1 (2026-11-06, before any result was read)

The first pipeline check stopped on BPI 2020 only: the frozen file
`full_benchmark/bpi2020/ordered_pyramid_embeddings.npy` stores the
uncompacted 7 × 1024 = 7,168-dimensional pyramid, while the rebuild uses the
rank-preserving compact projection (238 dimensions) that the later logs use.
The activity vectors match the frozen `activity_embeddings.npy` (`allclose`),
the rebuilt partition has ARI = 1 against the frozen partition, and the
pairwise cosine distances agree to 1.5e-6. The embedding criterion is
therefore restated as: maximum absolute difference between the rebuilt and
frozen pairwise cosine-distance matrices ≤ 1e-5. Clustering and silhouette
depend only on these distances. No other criterion changed.

## Amendment 2 (2026-11-06, before any result was read)

A survey of the investigator's own Azure resources found one hosted
non-OpenAI model: DeepSeek-V4-Pro (version 2026-04-23) on the secondary endpoint.
It accepted the frozen protocol unchanged (temperature 0, max 1000 tokens,
strict tool call). It is added as `deepseek_v4_pro` and enters the
normalizer decision rule. Claude via CAPI remains inaccessible (client
certificate expired 2026-08-28).

Model selection is deliberately not exhaustive. One model per question:
re-generation reproducibility (`gpt41_rerun`), scale (`gpt41_mini`),
generation (`gpt4o`), vendor (`deepseek_v4_pro`), small open-weight local
(`qwen25_7b`), reasoning model (`gpt5_minimal`, exploratory).

Because `qwen25_7b` runs on CPU at several minutes per label, the decision
is first computed without it and labelled provisional; it is recomputed
when the Qwen run completes. If Qwen has not completed by 2026-11-12, the
paper reports the decision over the four hosted models and states that the
open-weight run was not completed.
