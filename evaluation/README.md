# Evaluation protocols

Every script writes its outputs below `evaluation/`. Outputs are not tracked; only the preregistrations below and one analysis script are part of the repository.

Each supplementary study was specified in `PRESPEC.md` before its results were computed. Later deviations are logged as dated amendments, either inside the PRESPEC or in `AMENDMENTS.md`, and never by silent edits. Where a `PRESPEC*.sha256` file exists, it records the hash taken when the PRESPEC was frozen.

| Folder | Study | Paper | Script |
|---|---|---|---|
| `review_bootstrap_tails_v1/` | 5,000-replicate paired bootstrap of the LLM-Desc minus raw-label silhouette | Sect. 5.1, Table 2 | `evaluation/review_bootstrap_tails_v1/run.py` |
| `supp_semantic_tagger_v1/` | Non-LLM semantic tagger as the descriptor source | Sects. 5.1, 5.2 | `scripts/run_supp_semantic_tagger.py` |
| `cluster_naming_v1/` | Post-hoc cluster names and their separability | Sect. 5.4 | `scripts/run_cluster_naming_separability.py` |
| `normalizer_robustness_v1/` | Other LLM normalizers and embedders | Sect. 6 | `scripts/run_normalizer_robustness.py` |
| `supp_extra_logs_v1/` | Frozen LLM-Desc on BPI 2012 | Sect. 6 | `scripts/run_supp_extra_logs.py` |
| `supp_bpi2015_english_v1/` | BPI 2015 with Dutch codes and English activity names | Sect. 6 | `scripts/run_supp_bpi2015_english.py` |
| `supp_trace_topics_v1/` | Trace-level semantic topic baseline | Sect. 6 | `scripts/run_supp_trace_topics.py` |

## Hashes

| File | SHA-256 |
|---|---|
| `supp_semantic_tagger_v1/PRESPEC.md` | `28aec2d3b9a42613c60fa8626d34b6e55f456ac8436d5ffadca487b99ebb0c18` |
| `supp_trace_topics_v1/PRESPEC.md` | `3dd84d31ce63ef334193387062d3f953d7a9fb9e71aa1567329657d3dc7fba48` |
| `supp_bpi2015_english_v1/PRESPEC.md` | `fe28ae5b387087d2c5af15f2aedc3df8ac9ca1eed6eef9ee8abceed59c709369` |
| `supp_extra_logs_v1/PRESPEC_original.md` (as frozen) | `f0b3bf4f73d1257d3bcc4f0b82a860a449301db255d73542db75b4f06ba04437` |
| `supp_extra_logs_v1/PRESPEC.md` (after Amendments 1 and 2) | `66788e64a4ce73ff4a3dc6424cdf97c1c4fb5d841a1b00790492d62ef08d5ee4` |

### Redacted preregistrations

Two preregistrations named the internal Azure endpoints used for the original runs. In this release the endpoint URLs and region names are replaced with neutral wording such as `<primary Azure OpenAI endpoint>` and "the secondary endpoint"; no other text changed. The hashes of the original files were:

| File | Original SHA-256 |
|---|---|
| `cluster_naming_v1/PRESPEC.md` | `bfc6588770fc50516000c45f7efaaf544eabc07d08891fcfa1197f034c25b08b` |
| `normalizer_robustness_v1/PRESPEC.md` | `e0ff113217db34ea309e652651009e7635b620d38e239a571c23b695c2a5d460` |

`run_cluster_naming_separability.py` asks for the hash of the PRESPEC as it is on disk (`--confirm-prespec`), so the redacted file works as is. Compute it with `python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" evaluation/cluster_naming_v1/PRESPEC.md`.

Some PRESPECs also refer to paths and studies of the wider research project that are not part of this paper. They are kept unchanged as historical records.
