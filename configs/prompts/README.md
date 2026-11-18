# Prompt family

Prompts are plain-text templates. Replace angle-bracket placeholders without
changing any other text, then record the exact rendered prompt and its
SHA-256 digest.

| Version | Purpose | Schema |
|---|---|---|
| P1 | Minimal semantic signature | `variant_signature_p1.schema.json` |
| P2 | Evidence-grounded semantic signature | `variant_signature_p2.schema.json` |
| P3 | Ordered structural-semantic signature | `variant_signature_p3.schema.json` |
| P7 | Activity descriptor used by LLM-Desc (frozen copy in `configs/frozen/p7_activity_gpt41_v1`) | `activity_semantics_p7.schema.json` |
| P8 | Business cluster naming (all K clusters of one partition at once, letter-coded, blind to method/dataset) | `business_cluster_names.schema.json` |
| P9 | Held-out judge: match a variant's activity sequence to a candidate business name, or "unclear" | `judge_variant_match.schema.json` |

Prompts receive no dataset name, process description, case metadata, event
metadata, or held-out labels. Naming prompts may receive variant frequency because
frequency is an allowed weight.

P1-P3 are the variant-level signature prompts from prompt development. They are
kept because the development pilot code that the P7 normalizer reuses imports them.

