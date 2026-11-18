# Amendments to PRESPEC.md (supp_bpi2015_english_v1)

PRESPEC.md is preserved byte-identical (SHA-256 in PRESPEC.sha256). Deviations and
corrections are logged here, dated, in order.

## Amendment 1 (2026-11-08): correction of descriptive data counts; no rule change

PRESPEC section 2 quotes "278 distinct English texts; 85 texts shared by more than one
code, covering 205 codes". Those counts came from the pre-PRESPEC inspection, which
assigned each code its lexicographically first English name. They were not computed
under the pre-specified event-modal rule. The `prepare` stage applies the
pre-specified rule (event-modal name, lexicographic tie-break) and gives:
- 281 distinct English texts;
- 84 texts shared by more than one code, covering 201 codes;
- 1,715 of 52,217 events (3.3%) carry a non-modal English name;
- 0 missing names.

The data-preparation rule itself is unchanged. Section 5's "(278)" therefore reads 281.
This correction was recorded before any LLM call or clustering on English labels.
