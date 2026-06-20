# Data and Output License Notice

This release separates analysis code from model-output and benchmark-derived files.

## Code

The analysis code under `src/` is released under the MIT License. See `LICENSE`.

## Derived Files

The files under `analysis/raw_tpu_results/`, `analysis/paper_metrics/`, `analysis/parser_audit/`, `analysis/external_benchmark_context/`, and `analysis/external_evidence/` are released as research artifacts for inspection and reproducibility of the accompanying preprint.

These files include model labels, aggregate per-model summaries, category summaries, extracted-prediction summaries, parser-audit summaries, and derived correctness/accounting metrics. They do not intentionally redistribute full benchmark question text, full answer-option text, raw model responses, or item-level response excerpts.

Raw model-output JSONL files and item-level response-tail CSVs are intentionally withheld from this public package because some model responses echo benchmark question or answer-option text. The redacted parser-audit sample preserves audit metadata and response-excerpt hashes, but the excerpt text itself is withheld.

The WILD replication files under `analysis/external_evidence/` are derived from the public `kensho/WILD` dataset, which is listed as Apache-2.0 on Hugging Face. These files contain model labels, task labels, WILD item hashes, token counts, and binary correctness scores. They do not include raw prompt text or raw model responses.

Because MMLU-Pro redistribution terms and the original 3,008-row selection rationale must still be finalized for journal submission, do not apply a blanket open-data license to benchmark-derived material unless those rights are confirmed. If a journal archive is created later, it should specify either:

- the license under which the benchmark-derived material can be redistributed, or
- a restricted-data statement with item identifiers, preprocessing scripts, and access instructions that allow readers to reconstruct the same item set from the original benchmark source.

## Current Release Status

This GitHub/SSRN release is a public proof-of-analysis package. Parser-dependent MMLU-Pro claims remain diagnostic until the human parser audit is completed; the core item-level replacement claim is supported by parser-independent WILD correctness records and source-manifested MMLU-Pro provenance.
