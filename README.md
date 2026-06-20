# Error Persistence and Regression Burden Release Package

This folder is the public GitHub/SSRN release package for the paper:

**An Item-Level Audit of Error Persistence and Regression Burden in Small Language Model Replacement**

It contains the manuscript, analysis code, derived metrics, figures, tables, and reproducibility commands needed to inspect the current paper results.

This is not the full working repository. Internal review notes, venue strategy files, old experiments, development caches, and unrelated exploratory outputs are intentionally excluded.

## Current Status

This package is intended for public proof-of-analysis on GitHub and for an SSRN preprint. The active readiness target is `github_public_ready`, not a journal archive gate.

Known limitations for the current public package:

- Raw model-output JSONL files are not included in this public package because some model responses echo benchmark question or option text.
- The benchmark is MMLU-Pro; the public source manifest verifies all 3,008 evaluated IDs against the MMLU-Pro test split. The original rationale for selecting exactly these rows and the final redistribution terms still need journal-ready documentation.
- The package now includes a public WILD correctness-only replication: 276,685 item-level records across five small Qwen/Llama models and ten tasks. These rows contain WILD item hashes and binary correctness scores, not prompt text or raw model responses.
- Parser-audit sampling and automated consistency checks are included. The private/source audit sample is withheld because it contains response excerpts. Public files contain only aggregate/redacted summaries. The required 452-row human audit and 100-row second-pass consistency audit have not been completed.
- Scoring-mode sensitivity summaries are included for the runs where saved predictions and raw responses were both available; these are validation artifacts, not a substitute for the manual parser audit.
- Model labels are included exactly as they appear in the raw result files; revision hashes, exact repository IDs, prompt templates, decoding settings, package versions, and run dates were not captured in the raw JSONL files.
- `analysis/artifact_release_status/artifact_release_status.json` is the release gate. It reports `github_public_ready=true` when the public repository URL, licenses, tests, hygiene scan, public manifest, WILD claim gate, and MMLU-Pro source-manifest gate are complete. Parser validation and MMLU-Pro confirmatory status remain visible but non-blocking because the core claim uses parser-independent WILD evidence.

## Folder Map

- `paper/`: manuscript source, PDF, references, generated figures, and generated LaTeX tables.
- `src/boundary_slm/`: analysis and artifact-generation code required for this paper.
- `data/`: reserved for future sanitized or licensed data releases. Raw model-output JSONL files are intentionally withheld from this public package.
- `analysis/raw_tpu_results/`: standardized leaderboard, aggregate per-model summaries, and pairwise error-ecology outputs.
- `analysis/source_manifest/`: MMLU-Pro source-row hashes and source-manifest status.
- `analysis/paper_metrics/`: paper-ready metrics, model-run metadata status, all-pairs appendix metrics, replacement-audit cards, tables, and JSON summaries.
- `analysis/parser_audit/`: public parser-audit summaries, scoring sensitivity summaries, and claim-gate reports. Private first-pass and second-pass audit samples are withheld.
- `analysis/artifact_release_status/`: journal-release metadata and gate status.
- `analysis/external_benchmark_context/`: context-only aggregate benchmark source table.
- `analysis/external_evidence/`: external evidence map, public WILD item-level correctness replication, and gated-source exclusion manifests.
- `outputs/`: reserved for optional CPU reruns; real model rerun outputs are not included.
- `docs/`: public documentation for data status, model runs, benchmark provenance, and reproduction.
- `tests/`: minimal tests for parsing, raw-result analysis, and replacement-audit cards.
- `CITATION.cff`: citation metadata for the GitHub/SSRN artifact.

## Reproduce

From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=src python -m boundary_slm.paper_metrics \
  --input-dir analysis/raw_tpu_results \
  --output-dir analysis/paper_metrics

PYTHONPATH=src python -m boundary_slm.mmlu_pro_manifest \
  --input-dir analysis/raw_tpu_results/per_model \
  --output-dir analysis/source_manifest

PYTHONPATH=src python -m boundary_slm.replacement_audit_cards \
  --input-csv analysis/paper_metrics/enriched_pairwise_metrics.csv \
  --output-dir analysis/paper_metrics

PYTHONPATH=src python -m boundary_slm.external_benchmark_context \
  --output-dir analysis/external_benchmark_context

PYTHONPATH=src python -m boundary_slm.external_evidence_registry \
  --output-dir analysis/external_evidence

PYTHONPATH=src python -m boundary_slm.external_wild_ingest \
  --output-dir analysis/external_evidence

PYTHONPATH=src python -m boundary_slm.open_llm_details_ingest \
  --output-dir analysis/external_evidence/open_llm_details

PYTHONPATH=src python -m boundary_slm.artifact_release_status \
  --root . \
  --output-dir analysis/artifact_release_status \
  --repository-url https://github.com/DsChauhan08/error_persistence_regression_gap \
  --parser-gate-path analysis/parser_audit/parser_audit_claim_gate.json \
  --mmlu-gate-path analysis/parser_audit/mmlu_claim_gate.json \
  --wild-claim-path analysis/external_evidence/external_claim_check.json \
  --source-manifest-path analysis/source_manifest/mmlu_pro_manifest_report.json \
  --hygiene-report-path PUBLIC_RELEASE_HYGIENE.json \
  --public-manifest-path PUBLIC_RELEASE_MANIFEST.json

PYTHONPATH=src python -m boundary_slm.paper_figures \
  --input-dir analysis/paper_metrics \
  --output-dir paper/figures

PYTHONPATH=src pytest -q
```

The following validation commands require withheld private/source files and are therefore not expected to run from the public package alone:

```bash
PYTHONPATH=src python -m boundary_slm.scoring_mode_sensitivity \
  --input-dir data/model_outputs \
  --output-dir analysis/parser_audit

PYTHONPATH=src python -m boundary_slm.parser_audit_impact \
  --sample-csv analysis/parser_audit/parser_audit_sample.csv \
  --output-dir analysis/parser_audit

PYTHONPATH=src python -m boundary_slm.parser_audit_workbook \
  --sample-csv analysis/parser_audit/parser_audit_sample.csv \
  --output-dir analysis/parser_audit

PYTHONPATH=src python -m boundary_slm.parser_audit_second_pass \
  --sample-csv analysis/parser_audit/parser_audit_sample.csv \
  --output-dir analysis/parser_audit

PYTHONPATH=src python -m boundary_slm.mmlu_scoring_robustness \
  --input-dir data/model_outputs \
  --output-dir analysis/parser_audit \
  --parser-gate-path analysis/parser_audit/parser_audit_claim_gate.json
```

## Main Outputs

- `analysis/paper_metrics/family_summary.csv`
- `analysis/paper_metrics/quality_gate_sensitivity.csv`
- `analysis/paper_metrics/model_run_metadata.csv`
- `analysis/paper_metrics/model_run_metadata_summary.csv`
- `analysis/source_manifest/mmlu_pro_manifest_report.json`
- `analysis/source_manifest/mmlu_pro_source_manifest.csv`
- `analysis/external_benchmark_context/external_benchmark_context.csv`
- `analysis/external_evidence/external_evidence_map.csv`
- `analysis/external_evidence/wild_pairwise_replacement_metrics.csv`
- `analysis/external_evidence/wild_task_family_summary.csv`
- `analysis/external_evidence/source_coverage_report.json`
- `analysis/external_evidence/open_llm_details/open_llm_details_exclusions.json`
- `analysis/parser_audit/scoring_mode_sensitivity.csv`
- `analysis/parser_audit/scoring_mode_pairwise_delta.csv`
- `analysis/parser_audit/parser_audit_claim_gate.json`
- `analysis/parser_audit/mmlu_claim_gate.json`
- `analysis/parser_audit/second_pass_parser_audit_public_summary.json`
- `analysis/artifact_release_status/artifact_release_status.json`
- `analysis/paper_metrics/tables/all_pairwise_comparisons.tex`
- `analysis/paper_metrics/replacement_audit_cards.csv`
- `analysis/paper_metrics/replacement_audit_cards.md`
- `analysis/raw_tpu_results/standardized_pairwise_error_ecology_all_models.csv`
- `paper/paper.pdf`

## Raw-Output Regeneration

The script `boundary_slm.raw_results` is included, but rerunning it requires the private/source raw model-output JSONL files. Those files are withheld from this public package until benchmark redistribution terms are finalized.

## Optional CPU Rerun

The CPU rerun harness is included for local validation. It writes `run_manifest.json`, `environment.json`, `selected_item_manifest.csv`, resumable `records.jsonl`, `summary.csv`, `pairwise_cpu_audit.csv`, and `claim_check.json`. The run manifest records model revisions, prompt-template text/hash, decoding settings, tokenizer/chat-template metadata when available, and resume policy.

A small mock smoke run is:

```bash
PYTHONPATH=src python -m boundary_slm.cpu_mmlu_pro_rerun \
  --backend mock \
  --sample-size 5 \
  --stress-size 2 \
  --model mock-a \
  --model mock-b
```

For a real CPU run, omit `--backend mock` and use the default Qwen tiny-model list. Real outputs may contain model responses and should not be added to the public package without redaction review.

## License

Analysis code is released under the MIT License. Data/output licensing is described separately in `DATA_LICENSE.md`.
