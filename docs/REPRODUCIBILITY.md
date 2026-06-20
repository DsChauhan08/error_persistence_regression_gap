# Reproducibility Notes

This release is designed so the paper's derived metrics, audit-card outputs, and figures can be regenerated from public derived outputs. Raw model-output JSONL files are not included in this public package because some responses echo benchmark question or answer-option text.

## Environment

Tested with Python 3.12. The core analysis uses mostly standard Python. Additional packages are used for source-manifest reconstruction, figures, tests, and optional CPU reruns:

- `huggingface_hub`, `pandas`, and `pyarrow` for MMLU-Pro source-manifest reconstruction.
- `pandas`, `pyarrow`, and Hugging Face file access for the public WILD correctness-only replication.
- `matplotlib` for figure regeneration.
- `torch` and `transformers` for optional real CPU reruns.
- `pytest` for tests.

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Regenerate Public Analysis Outputs

```bash
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
```

This regenerates the all-pairs appendix table, model-run metadata status files, quality-gate summaries, replacement-audit cards, MMLU-Pro source-manifest hashes, context-only external benchmark table, external evidence map, WILD correctness-only replication files, gated-source exclusion manifests, and paper-ready CSV/JSON metrics from the public derived inputs.

## Regenerate Parser-Audit Files From Private/Source Outputs

The commands below require the withheld raw model-output JSONL files. They document the regeneration path, but they will not run from this public folder unless those source files are supplied.

```bash
PYTHONPATH=src python -m boundary_slm.raw_results \
  --input-dir data/model_outputs \
  --output-dir analysis/raw_tpu_results \
  --bootstrap-iters 1000

PYTHONPATH=src python -m boundary_slm.parser_audit \
  --input-dir data/model_outputs \
  --output-dir analysis/parser_audit

PYTHONPATH=src python -m boundary_slm.parser_audit_report \
  --input-dir data/model_outputs \
  --sample-csv analysis/parser_audit/parser_audit_sample.csv \
  --output-dir analysis/parser_audit

PYTHONPATH=src python -m boundary_slm.scoring_mode_sensitivity \
  --input-dir data/model_outputs \
  --output-dir analysis/parser_audit
```

The parser-audit sample itself is withheld from the public release because it contains raw response excerpts. This public package includes parser-audit manifests, aggregate summaries, labeling instructions, and the claim gate status only. Regenerate or access the private/source sample before filling the `human_*` columns or claiming human parser-audit agreement.
The scoring-mode sensitivity command also requires private/source raw model-output files because it compares saved prediction fields with raw-response parsing.

## Regenerate Figures

```bash
PYTHONPATH=src python -m boundary_slm.paper_figures \
  --input-dir analysis/paper_metrics \
  --output-dir paper/figures
```

## Optional CPU Rerun

The CPU rerun harness writes `run_manifest.json`, `environment.json`, `selected_item_manifest.csv`, resumable `records.jsonl`, `summary.csv`, `pairwise_cpu_audit.csv`, and `claim_check.json`. The run manifest records model revisions, prompt-template text/hash, decoding settings, tokenizer/chat-template metadata when available, and resume policy.

Mock smoke run:

```bash
PYTHONPATH=src python -m boundary_slm.cpu_mmlu_pro_rerun \
  --backend mock \
  --sample-size 5 \
  --stress-size 2 \
  --model mock-a \
  --model mock-b
```

Real CPU rerun:

```bash
PYTHONPATH=src python -m boundary_slm.cpu_mmlu_pro_rerun
```

Real rerun outputs can contain model responses. Review and redact them before adding them to a public release.

## Run Tests

```bash
PYTHONPATH=src pytest -q
```

## Release Manifest

`PUBLIC_RELEASE_MANIFEST.json` lists every file included in this public package with byte counts and SHA256 hashes.
