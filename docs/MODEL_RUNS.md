# Model Runs

The public release includes aggregate per-model summaries and pairwise metrics derived from 14 raw model-output runs. The raw JSONL files themselves are withheld from this public package because some model responses echo benchmark question or answer-option text.

The source raw JSONL files preserve the model labels used during analysis. They do not include full model-card URLs or immutable revision hashes. Before journal submission, the run metadata should be expanded with exact source identifiers, revision hashes where available, prompting template, decoding settings, and runtime details.

The public package now includes explicit metadata-completeness outputs:

- `analysis/paper_metrics/model_run_metadata.csv`
- `analysis/paper_metrics/model_run_metadata_summary.csv`

These files show that item IDs, categories, ground-truth labels, model labels, and raw responses were captured in the private/source raw files, while exact HF repositories, revision hashes, prompt/chat templates, decoding parameters, runtime/package versions, and run dates were not captured.

## Included Model Labels

| Model label | Family | Public release status |
|---|---|---|
| Gemma-2-2B-Instruct | Gemma | Aggregate summary included; raw JSONL withheld |
| Gemma-3-1B-Instruct | Gemma | Aggregate summary included; raw JSONL withheld |
| Gemma-3-1B-it | Gemma | Aggregate summary included; raw JSONL withheld |
| Gemma-3-270M-Instruct | Gemma | Aggregate summary included; raw JSONL withheld |
| Gemma-4-E2B-Instruct | Gemma | Aggregate summary included; raw JSONL withheld |
| Llama-3.2-1B-Instruct | Llama | Aggregate summary included; raw JSONL withheld |
| Phi-2 | Phi | Aggregate summary included; raw JSONL withheld |
| Qwen2-0.5B-Instruct | Qwen | Aggregate summary included; raw JSONL withheld |
| Qwen2.5-0.5B-Instruct | Qwen | Aggregate summary included; raw JSONL withheld |
| Qwen2.5-3B-Instruct | Qwen | Aggregate summary included; raw JSONL withheld |
| Qwen3-0.6B | Qwen | Aggregate summary included; raw JSONL withheld |
| Qwen3-4B-Instruct-2507 | Qwen | Aggregate summary included; raw JSONL withheld |
| Qwen3.5-0.8B | Qwen | Aggregate summary included; raw JSONL withheld |
| Qwen3.5-2B | Qwen | Aggregate summary included; raw JSONL withheld |

## Analysis Roles

The paper retains all 14 model-output runs in descriptive outputs. The primary same-family pairwise analysis applies an answered-rate quality gate of 0.80. Under that gate:

- Qwen has 7 eligible models and 21 same-family directed comparisons.
- Gemma has 4 eligible models and 6 same-family directed comparisons.
- Llama and Phi are retained as single-model anchors but do not support same-family pairwise claims.
- Gemma-4-E2B-Instruct is retained in descriptive outputs but excluded from primary pairwise claims because its answered rate is below the quality gate.

## Direction Status

The primary analysis is a standardized all-pairs directed audit, not a reconstruction of vendor release chronology. Pairwise outputs include `path_interpretation` to distinguish plausible size/generation paths from all-pairs diagnostic comparisons where chronology is unclear.
