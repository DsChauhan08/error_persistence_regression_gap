# Parser Audit Labeling Guide

Purpose: measure whether the deterministic parser extracted the same final option letter a human would recover from the saved response. This is a parser audit, not a benchmark-item re-solving task.

Rules:

1. Do not edit parser columns: `parser_prediction`, `parser_answered`, `extraction_method`, or `extraction_confidence`.
2. Read only the response excerpt supplied in the private/source sample. If the excerpt is public-redacted, return to the private/source sample before labeling.
3. Fill `human_prediction` with one A-J letter only when the model's final answer is recoverable.
4. Fill `human_answered` with `true` when a final answer is recoverable and `false` otherwise.
5. Fill `human_parser_correct` with `true` only when the parser prediction equals the human prediction and the parser/human answered status is also compatible.
6. Use `human_notes` for multiple answers, prompt echo, no final answer, answer changed after reasoning, or any ambiguity.

Minimum paper gate: 452 completed first-pass labels, all 120 high-risk rows labeled, and at least 95% overall parser agreement. AI-generated labels cannot be used as human labels. If agreement is lower, the paper must report parser-sensitivity results instead of treating MMLU-Pro scoring as validated.

Second-pass consistency check: after first-pass labels are complete, create a 100-row blind relabeling sample with `python -m boundary_slm.parser_audit_second_pass`. Fill the `second_human_*` fields without looking at the first-pass human columns, then regenerate the report. Report prediction, answered-status, and parser-correct consistency rates as audit reliability evidence.

Batching workflow: `python -m boundary_slm.parser_audit_workbook --batch-size 50` creates private CSV chunks under `labeling_batches/` and a redacted progress report. Do not put batch CSV files in the public release.
