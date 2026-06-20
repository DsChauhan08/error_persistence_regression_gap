# Parser Audit Sample

This folder contains a reproducible stratified sample for manual answer-extraction audit.
Complete the blank `human_*` columns in `parser_audit_sample.csv`; do not edit parser columns.

Use `PARSER_AUDIT_LABELING_GUIDE.md` before filling labels. The manuscript should not report human/parser agreement until all 452 audit rows, including all 120 high-risk rows, are labeled and the report generator marks the manual claim gate as ready. AI-generated labels must not be reported as human validation.

After first-pass labeling, run `python -m boundary_slm.parser_audit_second_pass` to create a 100-row blind relabeling sample. The second pass is reported as consistency evidence; it does not replace the required first-pass labels.

For manageable labeling chunks, run `python -m boundary_slm.parser_audit_workbook --batch-size 50`. The generated batch CSV files are private because they include response excerpts.

Suggested acceptance reporting: parser agreement rate overall, by family, and by extraction method; plus a short list of observed failure modes. Rows tagged `audit_source=high_risk` deliberately target saved/raw disagreements, fallback extraction, prompt echoes, and low-confidence parser decisions.
