from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from boundary_slm.io import read_jsonl, write_csv, write_json
from boundary_slm.raw_results import extract_answer_from_raw, infer_model_meta, raw_item_id
from boundary_slm.raw_results import extract_answer, looks_like_prompt_echo_without_completion


HIGH_RISK_METHODS = {
    "tail_option",
    "last_standalone_letter",
    "single_letter",
    "empty",
    "none",
    "prompt_echo_without_completion",
}


def raw_response_prediction(row: dict[str, Any]) -> tuple[str | None, str, float]:
    response = str(row.get("response", row.get("response_text", "")))
    if looks_like_prompt_echo_without_completion(response):
        return None, "prompt_echo_without_completion", 0.0
    return extract_answer(response)


def base_record(row: dict[str, Any], model: str) -> dict[str, Any]:
    meta = infer_model_meta(model)
    prediction, method, confidence = extract_answer_from_raw(row)
    response = str(row.get("response", row.get("response_text", "")))
    return {
        "model": model,
        "family": meta.family,
        "item_id": raw_item_id(row),
        "category": str(row.get("category", row.get("task_family", "unknown"))),
        "ground_truth": str(row.get("ground_truth", row.get("expected", ""))).strip().upper(),
        "parser_prediction": prediction or "",
        "parser_answered": bool(prediction),
        "extraction_method": method,
        "extraction_confidence": confidence,
        "response_chars": len(response),
        "response_excerpt": response[-2500:].replace("\n", " ").strip(),
        "audit_source": "stratified",
        "risk_reason": "",
        "human_prediction": "",
        "human_answered": "",
        "human_parser_correct": "",
        "human_notes": "",
    }


def risk_reason(row: dict[str, Any]) -> str:
    reasons: list[str] = []
    method = str(row.get("extraction_method", ""))
    if method in HIGH_RISK_METHODS:
        reasons.append(f"high_risk_method:{method}")
    saved_prediction = str(row.get("saved_prediction", row.get("prediction", ""))).strip().upper()
    if len(saved_prediction) == 1 and "A" <= saved_prediction <= "J":
        raw_prediction, raw_method, _confidence = raw_response_prediction(row)
        if raw_prediction != saved_prediction:
            reasons.append(f"saved_raw_disagreement:{raw_method}")
    confidence = float(row.get("extraction_confidence", 0.0) or 0.0)
    if confidence < 0.62:
        reasons.append("low_parser_confidence")
    return ";".join(reasons)


def build_audit_rows(input_dir: Path, *, per_stratum: int, high_risk_rows: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    high_risk_candidates: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.jsonl")):
        rows = read_jsonl(path)
        if not rows:
            continue
        model = str(rows[0].get("model") or rows[0].get("model_label") or path.stem)
        for row in rows:
            record = base_record(row, model)
            strata[(str(record["family"]), str(record["extraction_method"]))].append(record)
            reason = risk_reason({**row, **record, "saved_prediction": row.get("prediction", "")})
            if reason:
                high_risk = dict(record)
                high_risk["audit_source"] = "high_risk"
                high_risk["risk_reason"] = reason
                high_risk_candidates.append(high_risk)

    sampled: list[dict[str, Any]] = []
    for key in sorted(strata):
        values = strata[key]
        rng.shuffle(values)
        sampled.extend(values[:per_stratum])
    rng.shuffle(high_risk_candidates)
    seen = {(row["model"], row["item_id"]) for row in sampled}
    for row in high_risk_candidates:
        key = (row["model"], row["item_id"])
        if key in seen:
            continue
        sampled.append(row)
        seen.add(key)
        if sum(1 for item in sampled if item["audit_source"] == "high_risk") >= high_risk_rows:
            break
    sampled.sort(key=lambda row: (row["family"], row["extraction_method"], row["model"], str(row["item_id"])))
    return sampled


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        counts[(str(row["family"]), str(row["extraction_method"]), str(row.get("audit_source", "stratified")))] += 1
    return [
        {"family": family, "extraction_method": method, "audit_source": audit_source, "sampled_rows": count}
        for (family, method, audit_source), count in sorted(counts.items())
    ]


def generate(input_dir: Path, output_dir: Path, *, per_stratum: int, high_risk_rows: int, seed: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_audit_rows(input_dir, per_stratum=per_stratum, high_risk_rows=high_risk_rows, seed=seed)
    write_csv(output_dir / "parser_audit_sample.csv", rows)
    summary = {
        "input_dir": str(input_dir),
        "per_stratum": per_stratum,
        "high_risk_rows_requested": high_risk_rows,
        "seed": seed,
        "sampled_rows": len(rows),
        "high_risk_sampled_rows": sum(1 for row in rows if row.get("audit_source") == "high_risk"),
        "strata": summarize(rows),
        "instructions": [
            "Fill human_prediction with A-J when a valid final option is recoverable by a human.",
            "Fill human_answered with true/false.",
            "Fill human_parser_correct with true when parser_prediction matches the human label and false otherwise.",
            "Use human_notes for prompt echo, ambiguous answer, multiple answer, or other parser failure notes.",
            "Do not use AI-generated labels as human labels.",
            "After first-pass labeling, generate the 100-row second-pass sample for consistency reporting.",
        ],
    }
    write_json(output_dir / "parser_audit_manifest.json", summary)
    readme = (
        "# Parser Audit Sample\n\n"
        "This folder contains a reproducible stratified sample for manual answer-extraction audit.\n"
        "Complete the blank `human_*` columns in `parser_audit_sample.csv`; do not edit parser columns.\n\n"
        "Use `PARSER_AUDIT_LABELING_GUIDE.md` before filling labels. The manuscript should not report "
        "human/parser agreement until all 452 audit rows, including all 120 high-risk rows, are labeled and "
        "the report generator marks the manual claim gate as ready. AI-generated labels must not be reported "
        "as human validation.\n\n"
        "After first-pass labeling, run `python -m boundary_slm.parser_audit_second_pass` to create a 100-row "
        "blind relabeling sample. The second pass is reported as consistency evidence; it does not replace the "
        "required first-pass labels.\n\n"
        "For manageable labeling chunks, run `python -m boundary_slm.parser_audit_workbook --batch-size 50`. "
        "The generated batch CSV files are private because they include response excerpts.\n\n"
        "Suggested acceptance reporting: parser agreement rate overall, by family, and by extraction method; "
        "plus a short list of observed failure modes. Rows tagged `audit_source=high_risk` deliberately target "
        "saved/raw disagreements, fallback extraction, prompt echoes, and low-confidence parser decisions.\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    guide = (
        "# Parser Audit Labeling Guide\n\n"
        "Purpose: measure whether the deterministic parser extracted the same final option letter a human would "
        "recover from the saved response. This is a parser audit, not a benchmark-item re-solving task.\n\n"
        "Rules:\n\n"
        "1. Do not edit parser columns: `parser_prediction`, `parser_answered`, `extraction_method`, or "
        "`extraction_confidence`.\n"
        "2. Read only the response excerpt supplied in the private/source sample. If the excerpt is public-redacted, "
        "return to the private/source sample before labeling.\n"
        "3. Fill `human_prediction` with one A-J letter only when the model's final answer is recoverable.\n"
        "4. Fill `human_answered` with `true` when a final answer is recoverable and `false` otherwise.\n"
        "5. Fill `human_parser_correct` with `true` only when the parser prediction equals the human prediction and "
        "the parser/human answered status is also compatible.\n"
        "6. Use `human_notes` for multiple answers, prompt echo, no final answer, answer changed after reasoning, "
        "or any ambiguity.\n\n"
        "Minimum paper gate: 452 completed first-pass labels, all 120 high-risk rows labeled, and at least 95% "
        "overall parser agreement. AI-generated labels cannot be used as human labels. If agreement is lower, "
        "the paper must report parser-sensitivity results instead of treating MMLU-Pro scoring as validated.\n\n"
        "Second-pass consistency check: after first-pass labels are complete, create a 100-row blind relabeling "
        "sample with `python -m boundary_slm.parser_audit_second_pass`. Fill the `second_human_*` fields without "
        "looking at the first-pass human columns, then regenerate the report. Report prediction, answered-status, "
        "and parser-correct consistency rates as audit reliability evidence.\n\n"
        "Batching workflow: `python -m boundary_slm.parser_audit_workbook --batch-size 50` creates private CSV "
        "chunks under `labeling_batches/` and a redacted progress report. Do not put batch CSV files in the public "
        "release.\n"
    )
    (output_dir / "PARSER_AUDIT_LABELING_GUIDE.md").write_text(guide, encoding="utf-8")
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a manual parser-audit sample.")
    parser.add_argument("--input-dir", type=Path, default=Path("main/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("main/analysis/parser_audit"))
    parser.add_argument("--per-stratum", type=int, default=12)
    parser.add_argument("--high-risk-rows", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260617)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    summary = generate(
        args.input_dir,
        args.output_dir,
        per_stratum=args.per_stratum,
        high_risk_rows=args.high_risk_rows,
        seed=args.seed,
    )
    print(summary)


if __name__ == "__main__":
    main()
