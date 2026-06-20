from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from boundary_slm.io import read_jsonl, write_csv, write_json
from boundary_slm.raw_results import (
    extract_answer,
    extract_answer_from_raw,
    infer_model_meta,
    looks_like_prompt_echo_without_completion,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truthy(value: str) -> bool | None:
    cleaned = str(value).strip().lower()
    if cleaned in {"true", "1", "yes", "y"}:
        return True
    if cleaned in {"false", "0", "no", "n"}:
        return False
    return None


def summarize_manual_audit(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "audited_rows": 0,
            "parser_correct_rows": 0,
            "false_extraction_rows": 0,
            "false_unanswered_rows": 0,
            "human_answered_rows": 0,
            "parser_answered_rows": 0,
            "correctness_changed_by_human_label_rows": 0,
            "parser_correct_but_human_wrong_rows": 0,
            "parser_wrong_but_human_correct_rows": 0,
        }
    )
    for row in rows:
        parser_correct = truthy(row.get("human_parser_correct", ""))
        human_answered = truthy(row.get("human_answered", ""))
        parser_answered = truthy(row.get("parser_answered", ""))
        if parser_correct is None:
            continue
        ground_truth = str(row.get("ground_truth", "")).strip().upper()
        parser_prediction = str(row.get("parser_prediction", "")).strip().upper()
        human_prediction = str(row.get("human_prediction", "")).strip().upper()
        parser_is_correct_for_task = parser_answered is True and parser_prediction == ground_truth
        human_is_correct_for_task = human_answered is True and human_prediction == ground_truth
        key = (row.get("family", "unknown"), row.get("extraction_method", "unknown"))
        for bucket_key in [key, ("__overall__", "__overall__")]:
            bucket = buckets[bucket_key]
            bucket["audited_rows"] += 1
            bucket["parser_correct_rows"] += int(parser_correct)
            if human_answered:
                bucket["human_answered_rows"] += 1
            if parser_answered:
                bucket["parser_answered_rows"] += 1
            if parser_answered and not parser_correct:
                bucket["false_extraction_rows"] += 1
            if parser_answered is False and human_answered:
                bucket["false_unanswered_rows"] += 1
            if parser_is_correct_for_task != human_is_correct_for_task:
                bucket["correctness_changed_by_human_label_rows"] += 1
            if parser_is_correct_for_task and not human_is_correct_for_task:
                bucket["parser_correct_but_human_wrong_rows"] += 1
            if not parser_is_correct_for_task and human_is_correct_for_task:
                bucket["parser_wrong_but_human_correct_rows"] += 1

    out: list[dict[str, Any]] = []
    for (family, method), bucket in sorted(buckets.items()):
        audited = int(bucket["audited_rows"])
        correct = int(bucket["parser_correct_rows"])
        out.append(
            {
                "family": family,
                "extraction_method": method,
                "audited_rows": audited,
                "parser_correct_rows": correct,
                "parser_agreement_rate": round(correct / audited, 6) if audited else 0.0,
                "false_extraction_rows": bucket["false_extraction_rows"],
                "false_extraction_rate": round(bucket["false_extraction_rows"] / audited, 6) if audited else 0.0,
                "false_unanswered_rows": bucket["false_unanswered_rows"],
                "false_unanswered_rate": round(bucket["false_unanswered_rows"] / audited, 6) if audited else 0.0,
                "human_answered_rows": bucket["human_answered_rows"],
                "parser_answered_rows": bucket["parser_answered_rows"],
                "correctness_changed_by_human_label_rows": bucket["correctness_changed_by_human_label_rows"],
                "correctness_changed_by_human_label_rate": round(
                    bucket["correctness_changed_by_human_label_rows"] / audited,
                    6,
                )
                if audited
                else 0.0,
                "parser_correct_but_human_wrong_rows": bucket["parser_correct_but_human_wrong_rows"],
                "parser_wrong_but_human_correct_rows": bucket["parser_wrong_but_human_correct_rows"],
            }
        )
    return out


def manual_claim_gate(manual_rows: list[dict[str, Any]], *, min_rows: int = 452, min_agreement: float = 0.95) -> dict[str, Any]:
    overall = next((row for row in manual_rows if row["family"] == "__overall__"), None)
    if overall is None:
        return {
            "claim_ready": False,
            "status": "blocked_no_completed_human_labels",
            "completed_manual_rows": 0,
            "required_manual_rows": min_rows,
            "minimum_parser_agreement": min_agreement,
            "reason": "No completed human_parser_correct labels were found.",
        }
    completed = int(overall["audited_rows"])
    agreement = float(overall["parser_agreement_rate"])
    if completed < min_rows:
        status = "blocked_insufficient_human_labels"
        reason = f"{completed} completed labels is below the required minimum of {min_rows}."
    elif agreement < min_agreement:
        status = "blocked_parser_agreement_below_threshold"
        reason = f"Parser agreement {agreement:.3f} is below the {min_agreement:.3f} threshold."
    else:
        status = "ready"
        reason = "Manual parser-audit minimum row count and agreement threshold are satisfied."
    return {
        "claim_ready": status == "ready",
        "status": status,
        "completed_manual_rows": completed,
        "required_manual_rows": min_rows,
        "minimum_parser_agreement": min_agreement,
        "observed_parser_agreement": agreement,
        "correctness_changed_by_human_label_rows": overall["correctness_changed_by_human_label_rows"],
        "correctness_changed_by_human_label_rate": overall["correctness_changed_by_human_label_rate"],
        "reason": reason,
    }


def high_confidence_scoring_summary(input_dir: Path, *, threshold: float = 0.62) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "rows": 0,
            "all_parser_correct": 0,
            "high_confidence_answered": 0,
            "high_confidence_correct": 0,
            "low_confidence_rows": 0,
        }
    )
    for path in sorted(input_dir.glob("*.jsonl")):
        raw_rows = read_jsonl(path)
        if not raw_rows:
            continue
        model = str(raw_rows[0].get("model") or raw_rows[0].get("model_label") or path.stem)
        family = infer_model_meta(model).family
        for row in raw_rows:
            truth = str(row.get("ground_truth", row.get("expected", ""))).strip().upper()
            prediction, _method, confidence = extract_answer_from_raw(row)
            bucket = buckets[family]
            bucket["family"] = family
            bucket["rows"] += 1
            bucket["all_parser_correct"] += int(prediction == truth)
            if confidence >= threshold and prediction:
                bucket["high_confidence_answered"] += 1
                bucket["high_confidence_correct"] += int(prediction == truth)
            else:
                bucket["low_confidence_rows"] += 1
    out: list[dict[str, Any]] = []
    for _family, bucket in sorted(buckets.items()):
        rows = int(bucket["rows"])
        high = int(bucket["high_confidence_answered"])
        bucket["all_parser_accuracy"] = round(bucket["all_parser_correct"] / rows, 6) if rows else 0.0
        bucket["high_confidence_accuracy_on_answered"] = round(bucket["high_confidence_correct"] / high, 6) if high else 0.0
        bucket["high_confidence_coverage"] = round(high / rows, 6) if rows else 0.0
        out.append(dict(bucket))
    return out


def saved_prediction_consistency(input_dir: Path) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "rows_with_saved_prediction": 0,
            "raw_response_parseable_rows": 0,
            "saved_prediction_matches_raw_response_parser": 0,
            "saved_prediction_differs_from_raw_response_parser": 0,
            "raw_response_unparseable_rows": 0,
        }
    )
    for path in sorted(input_dir.glob("*.jsonl")):
        raw_rows = read_jsonl(path)
        if not raw_rows:
            continue
        model = str(raw_rows[0].get("model") or raw_rows[0].get("model_label") or path.stem)
        family = infer_model_meta(model).family
        for row in raw_rows:
            saved_prediction = str(row.get("prediction", "")).strip().upper()
            if len(saved_prediction) != 1 or saved_prediction < "A" or saved_prediction > "J":
                continue

            response = str(row.get("response", row.get("response_text", "")))
            if looks_like_prompt_echo_without_completion(response):
                parsed, method, _confidence = None, "prompt_echo_without_completion", 0.0
            else:
                parsed, method, _confidence = extract_answer(response)

            key = (family, method)
            bucket = buckets[key]
            bucket["family"] = family
            bucket["raw_response_extraction_method"] = method
            bucket["rows_with_saved_prediction"] += 1
            if parsed:
                bucket["raw_response_parseable_rows"] += 1
                if parsed == saved_prediction:
                    bucket["saved_prediction_matches_raw_response_parser"] += 1
                else:
                    bucket["saved_prediction_differs_from_raw_response_parser"] += 1
            else:
                bucket["raw_response_unparseable_rows"] += 1

    out: list[dict[str, Any]] = []
    for _key, bucket in sorted(buckets.items()):
        parseable = bucket["raw_response_parseable_rows"]
        matches = bucket["saved_prediction_matches_raw_response_parser"]
        total = bucket["rows_with_saved_prediction"]
        bucket["agreement_rate_among_parseable_raw_responses"] = round(matches / parseable, 6) if parseable else 0.0
        bucket["raw_response_parseable_rate"] = round(parseable / total, 6) if total else 0.0
        out.append(dict(bucket))
    return out


def write_latex_table(path: Path, manual_rows: list[dict[str, Any]], consistency_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Audit & Family/method & Rows & Parseable & Matches & Rate \\\\",
        "\\midrule",
    ]
    if manual_rows:
        for row in manual_rows:
            if row["family"] == "__overall__":
                continue
            lines.append(
                f"Manual & {row['family']} / {row['extraction_method']} & "
                f"{row['audited_rows']} & {row['audited_rows']} & {row['parser_correct_rows']} & "
                f"{100 * row['parser_agreement_rate']:.1f} \\\\"
            )
    else:
        lines.append("Manual & pending human labels & 0 & 0 & 0 & -- \\\\")
    for row in consistency_rows:
        lines.append(
            f"Saved-prediction check & {row['family']} / {row['raw_response_extraction_method']} & "
            f"{row['rows_with_saved_prediction']} & {row['raw_response_parseable_rows']} & "
            f"{row['saved_prediction_matches_raw_response_parser']} & "
            f"{100 * row['agreement_rate_among_parseable_raw_responses']:.1f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_status_table(
    path: Path,
    *,
    sample_rows: list[dict[str, str]],
    manual_rows: list[dict[str, Any]],
    consistency_rows: list[dict[str, Any]],
    confidence_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    completed_manual_rows = sum(int(row["audited_rows"]) for row in manual_rows)
    completed_manual_rows = next(
        (int(row["audited_rows"]) for row in manual_rows if row["family"] == "__overall__"),
        completed_manual_rows,
    )
    saved_rows = sum(int(row["rows_with_saved_prediction"]) for row in consistency_rows)
    high_risk_rows = sum(1 for row in sample_rows if row.get("audit_source") == "high_risk")
    lines = [
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Validation item & Value \\\\",
        "\\midrule",
        f"Audit sample rows & {len(sample_rows)} \\\\",
        f"High-risk audit rows & {high_risk_rows} \\\\",
        f"Completed human-labeled rows & {completed_manual_rows} \\\\",
        f"Saved-prediction consistency rows & {saved_rows} \\\\",
        f"High-confidence sensitivity families & {len(confidence_rows)} \\\\",
        f"Manual audit ready & {'yes' if completed_manual_rows > 0 else 'no'} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def generate(input_dir: Path, sample_csv: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_rows = read_csv(sample_csv)
    manual_rows = summarize_manual_audit(sample_rows)
    consistency_rows = saved_prediction_consistency(input_dir)
    confidence_rows = high_confidence_scoring_summary(input_dir)
    write_csv(output_dir / "manual_parser_audit_summary.csv", manual_rows)
    write_csv(output_dir / "saved_prediction_consistency.csv", consistency_rows)
    write_csv(output_dir / "high_confidence_scoring_summary.csv", confidence_rows)
    write_latex_table(output_dir / "parser_audit_report.tex", manual_rows, consistency_rows)
    write_status_table(
        output_dir / "parser_validation_status.tex",
        sample_rows=sample_rows,
        manual_rows=manual_rows,
        consistency_rows=consistency_rows,
        confidence_rows=confidence_rows,
    )
    gate = manual_claim_gate(manual_rows)
    completed_manual_rows = int(gate["completed_manual_rows"])
    public_summary = {
        "completed_manual_audit_rows": completed_manual_rows,
        "manual_claim_gate": gate,
        "manual_parser_audit_summary": manual_rows,
        "saved_prediction_consistency_rows": sum(
            int(row["rows_with_saved_prediction"]) for row in consistency_rows
        ),
        "high_confidence_sensitivity_family_count": len(confidence_rows),
        "public_text_policy": "No raw responses or response excerpts are included in this public parser-audit summary.",
    }
    write_json(output_dir / "parser_audit_public_summary.json", public_summary)
    report = {
        "input_dir": str(input_dir),
        "sample_csv": str(sample_csv),
        "completed_manual_audit_rows": completed_manual_rows,
        "manual_audit_ready_for_submission": bool(gate["claim_ready"]),
        "manual_claim_gate": gate,
        "manual_parser_audit_summary": manual_rows,
        "saved_prediction_consistency": consistency_rows,
        "high_confidence_scoring_summary": confidence_rows,
        "notes": [
            "The saved-prediction consistency check is not a substitute for a manual parser audit.",
            "A manuscript should report manual parser-audit results only after human_* columns are completed.",
        ],
    }
    write_json(output_dir / "parser_audit_report.json", report)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize parser-audit and saved-prediction consistency checks.")
    parser.add_argument("--input-dir", type=Path, default=Path("main/results"))
    parser.add_argument("--sample-csv", type=Path, default=Path("main/analysis/parser_audit/parser_audit_sample.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("main/analysis/parser_audit"))
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    report = generate(args.input_dir, args.sample_csv, args.output_dir)
    print(report)


if __name__ == "__main__":
    main()
