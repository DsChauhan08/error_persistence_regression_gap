from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from boundary_slm.io import write_csv, write_json
from boundary_slm.parser_audit_report import summarize_manual_audit, truthy
from boundary_slm.raw_results import infer_model_meta, model_order_key


VALID_OPTIONS = set("ABCDEFGHIJ")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def pct(value: Any) -> str:
    if value in {None, ""}:
        return "--"
    try:
        return f"{100.0 * float(value):.1f}"
    except Exception:
        return "--"


def completed_label(row: dict[str, str]) -> bool:
    return truthy(row.get("human_parser_correct", "")) is not None and truthy(row.get("human_answered", "")) is not None


def parser_task_correct(row: dict[str, str]) -> bool:
    parser_answered = truthy(row.get("parser_answered", ""))
    prediction = str(row.get("parser_prediction", "")).strip().upper()
    truth = str(row.get("ground_truth", "")).strip().upper()
    return parser_answered is True and prediction in VALID_OPTIONS and prediction == truth


def human_task_correct(row: dict[str, str]) -> bool:
    human_answered = truthy(row.get("human_answered", ""))
    prediction = str(row.get("human_prediction", "")).strip().upper()
    truth = str(row.get("ground_truth", "")).strip().upper()
    return human_answered is True and prediction in VALID_OPTIONS and prediction == truth


def summary_by_dimension(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    completed = [row for row in rows if completed_label(row)]
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in completed:
        groups[("overall", "all")].append(row)
        groups[("family", row.get("family", "unknown") or "unknown")].append(row)
        groups[("extraction_method", row.get("extraction_method", "unknown") or "unknown")].append(row)
        groups[("audit_source", row.get("audit_source", "unknown") or "unknown")].append(row)

    out: list[dict[str, Any]] = []
    for (dimension, value), vals in sorted(groups.items()):
        parser_correct_rows = 0
        false_correct = 0
        false_incorrect = 0
        false_unanswered = 0
        for row in vals:
            parser_ok = truthy(row.get("human_parser_correct", "")) is True
            parser_correct_rows += int(parser_ok)
            parser_task = parser_task_correct(row)
            human_task = human_task_correct(row)
            false_correct += int(parser_task and not human_task)
            false_incorrect += int(not parser_task and human_task)
            false_unanswered += int(truthy(row.get("parser_answered", "")) is False and truthy(row.get("human_answered", "")) is True)
        n = len(vals)
        out.append(
            {
                "dimension": dimension,
                "value": value,
                "audited_rows": n,
                "parser_agreement_rate": round(parser_correct_rows / n, 6) if n else None,
                "false_correct_rows": false_correct,
                "false_correct_rate": round(false_correct / n, 6) if n else None,
                "false_incorrect_rows": false_incorrect,
                "false_incorrect_rate": round(false_incorrect / n, 6) if n else None,
                "false_unanswered_rows": false_unanswered,
                "false_unanswered_rate": round(false_unanswered / n, 6) if n else None,
            }
        )
    return out


def impact_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    completed = [row for row in rows if completed_label(row)]
    by_family_item: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in completed:
        by_family_item[(row.get("family", "unknown") or "unknown", str(row.get("item_id", "")))].append(row)

    out: list[dict[str, Any]] = []
    for (family, item_id), item_rows in sorted(by_family_item.items()):
        ordered = sorted(
            item_rows,
            key=lambda row: model_order_key(infer_model_meta(row.get("model", ""))),
        )
        for old_idx, old in enumerate(ordered):
            for new in ordered[old_idx + 1 :]:
                old_model = old.get("model", "")
                new_model = new.get("model", "")
                if not old_model or not new_model or old_model == new_model:
                    continue
                parser_old = parser_task_correct(old)
                parser_new = parser_task_correct(new)
                human_old = human_task_correct(old)
                human_new = human_task_correct(new)
                parser_regression = parser_old and not parser_new
                human_regression = human_old and not human_new
                parser_improvement = not parser_old and parser_new
                human_improvement = not human_old and human_new
                false_regression = parser_regression and not human_regression
                false_improvement = parser_improvement and not human_improvement
                if false_regression or false_improvement or parser_regression != human_regression or parser_improvement != human_improvement:
                    out.append(
                        {
                            "family": family,
                            "item_id": item_id,
                            "old_model": old_model,
                            "new_model": new_model,
                            "parser_old_correct": parser_old,
                            "parser_new_correct": parser_new,
                            "human_old_correct": human_old,
                            "human_new_correct": human_new,
                            "parser_regression": parser_regression,
                            "human_regression": human_regression,
                            "false_regression_impact": false_regression,
                            "parser_improvement": parser_improvement,
                            "human_improvement": human_improvement,
                            "false_improvement_impact": false_improvement,
                        }
                    )
    return out


def claim_gate(
    *,
    sample_rows: list[dict[str, str]],
    summary_rows: list[dict[str, Any]],
    impacts: list[dict[str, Any]],
    required_rows: int,
    required_high_risk_rows: int,
    min_agreement: float,
) -> dict[str, Any]:
    overall = next((row for row in summary_rows if row["dimension"] == "overall"), None)
    completed_rows = int(overall["audited_rows"]) if overall else 0
    observed_agreement = overall.get("parser_agreement_rate") if overall else None
    completed_high_risk_rows = sum(
        1 for row in sample_rows if completed_label(row) and row.get("audit_source") == "high_risk"
    )
    false_regression = sum(1 for row in impacts if row["false_regression_impact"])
    false_improvement = sum(1 for row in impacts if row["false_improvement_impact"])
    blockers: list[str] = []
    if completed_rows < required_rows:
        blockers.append(f"{completed_rows} completed labels below required {required_rows}")
    if completed_high_risk_rows < required_high_risk_rows:
        blockers.append(f"{completed_high_risk_rows} completed high-risk labels below required {required_high_risk_rows}")
    if observed_agreement is None or float(observed_agreement) < min_agreement:
        observed = "none" if observed_agreement is None else f"{float(observed_agreement):.3f}"
        blockers.append(f"parser agreement {observed} below required {min_agreement:.3f}")
    if false_regression or false_improvement:
        blockers.append(
            f"manual labels reveal {false_regression} false regression-impact and {false_improvement} false improvement-impact cases"
        )
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parser_validated": not blockers,
        "claim_ready": not blockers,
        "status": "ready" if not blockers else "blocked",
        "required_manual_rows": required_rows,
        "required_high_risk_rows": required_high_risk_rows,
        "completed_manual_rows": completed_rows,
        "completed_high_risk_rows": completed_high_risk_rows,
        "minimum_parser_agreement": min_agreement,
        "observed_parser_agreement": observed_agreement,
        "false_regression_impact_cases": false_regression,
        "false_improvement_impact_cases": false_improvement,
        "blockers": blockers,
    }


def write_latex_table(path: Path, summary_rows: list[dict[str, Any]], gate: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = [row for row in summary_rows if row["dimension"] in {"overall", "audit_source", "extraction_method"}][:8]
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dimension & Value & Rows & Agree & False corr. & False incorr. & False unans. \\",
        r"\midrule",
    ]
    if not selected:
        lines.append(r"overall & pending labels & 0 & -- & -- & -- & -- \\")
    for row in selected:
        lines.append(
            f"{latex_escape(row['dimension'])} & {latex_escape(row['value'])} & "
            f"{row['audited_rows']} & {pct(row['parser_agreement_rate'])} & "
            f"{pct(row['false_correct_rate'])} & {pct(row['false_incorrect_rate'])} & "
            f"{pct(row['false_unanswered_rate'])} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            f"Claim gate & {latex_escape(gate['status'])} & {gate['completed_manual_rows']} & "
            f"{pct(gate['observed_parser_agreement'])} & "
            f"{gate['false_regression_impact_cases']} & {gate['false_improvement_impact_cases']} & -- \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def generate(
    *,
    sample_csv: Path,
    output_dir: Path,
    required_rows: int = 452,
    required_high_risk_rows: int = 120,
    min_agreement: float = 0.95,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_rows = read_csv(sample_csv)
    summary_rows = summary_by_dimension(sample_rows)
    # Keep compatibility with the older report by computing its family/method summary too.
    legacy_summary = summarize_manual_audit(sample_rows)
    impacts = impact_rows(sample_rows)
    gate = claim_gate(
        sample_rows=sample_rows,
        summary_rows=summary_rows,
        impacts=impacts,
        required_rows=required_rows,
        required_high_risk_rows=required_high_risk_rows,
        min_agreement=min_agreement,
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_csv": str(sample_csv),
        "sample_row_count": len(sample_rows),
        "parser_audit_impact": summary_rows,
        "legacy_family_method_summary": legacy_summary,
        "false_regression_improvement_impact": impacts,
        "claim_gate": gate,
        "public_text_policy": "Only aggregate parser-audit impact summaries are public; raw responses and response excerpts remain private.",
    }
    write_csv(output_dir / "parser_audit_impact.csv", summary_rows)
    write_json(output_dir / "parser_audit_impact.json", report)
    write_csv(output_dir / "false_regression_improvement_impact.csv", impacts)
    write_json(output_dir / "parser_audit_claim_gate.json", gate)
    write_latex_table(output_dir / "parser_audit_impact.tex", summary_rows, gate)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report parser-audit impact on regression/improvement claims.")
    parser.add_argument("--sample-csv", type=Path, default=Path("main/analysis/parser_audit/parser_audit_sample.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("main/analysis/parser_audit"))
    parser.add_argument("--required-rows", type=int, default=452)
    parser.add_argument("--required-high-risk-rows", type=int, default=120)
    parser.add_argument("--min-agreement", type=float, default=0.95)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    report = generate(
        sample_csv=args.sample_csv,
        output_dir=args.output_dir,
        required_rows=args.required_rows,
        required_high_risk_rows=args.required_high_risk_rows,
        min_agreement=args.min_agreement,
    )
    print(json.dumps(report["claim_gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
