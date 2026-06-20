from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from boundary_slm.io import write_csv, write_json
from boundary_slm.parser_audit_report import truthy


SECOND_PASS_FIELDS = [
    "second_human_prediction",
    "second_human_answered",
    "second_human_parser_correct",
    "second_human_notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def completed_first_pass(row: dict[str, str]) -> bool:
    return truthy(row.get("human_answered", "")) is not None and truthy(row.get("human_parser_correct", "")) is not None


def completed_second_pass(row: dict[str, str]) -> bool:
    return (
        truthy(row.get("second_human_answered", "")) is not None
        and truthy(row.get("second_human_parser_correct", "")) is not None
    )


def normalize_prediction(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if len(text) == 1 and "A" <= text <= "J" else ""


def select_second_pass_rows(rows: list[dict[str, str]], *, sample_size: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    eligible = [dict(row) for row in rows]
    rng.shuffle(eligible)
    high_risk = [row for row in eligible if row.get("audit_source") == "high_risk"]
    standard = [row for row in eligible if row.get("audit_source") != "high_risk"]
    high_risk_target = min(len(high_risk), max(sample_size // 2, 0))

    selected: list[dict[str, str]] = high_risk[:high_risk_target]
    selected_keys = {(row.get("model", ""), row.get("item_id", "")) for row in selected}

    # Fill the remaining slots with a deterministic spread over family/method strata.
    strata: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in standard + high_risk[high_risk_target:]:
        key = (row.get("family", "unknown"), row.get("extraction_method", "unknown"), row.get("audit_source", "unknown"))
        strata[key].append(row)

    while len(selected) < min(sample_size, len(eligible)) and strata:
        progressed = False
        for key in sorted(list(strata)):
            bucket = strata[key]
            if not bucket:
                del strata[key]
                continue
            row = bucket.pop(0)
            row_key = (row.get("model", ""), row.get("item_id", ""))
            if row_key in selected_keys:
                progressed = True
                continue
            selected.append(row)
            selected_keys.add(row_key)
            progressed = True
            if len(selected) >= min(sample_size, len(eligible)):
                break
        if not progressed:
            break

    selected.sort(
        key=lambda row: (
            row.get("audit_source", ""),
            row.get("family", ""),
            row.get("extraction_method", ""),
            row.get("model", ""),
            str(row.get("item_id", "")),
        )
    )
    return selected


def blind_second_pass_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        clean = dict(row)
        for field in ["human_prediction", "human_answered", "human_parser_correct", "human_notes"]:
            clean.pop(field, None)
        for field in SECOND_PASS_FIELDS:
            clean[field] = str(clean.get(field, ""))
        out.append(clean)
    return out


def public_selection_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        counts[(row.get("family", "unknown"), row.get("extraction_method", "unknown"), row.get("audit_source", "unknown"))] += 1
    return [
        {
            "family": family,
            "extraction_method": method,
            "audit_source": source,
            "selected_rows": count,
        }
        for (family, method, source), count in sorted(counts.items())
    ]


def consistency_report(first_pass_rows: list[dict[str, str]], second_pass_rows: list[dict[str, str]]) -> dict[str, Any]:
    first_by_key = {
        (row.get("model", ""), str(row.get("item_id", ""))): row
        for row in first_pass_rows
        if row.get("model", "") and str(row.get("item_id", ""))
    }
    compared = 0
    prediction_agree = 0
    answered_agree = 0
    parser_correct_agree = 0
    disagreement_rows: list[dict[str, Any]] = []

    for second in second_pass_rows:
        key = (second.get("model", ""), str(second.get("item_id", "")))
        first = first_by_key.get(key)
        if not first or not completed_first_pass(first) or not completed_second_pass(second):
            continue
        compared += 1
        first_prediction = normalize_prediction(first.get("human_prediction"))
        second_prediction = normalize_prediction(second.get("second_human_prediction"))
        first_answered = truthy(first.get("human_answered", ""))
        second_answered = truthy(second.get("second_human_answered", ""))
        first_parser_correct = truthy(first.get("human_parser_correct", ""))
        second_parser_correct = truthy(second.get("second_human_parser_correct", ""))
        prediction_ok = first_prediction == second_prediction
        answered_ok = first_answered == second_answered
        parser_correct_ok = first_parser_correct == second_parser_correct
        prediction_agree += int(prediction_ok)
        answered_agree += int(answered_ok)
        parser_correct_agree += int(parser_correct_ok)
        if not (prediction_ok and answered_ok and parser_correct_ok):
            disagreement_rows.append(
                {
                    "model": key[0],
                    "item_id": key[1],
                    "family": second.get("family", ""),
                    "extraction_method": second.get("extraction_method", ""),
                    "audit_source": second.get("audit_source", ""),
                    "first_human_prediction": first_prediction,
                    "second_human_prediction": second_prediction,
                    "first_human_answered": first_answered,
                    "second_human_answered": second_answered,
                    "first_human_parser_correct": first_parser_correct,
                    "second_human_parser_correct": second_parser_correct,
                }
            )

    return {
        "completed_second_pass_rows": compared,
        "prediction_agreement_rate": round(prediction_agree / compared, 6) if compared else None,
        "answered_agreement_rate": round(answered_agree / compared, 6) if compared else None,
        "parser_correct_agreement_rate": round(parser_correct_agree / compared, 6) if compared else None,
        "disagreement_rows": len(disagreement_rows),
        "disagreements": disagreement_rows,
    }


def write_tex(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    consistency = report["consistency"]
    rows = [
        ("Selected rows", report["selected_rows"]),
        ("High-risk selected", report["high_risk_selected_rows"]),
        ("Completed relabel rows", consistency["completed_second_pass_rows"]),
        ("Prediction agreement", _pct(consistency["prediction_agreement_rate"])),
        ("Answered agreement", _pct(consistency["answered_agreement_rate"])),
        ("Parser-correct agreement", _pct(consistency["parser_correct_agreement_rate"])),
        ("Disagreement rows", consistency["disagreement_rows"]),
    ]
    lines = [r"\begin{tabular}{lr}", r"\toprule", r"Field & Value \\", r"\midrule"]
    for key, value in rows:
        lines.append(f"{_latex_escape(key)} & {_latex_escape(value)} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _pct(value: Any) -> str:
    if value in {None, ""}:
        return "--"
    return f"{100.0 * float(value):.1f}%"


def _latex_escape(value: Any) -> str:
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


def generate(
    *,
    sample_csv: Path,
    output_dir: Path,
    sample_size: int = 100,
    seed: int = 20260619,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    first_pass_rows = read_csv(sample_csv)
    selected = select_second_pass_rows(first_pass_rows, sample_size=sample_size, seed=seed)
    second_pass_path = output_dir / "second_pass_parser_audit_sample.csv"
    if second_pass_path.exists():
        second_pass_rows = read_csv(second_pass_path)
    else:
        second_pass_rows = blind_second_pass_rows(selected)
        write_csv(second_pass_path, second_pass_rows)

    consistency = consistency_report(first_pass_rows, second_pass_rows)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_csv": str(sample_csv),
        "second_pass_sample_csv": str(second_pass_path),
        "sample_size_requested": sample_size,
        "seed": seed,
        "selected_rows": len(second_pass_rows),
        "high_risk_selected_rows": sum(1 for row in second_pass_rows if row.get("audit_source") == "high_risk"),
        "selection_summary": public_selection_summary(second_pass_rows),
        "consistency": consistency,
        "status": "ready" if consistency["completed_second_pass_rows"] else "pending_labels",
        "public_text_policy": (
            "The second-pass private sample may contain response excerpts. Public release contains only aggregate "
            "selection and consistency summaries, never response excerpts."
        ),
    }
    write_csv(output_dir / "second_pass_parser_audit_report.csv", report["selection_summary"])
    write_json(output_dir / "second_pass_parser_audit_report.json", report)
    public_summary = dict(report)
    public_summary["second_pass_sample_csv"] = "withheld_private_response_excerpt_file"
    public_summary["consistency"] = {
        key: value
        for key, value in consistency.items()
        if key != "disagreements"
    }
    write_json(output_dir / "second_pass_parser_audit_public_summary.json", public_summary)
    write_tex(output_dir / "second_pass_parser_audit_report.tex", report)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate/report a blind second-pass parser-audit consistency sample.")
    parser.add_argument("--sample-csv", type=Path, default=Path("main/analysis/parser_audit/parser_audit_sample.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("main/analysis/parser_audit"))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260619)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    report = generate(
        sample_csv=args.sample_csv,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(json.dumps({"status": report["status"], "selected_rows": report["selected_rows"]}, indent=2))


if __name__ == "__main__":
    main()
