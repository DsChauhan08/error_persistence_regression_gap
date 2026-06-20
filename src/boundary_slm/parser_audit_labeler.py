from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from boundary_slm.io import write_json
from boundary_slm.parser_audit_report import truthy


VALID_OPTIONS = set("ABCDEFGHIJ")
HUMAN_FIELDS = ["human_prediction", "human_answered", "human_parser_correct", "human_notes"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def completed(row: dict[str, str]) -> bool:
    return truthy(row.get("human_answered", "")) is not None and truthy(row.get("human_parser_correct", "")) is not None


def parser_matches_human(row: dict[str, str], *, human_prediction: str, human_answered: bool) -> bool:
    parser_answered = truthy(row.get("parser_answered", ""))
    parser_prediction = str(row.get("parser_prediction", "")).strip().upper()
    if parser_answered is not human_answered:
        return False
    if not human_answered:
        return True
    return parser_prediction == human_prediction


def apply_label(
    row: dict[str, str],
    *,
    human_prediction: str,
    human_answered: bool,
    human_notes: str = "",
) -> dict[str, str]:
    prediction = human_prediction.strip().upper()
    if human_answered and prediction not in VALID_OPTIONS:
        raise ValueError("human_prediction must be one A-J letter when human_answered=true")
    if not human_answered:
        prediction = ""
    updated = dict(row)
    for field in HUMAN_FIELDS:
        updated.setdefault(field, "")
    updated["human_prediction"] = prediction
    updated["human_answered"] = "true" if human_answered else "false"
    updated["human_parser_correct"] = "true" if parser_matches_human(row, human_prediction=prediction, human_answered=human_answered) else "false"
    updated["human_notes"] = human_notes.strip()
    return updated


def progress(rows: list[dict[str, str]]) -> dict[str, Any]:
    total = len(rows)
    done = sum(1 for row in rows if completed(row))
    high_risk = [row for row in rows if row.get("audit_source") == "high_risk"]
    high_done = sum(1 for row in high_risk if completed(row))
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "total_rows": total,
        "completed_rows": done,
        "remaining_rows": total - done,
        "high_risk_rows": len(high_risk),
        "completed_high_risk_rows": high_done,
        "remaining_high_risk_rows": len(high_risk) - high_done,
        "completion_rate": round(done / total, 6) if total else None,
        "high_risk_completion_rate": round(high_done / len(high_risk), 6) if high_risk else None,
    }


def next_indices(rows: list[dict[str, str]], *, high_risk_first: bool, limit: int | None) -> list[int]:
    indexed = [(idx, row) for idx, row in enumerate(rows) if not completed(row)]
    if high_risk_first:
        indexed.sort(key=lambda item: (item[1].get("audit_source") != "high_risk", item[0]))
    indices = [idx for idx, _ in indexed]
    return indices[:limit] if limit is not None else indices


def render_row(row: dict[str, str], index: int, total: int) -> str:
    excerpt = str(row.get("response_excerpt", "")).replace("\n", " ").strip()
    return "\n".join(
        [
            f"Row {index + 1}/{total}",
            f"model: {row.get('model', '')}",
            f"item_id: {row.get('item_id', '')}",
            f"category: {row.get('category', '')}",
            f"ground_truth: {row.get('ground_truth', '')}",
            f"parser: answered={row.get('parser_answered', '')} prediction={row.get('parser_prediction', '')} "
            f"method={row.get('extraction_method', '')} confidence={row.get('extraction_confidence', '')}",
            f"audit_source: {row.get('audit_source', '')}",
            f"risk_reason: {row.get('risk_reason', '')}",
            "",
            excerpt,
        ]
    )


def write_progress(output_json: Path, rows: list[dict[str, str]]) -> None:
    write_json(output_json, progress(rows))


def interactive_label(
    *,
    sample_csv: Path,
    progress_json: Path,
    high_risk_first: bool = True,
    limit: int | None = None,
) -> dict[str, Any]:
    rows = read_csv(sample_csv)
    if not rows:
        raise FileNotFoundError(f"No parser-audit rows found at {sample_csv}")
    indices = next_indices(rows, high_risk_first=high_risk_first, limit=limit)
    for index in indices:
        row = rows[index]
        print("\n" + "=" * 80)
        print(render_row(row, index, len(rows)))
        while True:
            answer = input("\nHuman final answer A-J, U=unanswered, S=skip, Q=quit: ").strip().upper()
            if answer == "Q":
                write_progress(progress_json, rows)
                return progress(rows)
            if answer == "S":
                break
            if answer == "U":
                notes = input("Notes, optional: ").strip()
                rows[index] = apply_label(row, human_prediction="", human_answered=False, human_notes=notes)
                write_csv_atomic(sample_csv, rows)
                write_progress(progress_json, rows)
                break
            if answer in VALID_OPTIONS:
                notes = input("Notes, optional: ").strip()
                rows[index] = apply_label(row, human_prediction=answer, human_answered=True, human_notes=notes)
                write_csv_atomic(sample_csv, rows)
                write_progress(progress_json, rows)
                break
            print("Invalid input. Use A-J, U, S, or Q.")
    write_progress(progress_json, rows)
    return progress(rows)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive, resumable first-pass human labeling for parser-audit rows."
    )
    parser.add_argument("--sample-csv", type=Path, default=Path("main/analysis/parser_audit/parser_audit_sample.csv"))
    parser.add_argument(
        "--progress-json",
        type=Path,
        default=Path("main/analysis/parser_audit/parser_audit_interactive_progress.json"),
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum unlabeled rows to show in this session.")
    parser.add_argument("--original-order", action="store_true", help="Do not prioritize high-risk rows first.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    result = interactive_label(
        sample_csv=args.sample_csv,
        progress_json=args.progress_json,
        high_risk_first=not args.original_order,
        limit=args.limit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print("\nAfter labeling all rows, run:")
    print("PYTHONPATH=main/src python3 -m boundary_slm.parser_audit_impact")
    print("PYTHONPATH=main/src python3 -m boundary_slm.mmlu_scoring_robustness")


if __name__ == "__main__":
    main()
