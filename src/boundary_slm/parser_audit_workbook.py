from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from boundary_slm.io import write_csv, write_json
from boundary_slm.parser_audit_report import truthy


HUMAN_FIELDS = ["human_prediction", "human_answered", "human_parser_correct", "human_notes"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def completed_first_pass(row: dict[str, str]) -> bool:
    return truthy(row.get("human_answered", "")) is not None and truthy(row.get("human_parser_correct", "")) is not None


def redacted_progress_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"rows": 0, "completed_rows": 0, "high_risk_rows": 0, "completed_high_risk_rows": 0}
    )
    for row in rows:
        key = (
            row.get("family", "unknown") or "unknown",
            row.get("extraction_method", "unknown") or "unknown",
            row.get("audit_source", "unknown") or "unknown",
        )
        group = groups[key]
        complete = completed_first_pass(row)
        high_risk = row.get("audit_source") == "high_risk"
        group["rows"] += 1
        group["completed_rows"] += int(complete)
        group["high_risk_rows"] += int(high_risk)
        group["completed_high_risk_rows"] += int(complete and high_risk)

    out: list[dict[str, Any]] = []
    for (family, method, source), group in sorted(groups.items()):
        rows_n = int(group["rows"])
        out.append(
            {
                "family": family,
                "extraction_method": method,
                "audit_source": source,
                "rows": rows_n,
                "completed_rows": group["completed_rows"],
                "completion_rate": round(group["completed_rows"] / rows_n, 6) if rows_n else None,
                "high_risk_rows": group["high_risk_rows"],
                "completed_high_risk_rows": group["completed_high_risk_rows"],
            }
        )
    return out


def overall_progress(rows: list[dict[str, str]]) -> dict[str, Any]:
    total = len(rows)
    completed = sum(1 for row in rows if completed_first_pass(row))
    high_risk = sum(1 for row in rows if row.get("audit_source") == "high_risk")
    completed_high_risk = sum(1 for row in rows if row.get("audit_source") == "high_risk" and completed_first_pass(row))
    return {
        "total_rows": total,
        "completed_rows": completed,
        "completion_rate": round(completed / total, 6) if total else None,
        "high_risk_rows": high_risk,
        "completed_high_risk_rows": completed_high_risk,
        "high_risk_completion_rate": round(completed_high_risk / high_risk, 6) if high_risk else None,
        "remaining_rows": total - completed,
        "remaining_high_risk_rows": high_risk - completed_high_risk,
    }


def prepare_batch_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        prepared = dict(row)
        for field in HUMAN_FIELDS:
            prepared[field] = str(prepared.get(field, ""))
        out.append(prepared)
    return out


def chunked(rows: list[dict[str, str]], size: int) -> Iterable[list[dict[str, str]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def generate(
    *,
    sample_csv: Path,
    output_dir: Path,
    batch_size: int = 50,
    write_batches: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(sample_csv)
    progress = overall_progress(rows)
    progress_rows = redacted_progress_rows(rows)

    batch_manifest: list[dict[str, Any]] = []
    if write_batches and rows:
        batch_root = output_dir / "labeling_batches"
        batch_root.mkdir(parents=True, exist_ok=True)
        for index, batch in enumerate(chunked(prepare_batch_rows(rows), batch_size), start=1):
            path = batch_root / f"parser_audit_batch_{index:03d}.csv"
            write_csv(path, batch)
            batch_manifest.append(
                {
                    "batch_id": f"{index:03d}",
                    "path": str(path),
                    "rows": len(batch),
                    "completed_rows": sum(1 for row in batch if completed_first_pass(row)),
                    "high_risk_rows": sum(1 for row in batch if row.get("audit_source") == "high_risk"),
                    "private_response_excerpt_file": True,
                }
            )

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_csv": str(sample_csv),
        "batch_size": batch_size,
        "write_batches": write_batches,
        "progress": progress,
        "progress_by_stratum": progress_rows,
        "batch_manifest": batch_manifest,
        "public_text_policy": (
            "Batch CSV files are private because they may include response excerpts. Public reports contain only "
            "counts and completion rates."
        ),
    }
    public_report = {
        key: value
        for key, value in report.items()
        if key not in {"batch_manifest"}
    }
    public_report["batch_count"] = len(batch_manifest)
    public_report["batch_manifest_public_note"] = "Private batch file paths are omitted from the public progress report."
    write_json(output_dir / "parser_audit_labeling_progress.json", public_report)
    write_csv(output_dir / "parser_audit_labeling_progress.csv", progress_rows)
    if batch_manifest:
        write_csv(output_dir / "parser_audit_labeling_batches.csv", batch_manifest)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create private parser-audit labeling batches and progress reports.")
    parser.add_argument("--sample-csv", type=Path, default=Path("main/analysis/parser_audit/parser_audit_sample.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("main/analysis/parser_audit"))
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--no-batches", action="store_true", help="Only write redacted progress reports.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    report = generate(
        sample_csv=args.sample_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        write_batches=not args.no_batches,
    )
    print(json.dumps(report["progress"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
