from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from boundary_slm.io import write_json


FORBIDDEN_PATH_PARTS = {
    "results",
    "data/model_outputs",
    "labeling_batches",
}
FORBIDDEN_FILENAMES = {
    "records.jsonl",
    "parser_audit_sample.csv",
    "second_pass_parser_audit_sample.csv",
}
FORBIDDEN_DATA_COLUMNS = {
    "response",
    "response_text",
    "response_tail",
}
REDACTABLE_EXCERPT_COLUMNS = {
    "response_excerpt",
}


def _as_posix(path: Path) -> str:
    return path.as_posix()


def scan_release(root: Path) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    if not root.exists():
        findings.append({"severity": "error", "path": str(root), "issue": "release root does not exist"})
        return {"root": str(root), "passed": False, "findings": findings}

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = _as_posix(path.relative_to(root))
        parts = set(Path(rel).parts)
        if path.suffix == ".jsonl":
            findings.append({"severity": "error", "path": rel, "issue": "JSONL files may contain raw model responses"})
        if path.name in FORBIDDEN_FILENAMES:
            findings.append({"severity": "error", "path": rel, "issue": "raw experiment record file is not public-safe"})
        if any(part in parts for part in FORBIDDEN_PATH_PARTS):
            findings.append({"severity": "error", "path": rel, "issue": "forbidden raw-output path component"})
        if path.suffix.lower() == ".csv":
            try:
                with path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.reader(handle)
                    header = next(reader, [])
            except UnicodeDecodeError:
                header = []
            forbidden = sorted(set(header) & FORBIDDEN_DATA_COLUMNS)
            if forbidden:
                findings.append(
                    {
                        "severity": "error",
                        "path": rel,
                        "issue": f"forbidden public data columns: {', '.join(forbidden)}",
                    }
                )
            redactable = sorted(set(header) & REDACTABLE_EXCERPT_COLUMNS)
            if redactable:
                with path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    for idx, row in enumerate(reader, start=2):
                        for column in redactable:
                            excerpt = str(row.get(column, ""))
                            if excerpt and excerpt != "[withheld_public_release]":
                                findings.append(
                                    {
                                        "severity": "error",
                                        "path": rel,
                                        "issue": f"unredacted response excerpt column `{column}` at CSV line {idx}",
                                    }
                                )
                                break
                        if findings and findings[-1]["path"] == rel and "unredacted response excerpt" in findings[-1]["issue"]:
                            break
        if path.suffix.lower() == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if _json_contains_key(payload, FORBIDDEN_DATA_COLUMNS | REDACTABLE_EXCERPT_COLUMNS):
                findings.append(
                    {
                        "severity": "error",
                        "path": rel,
                        "issue": "forbidden raw-response key appears in JSON data",
                    }
                )

    return {"root": str(root), "passed": not findings, "findings": findings}


def _json_contains_key(payload: Any, keys: set[str]) -> bool:
    if isinstance(payload, dict):
        return any(key in keys for key in payload) or any(_json_contains_key(value, keys) for value in payload.values())
    if isinstance(payload, list):
        return any(_json_contains_key(value, keys) for value in payload)
    return False


def generate(root: Path, output_json: Path | None = None) -> dict[str, Any]:
    report = scan_release(root)
    if output_json is not None:
        write_json(output_json, report)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan a public release folder for raw-response leakage hazards.")
    parser.add_argument("--root", type=Path, default=Path("public_release/error_persistence_regression_burden"))
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    report = generate(args.root, args.output_json)
    print(json.dumps({"passed": report["passed"], "finding_count": len(report["findings"])}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
