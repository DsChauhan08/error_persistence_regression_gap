from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from boundary_slm.io import write_csv, write_json


SOURCE_REPO = "TIGER-Lab/MMLU-Pro"
SOURCE_SPLIT = "test"
SOURCE_FILE = "data/test-00000-of-00001.parquet"


@dataclass(frozen=True)
class SourceRow:
    question_id: str
    question: str
    options: list[str]
    answer: str
    category: str
    src: str
    split: str = SOURCE_SPLIT


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", "\n").split())


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def options_to_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return [str(value)]


def source_row_hash(row: SourceRow) -> dict[str, str]:
    normalized_options = [normalize_text(option) for option in row.options]
    option_blob = json.dumps(normalized_options, ensure_ascii=True, sort_keys=True)
    item_blob = json.dumps(
        {
            "answer": normalize_text(row.answer),
            "category": normalize_text(row.category),
            "options": normalized_options,
            "question": normalize_text(row.question),
            "question_id": str(row.question_id),
            "src": normalize_text(row.src),
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    return {
        "question_sha256": sha256_text(normalize_text(row.question)),
        "options_sha256": sha256_text(option_blob),
        "answer_sha256": sha256_text(normalize_text(row.answer)),
        "item_sha256": sha256_text(item_blob),
    }


def read_current_items(input_dir: Path) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    for path in sorted(input_dir.glob("*_items.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                item_id = str(row.get("id", "")).strip()
                if not item_id:
                    continue
                record = items.setdefault(
                    item_id,
                    {
                        "id": item_id,
                        "categories": Counter(),
                        "ground_truths": Counter(),
                        "models": set(),
                        "row_count": 0,
                    },
                )
                record["categories"][normalize_text(row.get("category", ""))] += 1
                record["ground_truths"][normalize_text(row.get("ground_truth", "")).upper()] += 1
                record["models"].add(str(row.get("model", path.name.removesuffix("_items.csv"))))
                record["row_count"] += 1
    return items


def canonical_counter_value(counter: Counter[str]) -> str:
    if not counter:
        return ""
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]


def read_source_rows_from_parquet(path: Path, *, split: str = SOURCE_SPLIT) -> tuple[dict[str, SourceRow], str]:
    import pandas as pd

    frame = pd.read_parquet(path)
    rows: dict[str, SourceRow] = {}
    for record in frame.to_dict(orient="records"):
        question_id = str(record.get("question_id", "")).strip()
        if not question_id:
            continue
        rows[question_id] = SourceRow(
            question_id=question_id,
            question=str(record.get("question", "")),
            options=options_to_list(record.get("options", [])),
            answer=str(record.get("answer", "")).strip().upper(),
            category=normalize_text(record.get("category", "")),
            src=normalize_text(record.get("src", "")),
            split=split,
        )
    return rows, sha256_bytes(path.read_bytes())


def download_mmlu_pro_parquet(repo_id: str = SOURCE_REPO, filename: str = SOURCE_FILE, revision: str | None = None) -> tuple[Path, str]:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, revision=revision))
    snapshot = "unknown"
    parts = path.parts
    if "snapshots" in parts:
        idx = parts.index("snapshots")
        if idx + 1 < len(parts):
            snapshot = parts[idx + 1]
    return path, snapshot


def build_manifest_rows(
    current_items: dict[str, dict[str, Any]],
    source_rows: dict[str, SourceRow],
    *,
    source_repo: str,
    source_revision: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item_id in sorted(current_items, key=lambda value: int(value) if value.isdigit() else value):
        current = current_items[item_id]
        current_category = canonical_counter_value(current["categories"])
        current_answer = canonical_counter_value(current["ground_truths"])
        source = source_rows.get(item_id)
        status = "matched"
        notes: list[str] = []
        hashes = {
            "question_sha256": "",
            "options_sha256": "",
            "answer_sha256": "",
            "item_sha256": "",
        }
        source_category = ""
        source_answer = ""
        source_src = ""
        split = SOURCE_SPLIT
        if source is None:
            status = "missing_source_row"
            notes.append("No source row found for evaluated id.")
        else:
            hashes = source_row_hash(source)
            source_category = source.category
            source_answer = source.answer
            source_src = source.src
            split = source.split
            if normalize_text(source_category).lower() != normalize_text(current_category).lower():
                status = "mismatch"
                notes.append("Category differs from source row.")
            if source_answer.upper() != current_answer.upper():
                status = "mismatch"
                notes.append("Ground-truth answer differs from source row.")
        categories_seen = sorted(current["categories"])
        answers_seen = sorted(current["ground_truths"])
        if len(categories_seen) > 1:
            status = "mismatch"
            notes.append("Current outputs disagree on category for this id.")
        if len(answers_seen) > 1:
            status = "mismatch"
            notes.append("Current outputs disagree on ground truth for this id.")
        out.append(
            {
                "id": item_id,
                "source_repo": source_repo,
                "source_revision": source_revision,
                "source_split": split,
                "current_category": current_category,
                "source_category": source_category,
                "current_ground_truth": current_answer,
                "source_answer": source_answer,
                "source_src": source_src,
                "models_seen_count": len(current["models"]),
                "current_rows_seen": current["row_count"],
                "question_sha256": hashes["question_sha256"],
                "options_sha256": hashes["options_sha256"],
                "answer_sha256": hashes["answer_sha256"],
                "item_sha256": hashes["item_sha256"],
                "status": status,
                "notes": " ".join(notes),
            }
        )
    return out


def summarize_manifest(rows: list[dict[str, Any]], *, source_repo: str, source_revision: str, parquet_sha256: str) -> dict[str, Any]:
    counts = Counter(str(row["status"]) for row in rows)
    categories = Counter(str(row["current_category"]) for row in rows)
    source_hashes = [str(row["item_sha256"]) for row in rows if row.get("item_sha256")]
    manifest_hash = sha256_text(json.dumps(source_hashes, sort_keys=True))
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": source_repo,
        "source_revision": source_revision,
        "source_split": SOURCE_SPLIT,
        "source_parquet_sha256": parquet_sha256,
        "evaluated_item_count": len(rows),
        "matched_item_count": counts.get("matched", 0),
        "mismatch_item_count": counts.get("mismatch", 0),
        "missing_source_item_count": counts.get("missing_source_row", 0),
        "status_counts": dict(sorted(counts.items())),
        "category_counts": dict(sorted(categories.items())),
        "manifest_item_hash_sha256": manifest_hash,
        "claim_ready": len(rows) > 0 and counts.get("matched", 0) == len(rows),
        "public_text_policy": "Question text, answer-option text, and raw model responses are not written to this manifest.",
    }


def write_latex_status(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Field & Value \\\\",
        "\\midrule",
        f"Evaluated items & {report['evaluated_item_count']} \\\\",
        f"Matched MMLU-Pro rows & {report['matched_item_count']} \\\\",
        f"Mismatches & {report['mismatch_item_count']} \\\\",
        f"Missing source rows & {report['missing_source_item_count']} \\\\",
        f"Claim-ready manifest & {'yes' if report['claim_ready'] else 'no'} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def generate(
    *,
    input_dir: Path,
    output_dir: Path,
    parquet_path: Path | None = None,
    source_repo: str = SOURCE_REPO,
    revision: str | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_revision = revision or "unknown"
    if parquet_path is None:
        parquet_path, source_revision = download_mmlu_pro_parquet(source_repo, SOURCE_FILE, revision)
    source_rows, parquet_sha256 = read_source_rows_from_parquet(parquet_path)
    current_items = read_current_items(input_dir)
    rows = build_manifest_rows(current_items, source_rows, source_repo=source_repo, source_revision=source_revision)
    report = summarize_manifest(rows, source_repo=source_repo, source_revision=source_revision, parquet_sha256=parquet_sha256)
    write_csv(output_dir / "mmlu_pro_source_manifest.csv", rows)
    write_json(output_dir / "mmlu_pro_source_manifest.json", rows)
    write_json(output_dir / "mmlu_pro_manifest_report.json", report)
    write_latex_status(output_dir / "tables" / "mmlu_pro_manifest_status.tex", report)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct and hash the MMLU-Pro source manifest for evaluated items.")
    parser.add_argument("--input-dir", type=Path, default=Path("main/analysis/raw_tpu_results/per_model"))
    parser.add_argument("--output-dir", type=Path, default=Path("main/analysis/source_manifest"))
    parser.add_argument("--parquet-path", type=Path, default=None)
    parser.add_argument("--source-repo", default=SOURCE_REPO)
    parser.add_argument("--revision", default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    report = generate(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        parquet_path=args.parquet_path,
        source_repo=args.source_repo,
        revision=args.revision,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
