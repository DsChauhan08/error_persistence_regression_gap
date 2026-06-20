from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from boundary_slm.io import write_csv, write_json


TARGET_REPOS = [
    {
        "model": "Qwen/Qwen2-0.5B-Instruct",
        "repo_id": "open-llm-leaderboard/Qwen__Qwen2-0.5B-Instruct-details",
    },
    {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "repo_id": "open-llm-leaderboard/Qwen__Qwen2.5-0.5B-Instruct-details",
    },
    {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "repo_id": "open-llm-leaderboard/Qwen__Qwen2.5-1.5B-Instruct-details",
    },
    {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "repo_id": "open-llm-leaderboard/Qwen__Qwen2.5-3B-Instruct-details",
    },
    {
        "model": "google/gemma-2-2b-it",
        "repo_id": "open-llm-leaderboard/google__gemma-2-2b-it-details",
    },
    {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "repo_id": "open-llm-leaderboard/meta-llama__Llama-3.2-1B-Instruct-details",
    },
    {
        "model": "microsoft/phi-2",
        "repo_id": "open-llm-leaderboard/microsoft__phi-2-details",
    },
]


def has_hf_token() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def numeric_score(row: dict[str, Any]) -> float | None:
    candidates = [
        "acc",
        "acc_norm",
        "exact_match",
        "prompt_level_strict_acc",
        "prompt_level_loose_acc",
        "inst_level_strict_acc",
        "score",
    ]
    for key in candidates:
        value = row.get(key)
        if isinstance(value, (int, float)) and value in {0, 1, 0.0, 1.0}:
            return float(value)
    for value in row.values():
        if isinstance(value, dict):
            nested = numeric_score(value)
            if nested is not None:
                return nested
    return None


def item_identifier(row: dict[str, Any], fallback_blob: str) -> str:
    for key in ["doc_id", "id", "idx", "sample_id", "question_id"]:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return sha256_text(fallback_blob)[:16]


def normalize_sample_file(path: Path, *, model: str, repo_id: str, source_file: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        raw_rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        loaded = json.loads(text)
        raw_rows = loaded if isinstance(loaded, list) else loaded.get("samples", loaded.get("rows", []))
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        blob = json.dumps(raw, sort_keys=True, ensure_ascii=True)
        score = numeric_score(raw)
        if score is None:
            continue
        rows.append(
            {
                "source_id": "open_llm_leaderboard_details",
                "repo_id": repo_id,
                "source_file": source_file,
                "model": model,
                "task": "mmlu_pro" if "mmlu_pro" in source_file.lower() else "unknown",
                "item_id": item_identifier(raw, blob),
                "score": int(score),
                "raw_record_sha256": sha256_text(blob),
                "claim_boundary": "Normalized from gated Open LLM Leaderboard detail files; raw prompt/output fields are not emitted.",
            }
        )
    return rows


def generate(output_dir: Path, *, try_download: bool = False, targets: list[dict[str, str]] | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_rows = targets or TARGET_REPOS
    exclusions: list[dict[str, Any]] = []
    normalized: list[dict[str, Any]] = []
    token_available = has_hf_token()

    if try_download and token_available:
        from huggingface_hub import hf_hub_download, list_repo_files

    for target in target_rows:
        repo_id = target["repo_id"]
        model = target["model"]
        if not try_download:
            exclusions.append(
                {
                    "model": model,
                    "repo_id": repo_id,
                    "status": "not_downloaded",
                    "reason": "optional gated source; rerun with --try-download after accepting Hugging Face dataset terms",
                }
            )
            continue
        if not token_available:
            exclusions.append(
                {
                    "model": model,
                    "repo_id": repo_id,
                    "status": "excluded_no_hf_token",
                    "reason": "HF_TOKEN or HUGGINGFACE_HUB_TOKEN is required for gated detail datasets",
                }
            )
            continue
        try:
            files = list_repo_files(repo_id, repo_type="dataset")
            sample_files = [
                file_name
                for file_name in files
                if "samples_leaderboard_mmlu_pro" in file_name.lower()
                and (file_name.endswith(".json") or file_name.endswith(".jsonl"))
            ]
            if not sample_files:
                exclusions.append(
                    {
                        "model": model,
                        "repo_id": repo_id,
                        "status": "excluded_no_mmlu_pro_sample",
                        "reason": "no MMLU-Pro sample file found in detail repo",
                    }
                )
                continue
            source_file = sorted(sample_files)[-1]
            path = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=source_file))
            normalized.extend(normalize_sample_file(path, model=model, repo_id=repo_id, source_file=source_file))
        except Exception as exc:
            exclusions.append(
                {
                    "model": model,
                    "repo_id": repo_id,
                    "status": "excluded_download_failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_id": "open_llm_leaderboard_details",
        "try_download": try_download,
        "hf_token_available": token_available,
        "target_count": len(target_rows),
        "normalized_record_count": len(normalized),
        "exclusion_count": len(exclusions),
        "claim_boundary": "Gated detail files are optional and never silently skipped; exclusions are recorded.",
    }
    write_json(output_dir / "open_llm_details_manifest.json", report)
    write_json(output_dir / "open_llm_details_exclusions.json", exclusions)
    write_csv(output_dir / "open_llm_details_exclusions.csv", exclusions)
    if normalized:
        write_json(output_dir / "open_llm_details_normalized.json", normalized)
        write_csv(output_dir / "open_llm_details_normalized.csv", normalized)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optionally ingest gated Open LLM Leaderboard per-model detail files.")
    parser.add_argument("--output-dir", type=Path, default=Path("main/analysis/external_evidence/open_llm_details"))
    parser.add_argument("--try-download", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    report = generate(args.output_dir, try_download=args.try_download)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
