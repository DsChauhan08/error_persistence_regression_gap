from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

from boundary_slm.io import append_jsonl, read_jsonl, write_csv, write_json
from boundary_slm.mmlu_pro_manifest import (
    SOURCE_FILE,
    SOURCE_REPO,
    SourceRow,
    download_mmlu_pro_parquet,
    read_source_rows_from_parquet,
    source_row_hash,
)
from boundary_slm.raw_results import extract_answer, infer_model_meta, safe_ratio


DEFAULT_MODELS = [
    "Qwen/Qwen2-0.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
]


PROMPT_TEMPLATE = """Question: {question}

Options:
{options}

Answer with only the final option letter A-J."""


class TextGenerator(Protocol):
    def metadata(self) -> dict[str, Any]:
        ...

    def generate(self, prompt: str) -> str:
        ...

    def close(self) -> None:
        ...


class MockGenerator:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def metadata(self) -> dict[str, Any]:
        return {
            "generator_backend": "mock",
            "model_class": "MockGenerator",
            "tokenizer_class": "mock",
            "chat_template_sha256": "mock",
            "chat_template_available": False,
        }

    def generate(self, prompt: str) -> str:
        digest = sum(ord(ch) for ch in (self.model_id + prompt))
        return f"Final answer: {'ABCDEFGHIJ'[digest % 10]}"

    def close(self) -> None:
        return None


class TransformersCpuGenerator:
    def __init__(self, model_id: str, *, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def metadata(self) -> dict[str, Any]:
        chat_template = getattr(self.tokenizer, "chat_template", None) or ""
        generation_config = getattr(self.model, "generation_config", None)
        return {
            "generator_backend": "transformers",
            "model_class": type(self.model).__name__,
            "tokenizer_class": type(self.tokenizer).__name__,
            "tokenizer_name_or_path": getattr(self.tokenizer, "name_or_path", ""),
            "chat_template_available": bool(chat_template),
            "chat_template_sha256": hashlib.sha256(str(chat_template).encode("utf-8")).hexdigest() if chat_template else "",
            "generation_config": generation_config.to_dict() if hasattr(generation_config, "to_dict") else {},
        }

    def _format_prompt(self, prompt: str) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                return prompt
        return prompt

    def generate(self, prompt: str) -> str:
        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt")
        input_len = int(inputs["input_ids"].shape[-1])
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        generated = output[0][input_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def close(self) -> None:
        del self.model
        del self.tokenizer


def environment_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count": None,
        "ram_gb": None,
    }
    try:
        import os

        record["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    try:
        import psutil

        record["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 3)
    except Exception:
        pass
    for name in ["torch", "transformers", "huggingface_hub"]:
        try:
            mod = __import__(name)
            record[f"{name}_version"] = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            record[f"{name}_version"] = f"unavailable:{type(exc).__name__}"
    return record


def model_revision(model_id: str) -> str:
    try:
        from huggingface_hub import model_info

        return str(model_info(model_id).sha or "unknown")
    except Exception:
        return "unknown"


def format_options(options: list[str]) -> str:
    letters = "ABCDEFGHIJ"
    return "\n".join(f"{letters[idx]}. {option}" for idx, option in enumerate(options[:10]))


def prompt_for(row: SourceRow) -> str:
    return PROMPT_TEMPLATE.format(question=row.question, options=format_options(row.options))


def selected_item_manifest(source_rows: dict[str, SourceRow], sample_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item_id in enumerate(sample_ids, start=1):
        row = source_rows[item_id]
        hashes = source_row_hash(row)
        rows.append(
            {
                "sample_index": index,
                "item_id": item_id,
                "category": row.category,
                "src": row.src,
                "split": row.split,
                "question_sha256": hashes["question_sha256"],
                "options_sha256": hashes["options_sha256"],
                "answer_sha256": hashes["answer_sha256"],
                "item_sha256": hashes["item_sha256"],
            }
        )
    return rows


def read_item_csvs(per_model_dir: Path) -> dict[str, list[dict[str, str]]]:
    by_item: dict[str, list[dict[str, str]]] = defaultdict(list)
    for path in sorted(per_model_dir.glob("*_items.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                by_item[str(row.get("id", ""))].append(row)
    return by_item


def select_sample_ids(
    *,
    source_rows: dict[str, SourceRow],
    per_model_dir: Path,
    sample_size: int,
    stress_size: int,
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    current = read_item_csvs(per_model_dir)
    stress_scores: list[tuple[int, str]] = []
    for item_id, rows in current.items():
        correctness = {str(row.get("is_correct", "")).lower() for row in rows}
        methods = {str(row.get("extraction_method", "")) for row in rows}
        score = len(correctness) + sum(1 for method in methods if method in {"tail_option", "last_standalone_letter", "none"})
        if item_id in source_rows:
            stress_scores.append((score, item_id))
    stress_ids = [
        item_id
        for _score, item_id in sorted(stress_scores, key=lambda pair: (-pair[0], int(pair[1]) if pair[1].isdigit() else pair[1]))[:stress_size]
    ]
    remaining_slots = max(sample_size - len(stress_ids), 0)
    by_category: dict[str, list[str]] = defaultdict(list)
    stress_set = set(stress_ids)
    for item_id, row in source_rows.items():
        if item_id in current and item_id not in stress_set:
            by_category[row.category].append(item_id)
    random_ids: list[str] = []
    categories = sorted(by_category)
    while len(random_ids) < remaining_slots and categories:
        progressed = False
        for category in categories:
            values = by_category[category]
            if not values:
                continue
            idx = rng.randrange(len(values))
            random_ids.append(values.pop(idx))
            progressed = True
            if len(random_ids) >= remaining_slots:
                break
        if not progressed:
            break
    return stress_ids + random_ids


def load_source_rows(parquet_path: Path | None, revision: str | None) -> tuple[dict[str, SourceRow], dict[str, Any]]:
    source_revision = revision or "unknown"
    if parquet_path is None:
        parquet_path, source_revision = download_mmlu_pro_parquet(SOURCE_REPO, SOURCE_FILE, revision)
    rows, parquet_sha = read_source_rows_from_parquet(parquet_path)
    return rows, {
        "source_repo": SOURCE_REPO,
        "source_file": SOURCE_FILE,
        "source_revision": source_revision,
        "source_parquet_path": str(parquet_path),
        "source_parquet_sha256": parquet_sha,
    }


def completed_pairs(records_path: Path) -> set[tuple[str, str]]:
    return {
        (str(row.get("model_id")), str(row.get("item_id")))
        for row in read_jsonl(records_path)
        if row.get("model_id") and row.get("item_id")
    }


def generator_for(model_id: str, *, backend: str, max_new_tokens: int) -> TextGenerator:
    if backend == "mock":
        return MockGenerator(model_id)
    if backend == "transformers":
        return TransformersCpuGenerator(model_id, max_new_tokens=max_new_tokens)
    raise ValueError(f"Unknown backend: {backend}")


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_model[str(row["model_id"])].append(row)
    out: list[dict[str, Any]] = []
    for model_id, rows in sorted(by_model.items()):
        total = len(rows)
        correct = sum(1 for row in rows if row.get("is_correct"))
        answered = sum(1 for row in rows if row.get("answered"))
        elapsed = sum(float(row.get("elapsed_seconds", 0.0)) for row in rows)
        out.append(
            {
                "model_id": model_id,
                "n": total,
                "correct": correct,
                "accuracy": round(safe_ratio(correct, total), 6),
                "answered_rate": round(safe_ratio(answered, total), 6),
                "elapsed_seconds": round(elapsed, 3),
                "mean_seconds_per_item": round(safe_ratio(elapsed, total), 6),
            }
        )
    return out


def pairwise_audit(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model_item = {(str(row["model_id"]), str(row["item_id"])): bool(row.get("is_correct")) for row in records}
    models = sorted({str(row["model_id"]) for row in records})
    item_ids = sorted({str(row["item_id"]) for row in records}, key=lambda value: int(value) if value.isdigit() else value)
    out: list[dict[str, Any]] = []
    for old_idx, old_model in enumerate(models):
        for new_model in models[old_idx + 1 :]:
            common = [item_id for item_id in item_ids if (old_model, item_id) in by_model_item and (new_model, item_id) in by_model_item]
            if not common:
                continue
            old_correct = sum(1 for item_id in common if by_model_item[(old_model, item_id)])
            new_correct = sum(1 for item_id in common if by_model_item[(new_model, item_id)])
            improvements = sum(
                1
                for item_id in common
                if not by_model_item[(old_model, item_id)] and by_model_item[(new_model, item_id)]
            )
            regressions = sum(
                1
                for item_id in common
                if by_model_item[(old_model, item_id)] and not by_model_item[(new_model, item_id)]
            )
            out.append(
                {
                    "old_model": old_model,
                    "new_model": new_model,
                    "n_common": len(common),
                    "old_accuracy": round(safe_ratio(old_correct, len(common)), 6),
                    "new_accuracy": round(safe_ratio(new_correct, len(common)), 6),
                    "accuracy_delta": round(safe_ratio(new_correct - old_correct, len(common)), 6),
                    "improvement_mass": round(safe_ratio(improvements, len(common)), 6),
                    "regression_mass": round(safe_ratio(regressions, len(common)), 6),
                    "churn_mass": round(safe_ratio(improvements + regressions, len(common)), 6),
                }
            )
    return out


def write_claim_check(output_dir: Path, *, records: list[dict[str, Any]], expected_models: list[str], expected_items: int) -> dict[str, Any]:
    summaries = summarize_records(records)
    pairwise = pairwise_audit(records)
    completed_model_count = sum(1 for row in summaries if int(row["n"]) == expected_items)
    claim = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "expected_models": expected_models,
        "expected_items_per_model": expected_items,
        "completed_model_count": completed_model_count,
        "record_count": len(records),
        "pairwise_comparison_count": len(pairwise),
        "claim_ready": completed_model_count == len(expected_models) and len(pairwise) > 0,
        "claim_scope": "CPU rerun is a reproducibility validation only, not a replacement for the full TPU run.",
    }
    write_csv(output_dir / "summary.csv", summaries)
    write_csv(output_dir / "pairwise_cpu_audit.csv", pairwise)
    write_json(output_dir / "claim_check.json", claim)
    return claim


def run(
    *,
    output_dir: Path,
    per_model_dir: Path,
    models: list[str],
    backend: str,
    sample_size: int,
    stress_size: int,
    seed: int,
    max_new_tokens: int,
    parquet_path: Path | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    source_rows, source_meta = load_source_rows(parquet_path, revision)
    sample_ids = select_sample_ids(
        source_rows=source_rows,
        per_model_dir=per_model_dir,
        sample_size=sample_size,
        stress_size=stress_size,
        seed=seed,
    )
    write_json(output_dir / "environment.json", environment_record())
    write_csv(output_dir / "selected_item_manifest.csv", selected_item_manifest(source_rows, sample_ids))
    model_revisions = {model: ("mock" if backend == "mock" else model_revision(model)) for model in models}
    run_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "models": [
            {
                "model_id": model,
                "revision": model_revisions[model],
                "family": infer_model_meta(model).family,
                "loader_metadata": {},
            }
            for model in models
        ],
        "sample_size": len(sample_ids),
        "stress_size": min(stress_size, len(sample_ids)),
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "decoding": {"do_sample": False, "temperature": None, "top_p": None},
        "prompt_template": PROMPT_TEMPLATE,
        "prompt_template_sha256": hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest(),
        "source": source_meta,
        "item_ids": sample_ids,
        "selected_item_manifest": "selected_item_manifest.csv",
        "resume_policy": "Existing records.jsonl rows are skipped by (model_id, item_id).",
        "completed_models": [],
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    done = completed_pairs(records_path)
    for model_id in models:
        generator = generator_for(model_id, backend=backend, max_new_tokens=max_new_tokens)
        for model_entry in run_manifest["models"]:
            if model_entry["model_id"] == model_id:
                model_entry["loader_metadata"] = generator.metadata()
                break
        write_json(output_dir / "run_manifest.json", run_manifest)
        try:
            for item_id in sample_ids:
                if (model_id, item_id) in done:
                    continue
                item = source_rows[item_id]
                prompt = prompt_for(item)
                started = time.perf_counter()
                response = generator.generate(prompt)
                elapsed = time.perf_counter() - started
                prediction, method, confidence = extract_answer(response)
                record = {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "model_id": model_id,
                    "model_revision": model_revisions[model_id],
                    "item_id": item_id,
                    "category": item.category,
                    "ground_truth": item.answer,
                    "prediction": prediction,
                    "answered": prediction is not None,
                    "is_correct": prediction == item.answer,
                    "extraction_method": method,
                    "extraction_confidence": confidence,
                    "elapsed_seconds": round(elapsed, 6),
                    "response": response,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                }
                append_jsonl(records_path, [record])
                done.add((model_id, item_id))
        finally:
            generator.close()
        if model_id not in run_manifest["completed_models"]:
            run_manifest["completed_models"].append(model_id)
        write_json(output_dir / "run_manifest.json", run_manifest)
        write_claim_check(output_dir, records=read_jsonl(records_path), expected_models=models, expected_items=len(sample_ids))
    return write_claim_check(output_dir, records=read_jsonl(records_path), expected_models=models, expected_items=len(sample_ids))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a resumable CPU-only MMLU-Pro validation rerun.")
    parser.add_argument("--output-dir", type=Path, default=Path("main/outputs/cpu_mmlu_pro_rerun"))
    parser.add_argument("--per-model-dir", type=Path, default=Path("main/analysis/raw_tpu_results/per_model"))
    parser.add_argument("--model", action="append", dest="models", default=None)
    parser.add_argument("--backend", choices=["transformers", "mock"], default="transformers")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--stress-size", type=int, default=88)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--parquet-path", type=Path, default=None)
    parser.add_argument("--revision", default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    claim = run(
        output_dir=args.output_dir,
        per_model_dir=args.per_model_dir,
        models=args.models or DEFAULT_MODELS,
        backend=args.backend,
        sample_size=args.sample_size,
        stress_size=args.stress_size,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        parquet_path=args.parquet_path,
        revision=args.revision,
    )
    print(json.dumps(claim, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
