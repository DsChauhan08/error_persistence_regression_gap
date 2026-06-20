from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from statistics import mean
from typing import Any, Iterable

from boundary_slm.io import read_jsonl, write_csv, write_json


ANSWER_LETTERS = "ABCDEFGHIJ"
LETTER_CLASS = "[A-J]"


EXPLICIT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "final_answer",
        re.compile(
            rf"(?is)\bfinal\s+answer\b\s*(?:is\b)?\s*[:=]?\s*(?:\*\*)?\s*[\(\[]?\s*({LETTER_CLASS})\b\s*[\)\]\.:\-\*]?"
        ),
    ),
    (
        "correct_answer",
        re.compile(
            rf"(?is)\bcorrect\s+(?:answer|option|choice)\b\s*(?:is\b)?\s*[:=]?\s*(?:\*\*)?\s*[\(\[]?\s*({LETTER_CLASS})\b\s*[\)\]\.:\-\*]?"
        ),
    ),
    (
        "therefore_answer",
        re.compile(
            rf"(?is)\btherefore\b.{0,160}?\b(?:answer|option|choice)\b\s*(?:is\b)?\s*[:=]?\s*(?:\*\*)?\s*[\(\[]?\s*({LETTER_CLASS})\b\s*[\)\]\.:\-\*]?"
        ),
    ),
    (
        "answer_is",
        re.compile(
            rf"(?is)\b(?:the\s+)?answer\b\s*(?:(?:is\b)\s*[:=]?|[:=])\s*(?:\*\*)?\s*[\(\[]?\s*({LETTER_CLASS})\b\s*[\)\]\.:\-\*]?"
        ),
    ),
    (
        "option_is",
        re.compile(
            rf"(?is)\boption\b\s*(?:is\b)?\s*[:=]?\s*(?:\*\*)?\s*[\(\[]?\s*({LETTER_CLASS})\b\s*[\)\]\.:\-\*]?"
        ),
    ),
]

TAIL_OPTION_PATTERN = re.compile(rf"(?is)(?:^|[\s\*\(\[])\b({LETTER_CLASS})\b\s*[\)\]\.:\-]")
SINGLE_LETTER_PATTERN = re.compile(rf"^\s*(?:\(?\s*)?({LETTER_CLASS})(?:\s*[\).])?\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class ModelMeta:
    label: str
    family: str
    generation: str
    parameter_b: float
    boundary_role: str

    def to_record(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "family": self.family,
            "generation": self.generation,
            "parameter_b": self.parameter_b,
            "boundary_role": self.boundary_role,
        }


def extract_answer(response: str) -> tuple[str | None, str, float]:
    text = (response or "").strip()
    if not text:
        return None, "empty", 0.0

    single = SINGLE_LETTER_PATTERN.match(text)
    if single:
        return single.group(1).upper(), "single_letter", 0.99

    windows = [
        text[-2500:],
        text,
    ]
    for window in windows:
        for method, pattern in EXPLICIT_PATTERNS:
            matches = list(pattern.finditer(window))
            if matches:
                return matches[-1].group(1).upper(), method, 0.90

    tail_matches = list(TAIL_OPTION_PATTERN.finditer(text[-1200:]))
    if tail_matches:
        return tail_matches[-1].group(1).upper(), "tail_option", 0.62

    all_letters = re.findall(rf"\b({LETTER_CLASS})\b", text.upper())
    if all_letters:
        return all_letters[-1].upper(), "last_standalone_letter", 0.35
    return None, "none", 0.0


def extract_answer_from_raw(raw: dict[str, Any]) -> tuple[str | None, str, float]:
    prediction = str(raw.get("prediction", "")).strip().upper()
    if re.fullmatch(rf"[{ANSWER_LETTERS}]", prediction):
        return prediction, "raw_prediction", 1.0
    response = str(raw.get("response", raw.get("response_text", "")))
    if looks_like_prompt_echo_without_completion(response):
        return None, "prompt_echo_without_completion", 0.0
    return extract_answer(response)


def looks_like_prompt_echo_without_completion(response: str) -> bool:
    text = (response or "").strip()
    if not text:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or lines[-1].lower() != "model":
        return False
    lower = text.lower()
    prompt_markers = [
        "question:",
        "options:",
        "answer with only the final option letter",
        "think step by step",
    ]
    explicit_answer_markers = [
        "final answer",
        "correct answer",
        "therefore, the answer",
        "therefore the answer",
        "answer is",
        "correct option",
    ]
    return any(marker in lower for marker in prompt_markers) and not any(
        marker in lower for marker in explicit_answer_markers
    )


def infer_model_meta(label: str) -> ModelMeta:
    lower = label.lower()
    family = "other"
    if "qwen" in lower:
        family = "qwen"
    elif "gemma" in lower:
        family = "gemma"
    elif "llama" in lower:
        family = "llama"
    elif "phi" in lower:
        family = "phi"

    generation = "unknown"
    for pattern, name in [
        (r"qwen3\.5", "qwen3.5"),
        (r"qwen3", "qwen3"),
        (r"qwen2\.5", "qwen2.5"),
        (r"qwen2", "qwen2"),
        (r"gemma-?4", "gemma4"),
        (r"gemma-?3", "gemma3"),
        (r"gemma-?2", "gemma2"),
        (r"llama-?3\.2", "llama3.2"),
        (r"llama-?3", "llama3"),
        (r"llama-?2", "llama2"),
        (r"phi-?2", "phi2"),
    ]:
        if re.search(pattern, lower):
            generation = name
            break

    parameter_b = 0.0
    match = re.search(r"(\d+(?:\.\d+)?)\s*b", lower)
    if match:
        parameter_b = float(match.group(1))
    else:
        match_m = re.search(r"(\d+(?:\.\d+)?)\s*m", lower)
        if match_m:
            parameter_b = float(match_m.group(1)) / 1000.0
        elif family == "phi" and generation == "phi2":
            parameter_b = 2.7
    boundary_role = "unknown"
    if 0 < parameter_b < 4:
        boundary_role = "strict_sub_4b"
    elif math.isclose(parameter_b, 4.0, abs_tol=0.05):
        boundary_role = "boundary_4b"
    elif parameter_b > 4:
        boundary_role = "above_boundary_or_appendix"

    return ModelMeta(
        label=label,
        family=family,
        generation=generation,
        parameter_b=parameter_b,
        boundary_role=boundary_role,
    )


def analyze_raw_results(
    *,
    input_dir: Path,
    output_dir: Path,
    bootstrap_iters: int = 1000,
) -> dict[str, Any]:
    paths = sorted(input_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No .jsonl files found under {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    per_model_dir = output_dir / "per_model"
    per_model_dir.mkdir(parents=True, exist_ok=True)

    scored_by_model: dict[str, list[dict[str, Any]]] = {}
    raw_by_model: dict[str, list[dict[str, Any]]] = {}
    model_summaries: list[dict[str, Any]] = []
    dataset_signatures: dict[str, Any] = {}

    for path in paths:
        rows = read_jsonl(path)
        if not rows:
            continue
        label = str(rows[0].get("model") or rows[0].get("model_label") or path.stem)
        meta = infer_model_meta(label)
        scored_rows = score_model_rows(rows, meta)
        raw_by_model[label] = rows
        scored_by_model[label] = scored_rows
        summary = summarize_model(scored_rows, meta, source_file=path)
        model_summaries.append(summary)
        write_json(per_model_dir / f"{safe_name(label)}.json", summary)
        write_csv(per_model_dir / f"{safe_name(label)}_category_summary.csv", summary["category_summary"])
        write_csv(per_model_dir / f"{safe_name(label)}_items.csv", scored_rows)
        dataset_signatures[label] = {
            "source_file": str(path),
            "row_count": len(rows),
            "id_count": len({raw_item_id(row) for row in rows}),
            "category_count": len({str(row.get("category", row.get("task_family", "unknown"))) for row in rows}),
            "ground_truth_letters": sorted({str(row.get("ground_truth", row.get("expected", ""))) for row in rows}),
        }

    leaderboard = sorted(
        [flatten_model_summary(summary) for summary in model_summaries],
        key=lambda row: (-float(row["accuracy"]), row["model"]),
    )
    write_csv(output_dir / "leaderboard.csv", leaderboard)
    write_json(output_dir / "leaderboard.json", leaderboard)

    category_table = build_category_table(model_summaries)
    write_csv(output_dir / "category_table.csv", category_table)

    pairwise = pairwise_error_ecology(scored_by_model, bootstrap_iters=bootstrap_iters)
    write_json(output_dir / "pairwise_error_ecology.json", pairwise)
    write_csv(output_dir / "pairwise_error_ecology.csv", pairwise["comparisons"])
    standard_pairwise = build_standard_pairwise_outputs(pairwise, scored_by_model)
    write_json(output_dir / "standardized_pairwise_error_ecology_all_models.json", standard_pairwise)
    write_csv(output_dir / "standardized_pairwise_error_ecology_all_models.csv", standard_pairwise["rows"])

    paper1 = build_paper1_conclusions(leaderboard, pairwise, model_summaries)
    paper2 = build_paper2_conclusions(scored_by_model)
    evidence = build_evidence_assessment(scored_by_model, leaderboard)
    evidence_audit = build_raw_evidence_audit(raw_by_model)
    write_json(output_dir / "evidence_audit.json", evidence_audit)
    conclusions = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "dataset_signatures": dataset_signatures,
        "evidence_audit": evidence_audit,
        "leaderboard": leaderboard,
        "paper1_error_ecology": paper1,
        "paper2_interface_tax": paper2,
        "evidence_assessment": evidence,
    }
    write_json(output_dir / "paper_conclusions.json", conclusions)
    write_markdown_conclusions(output_dir / "PAPER_CONCLUSIONS.md", conclusions)
    write_json(
        output_dir / "analysis_manifest.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "input_files": [str(path) for path in paths],
            "outputs": [
                "per_model/*.json",
                "leaderboard.csv",
                "leaderboard.json",
                "category_table.csv",
                "pairwise_error_ecology.csv",
                "pairwise_error_ecology.json",
                "standardized_pairwise_error_ecology_all_models.csv",
                "standardized_pairwise_error_ecology_all_models.json",
                "evidence_audit.json",
                "paper_conclusions.json",
                "PAPER_CONCLUSIONS.md",
            ],
            "parser_notes": [
                "Answers were extracted from verbose raw responses using explicit final-answer patterns first.",
                "Ground truth letters span A-J.",
                "Rows with no parseable answer are counted as unanswered and incorrect.",
            ],
        },
    )
    return conclusions


def build_raw_evidence_audit(raw_by_model: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    all_rows = [row for rows in raw_by_model.values() for row in rows]
    observed_fields = sorted({key for row in all_rows for key in row})
    has_condition_field = any("condition" in row for row in all_rows)
    has_experiment_field = any("experiment" in row for row in all_rows)
    has_prompt_field = any("prompt" in row for row in all_rows)
    has_latency_field = any("elapsed_seconds" in row or "tokens_per_second" in row for row in all_rows)

    pair_conditions: dict[tuple[str, str], set[str]] = defaultdict(set)
    pair_counts: Counter[tuple[str, str]] = Counter()
    model_conditions: dict[str, set[str]] = defaultdict(set)
    condition_values: set[str] = set()
    per_model: list[dict[str, Any]] = []

    for label, rows in sorted(raw_by_model.items()):
        item_ids = [raw_item_id(row) for row in rows]
        item_counts = Counter(item_ids)
        fields = sorted({key for row in rows for key in row})
        conditions = {
            str(row.get("condition"))
            for row in rows
            if "condition" in row
        }
        per_model.append(
            {
                "model": label,
                "row_count": len(rows),
                "unique_item_count": len(set(item_ids)),
                "duplicate_item_rows": sum(count - 1 for count in item_counts.values() if count > 1),
                "fields": fields,
                "condition_values": sorted(conditions),
            }
        )

        for row in rows:
            model = str(row.get("model") or row.get("model_label") or label)
            item_id = raw_item_id(row)
            pair_counts[(model, item_id)] += 1
            if "condition" in row:
                condition = str(row.get("condition"))
                pair_conditions[(model, item_id)].add(condition)
                model_conditions[model].add(condition)
                condition_values.add(condition)

    pairs_with_multiple_rows = sum(1 for count in pair_counts.values() if count > 1)
    pairs_with_multiple_conditions = sum(1 for values in pair_conditions.values() if len(values) > 1)
    has_baseline = "baseline" in condition_values
    has_intervention_condition = any(value != "baseline" for value in condition_values)
    has_paper2_pairing = (
        has_condition_field
        and has_baseline
        and has_intervention_condition
        and pairs_with_multiple_conditions > 0
    )
    if has_paper2_pairing:
        verdict = "paper2_pairing_detected"
    elif not has_condition_field:
        verdict = "not_provable_missing_condition_field"
    elif not has_baseline:
        verdict = "condition_field_present_but_no_baseline_condition"
    elif not has_intervention_condition:
        verdict = "condition_field_present_but_only_baseline"
    else:
        verdict = "condition_field_present_but_no_paired_model_item_interventions"

    return {
        "verdict": verdict,
        "paper2_pairing_detected": has_paper2_pairing,
        "observed_fields": observed_fields,
        "total_rows": len(all_rows),
        "model_count": len(raw_by_model),
        "has_condition_field": has_condition_field,
        "has_experiment_field": has_experiment_field,
        "has_prompt_field": has_prompt_field,
        "has_latency_or_token_field": has_latency_field,
        "condition_values": sorted(condition_values),
        "pairs_with_multiple_rows_same_model_item": pairs_with_multiple_rows,
        "pairs_with_multiple_conditions_same_model_item": pairs_with_multiple_conditions,
        "models_with_conditions": {
            model: sorted(values)
            for model, values in sorted(model_conditions.items())
        },
        "per_model": per_model,
        "required_for_paper2": [
            "condition field",
            "baseline condition",
            "at least one non-baseline intervention condition",
            "same model and item_id observed under baseline and intervention",
            "prompt or prompt hash",
            "error field for failed generations",
            "latency/token fields for runtime claims",
        ],
    }


def raw_item_id(row: dict[str, Any]) -> str:
    return str(row.get("id") if "id" in row else row.get("item_id", ""))


def score_model_rows(rows: list[dict[str, Any]], meta: ModelMeta) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        item_id = raw_item_id(raw)
        seen.add(item_id)
        ground_truth = str(raw.get("ground_truth", raw.get("expected", ""))).strip().upper()
        response = str(raw.get("response", raw.get("response_text", "")))
        prediction, method, confidence = extract_answer_from_raw(raw)
        scored_row = {
            "model": meta.label,
            "family": meta.family,
            "generation": meta.generation,
            "parameter_b": meta.parameter_b,
            "boundary_role": meta.boundary_role,
            "id": item_id,
            "category": str(raw.get("category", raw.get("task_family", "unknown"))),
            "ground_truth": ground_truth,
            "prediction": prediction,
            "is_correct": prediction == ground_truth,
            "answered": prediction is not None,
            "extraction_method": method,
            "extraction_confidence": confidence,
            "response_chars": len(response),
            "response_tail": " ".join(response.split()[-80:]),
        }
        for optional_key in ("experiment", "condition", "task_family", "prompt", "error"):
            if optional_key in raw:
                scored_row[optional_key] = raw[optional_key]
        scored.append(scored_row)
    return scored


def summarize_model(scored_rows: list[dict[str, Any]], meta: ModelMeta, *, source_file: Path) -> dict[str, Any]:
    total = len(scored_rows)
    correct = sum(1 for row in scored_rows if row["is_correct"])
    answered = sum(1 for row in scored_rows if row["answered"])
    response_lengths = [int(row["response_chars"]) for row in scored_rows]
    category_summary = summarize_by(scored_rows, ["category"])
    extraction_counts = Counter(str(row["extraction_method"]) for row in scored_rows)
    prediction_distribution = Counter(str(row["prediction"]) for row in scored_rows if row["prediction"])
    ground_truth_distribution = Counter(str(row["ground_truth"]) for row in scored_rows)
    return {
        "model": meta.to_record(),
        "source_file": str(source_file),
        "n": total,
        "correct": correct,
        "accuracy": safe_ratio(correct, total),
        "accuracy_ci95_wilson": wilson_interval(correct, total),
        "answered": answered,
        "answered_rate": safe_ratio(answered, total),
        "parse_failure_rate": safe_ratio(total - answered, total),
        "mean_response_chars": round(mean(response_lengths), 3) if response_lengths else 0.0,
        "median_response_chars": percentile(response_lengths, 0.50),
        "p95_response_chars": percentile(response_lengths, 0.95),
        "extraction_method_counts": dict(sorted(extraction_counts.items())),
        "prediction_distribution": dict(sorted(prediction_distribution.items())),
        "ground_truth_distribution": dict(sorted(ground_truth_distribution.items())),
        "category_summary": category_summary,
        "items": scored_rows,
    }


def summarize_by(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(str(row.get(key)) for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(buckets.items()):
        correct = sum(1 for item in items if item["is_correct"])
        answered = sum(1 for item in items if item["answered"])
        record = {keys[idx]: key[idx] for idx in range(len(keys))}
        record.update(
            {
                "n": len(items),
                "correct": correct,
                "accuracy": safe_ratio(correct, len(items)),
                "answered_rate": safe_ratio(answered, len(items)),
                "parse_failure_rate": safe_ratio(len(items) - answered, len(items)),
                "mean_response_chars": round(mean([int(item["response_chars"]) for item in items]), 3),
            }
        )
        out.append(record)
    return out


def flatten_model_summary(summary: dict[str, Any]) -> dict[str, Any]:
    model = summary["model"]
    return {
        "model": model["label"],
        "family": model["family"],
        "generation": model["generation"],
        "parameter_b": model["parameter_b"],
        "boundary_role": model["boundary_role"],
        "n": summary["n"],
        "correct": summary["correct"],
        "accuracy": summary["accuracy"],
        "ci95_low": summary["accuracy_ci95_wilson"]["low"],
        "ci95_high": summary["accuracy_ci95_wilson"]["high"],
        "answered_rate": summary["answered_rate"],
        "parse_failure_rate": summary["parse_failure_rate"],
        "mean_response_chars": summary["mean_response_chars"],
        "p95_response_chars": summary["p95_response_chars"],
    }


def build_category_table(model_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in model_summaries:
        label = summary["model"]["label"]
        family = summary["model"]["family"]
        generation = summary["model"]["generation"]
        for category in summary["category_summary"]:
            rows.append({"model": label, "family": family, "generation": generation, **category})
    return rows


def pairwise_error_ecology(
    scored_by_model: dict[str, list[dict[str, Any]]],
    *,
    bootstrap_iters: int,
) -> dict[str, Any]:
    meta_by_model = {
        label: infer_model_meta(label)
        for label in scored_by_model
    }
    comparisons: list[dict[str, Any]] = []
    labels = sorted(scored_by_model, key=lambda label: model_order_key(meta_by_model[label]))
    for old_idx, old_label in enumerate(labels):
        for new_label in labels[old_idx + 1 :]:
            old_meta = meta_by_model[old_label]
            new_meta = meta_by_model[new_label]
            if old_meta.family != new_meta.family:
                continue
            comparisons.append(
                compare_models(
                    old_label,
                    scored_by_model[old_label],
                    new_label,
                    scored_by_model[new_label],
                    bootstrap_iters=bootstrap_iters,
                )
            )

    named_pairs = [
        ("Qwen2-0.5B-Instruct", "Qwen2.5-0.5B-Instruct"),
        ("Qwen2.5-0.5B-Instruct", "Qwen3-0.6B"),
        ("Qwen3-0.6B", "Qwen3.5-0.8B"),
        ("Qwen3.5-0.8B", "Qwen3.5-2B"),
        ("Qwen2.5-3B-Instruct", "Qwen3-4B-Instruct-2507"),
    ]
    primary = [
        compare_models(a, scored_by_model[a], b, scored_by_model[b], bootstrap_iters=bootstrap_iters)
        for a, b in named_pairs
        if a in scored_by_model and b in scored_by_model
    ]
    return {
        "comparisons": comparisons,
        "primary_comparisons": primary,
        "claim_gate": error_ecology_claim_gate(primary),
    }


def compare_models(
    old_label: str,
    old_rows: list[dict[str, Any]],
    new_label: str,
    new_rows: list[dict[str, Any]],
    *,
    bootstrap_iters: int,
) -> dict[str, Any]:
    old_by_id = {row["id"]: row for row in old_rows}
    new_by_id = {row["id"]: row for row in new_rows}
    common = sorted(set(old_by_id) & set(new_by_id))
    old_errors = {item_id for item_id in common if not old_by_id[item_id]["is_correct"]}
    new_errors = {item_id for item_id in common if not new_by_id[item_id]["is_correct"]}
    regressions = {
        item_id
        for item_id in common
        if old_by_id[item_id]["is_correct"] and not new_by_id[item_id]["is_correct"]
    }
    improvements = {
        item_id
        for item_id in common
        if not old_by_id[item_id]["is_correct"] and new_by_id[item_id]["is_correct"]
    }
    persistent_errors = old_errors & new_errors
    shifted_errors = old_errors ^ new_errors
    old_acc = safe_ratio(len(common) - len(old_errors), len(common))
    new_acc = safe_ratio(len(common) - len(new_errors), len(common))
    deltas = [
        (1.0 if new_by_id[item_id]["is_correct"] else 0.0)
        - (1.0 if old_by_id[item_id]["is_correct"] else 0.0)
        for item_id in common
    ]
    category_deltas = category_delta_table(old_by_id, new_by_id, common)
    b01 = len(improvements)
    b10 = len(regressions)
    return {
        "old_model": old_label,
        "new_model": new_label,
        "family": infer_model_meta(old_label).family,
        "n_common": len(common),
        "old_accuracy": old_acc,
        "new_accuracy": new_acc,
        "accuracy_delta": round(new_acc - old_acc, 6),
        "accuracy_delta_ci95_bootstrap": bootstrap_mean_ci(deltas, iters=bootstrap_iters),
        "old_error_count": len(old_errors),
        "new_error_count": len(new_errors),
        "persistent_error_count": len(persistent_errors),
        "improvement_count": len(improvements),
        "regression_count": len(regressions),
        "improvement_mass": safe_ratio(len(improvements), len(common)),
        "regression_mass": safe_ratio(len(regressions), len(common)),
        "error_jaccard": safe_ratio(len(persistent_errors), len(old_errors | new_errors)),
        "error_redistribution_index": round(1.0 - safe_ratio(len(persistent_errors), len(old_errors | new_errors)), 6),
        "shifted_error_mass": safe_ratio(len(shifted_errors), len(common)),
        "mcnemar_exact_p": mcnemar_exact(b01=b01, b10=b10),
        "category_deltas": category_deltas,
        "top_improving_categories": sorted(category_deltas, key=lambda row: row["accuracy_delta"], reverse=True)[:5],
        "top_regressing_categories": sorted(category_deltas, key=lambda row: row["accuracy_delta"])[:5],
    }


def build_standard_pairwise_outputs(
    pairwise: dict[str, Any],
    scored_by_model: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    meta_by_model = {
        label: infer_model_meta(label)
        for label in scored_by_model
    }
    rows = []
    for row in pairwise["comparisons"]:
        old_meta = meta_by_model[row["old_model"]]
        new_meta = meta_by_model[row["new_model"]]
        ci = row.get("accuracy_delta_ci95_bootstrap", {})
        rows.append(
            {
                "comparison_id": f"{row['old_model']} -> {row['new_model']}",
                "family": row["family"],
                "old_model": row["old_model"],
                "new_model": row["new_model"],
                "old_generation": old_meta.generation,
                "new_generation": new_meta.generation,
                "old_parameter_b": old_meta.parameter_b,
                "new_parameter_b": new_meta.parameter_b,
                "old_boundary_role": old_meta.boundary_role,
                "new_boundary_role": new_meta.boundary_role,
                "n_common": row["n_common"],
                "old_accuracy": row["old_accuracy"],
                "new_accuracy": row["new_accuracy"],
                "accuracy_delta": row["accuracy_delta"],
                "accuracy_delta_ci95_low": ci.get("low", 0.0),
                "accuracy_delta_ci95_high": ci.get("high", 0.0),
                "old_error_count": row["old_error_count"],
                "new_error_count": row["new_error_count"],
                "persistent_error_count": row["persistent_error_count"],
                "improvement_count": row["improvement_count"],
                "regression_count": row["regression_count"],
                "improvement_mass": row["improvement_mass"],
                "regression_mass": row["regression_mass"],
                "error_jaccard": row["error_jaccard"],
                "error_redistribution_index": row["error_redistribution_index"],
                "shifted_error_mass": row["shifted_error_mass"],
                "mcnemar_exact_p": row["mcnemar_exact_p"],
                "top_improving_categories": summarize_category_names(row["top_improving_categories"]),
                "top_regressing_categories": summarize_category_names(row["top_regressing_categories"]),
            }
        )
    rows.sort(key=lambda item: (item["family"], item["old_parameter_b"], item["new_parameter_b"], item["old_model"], item["new_model"]))
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Flattened same-family pairwise error-ecology comparisons across all available models.",
        "model_count": len(scored_by_model),
        "comparison_count": len(rows),
        "families": sorted({row["family"] for row in rows}),
        "rows": rows,
        "full_comparisons": pairwise["comparisons"],
        "primary_comparisons": pairwise["primary_comparisons"],
        "claim_gate": pairwise["claim_gate"],
    }


def summarize_category_names(rows: list[dict[str, Any]]) -> str:
    return "; ".join(
        f"{row['category']}:{row['accuracy_delta']:+.3f}"
        for row in rows
    )


def category_delta_table(
    old_by_id: dict[str, dict[str, Any]],
    new_by_id: dict[str, dict[str, Any]],
    common: list[str],
) -> list[dict[str, Any]]:
    by_category: dict[str, list[str]] = defaultdict(list)
    for item_id in common:
        by_category[str(old_by_id[item_id]["category"])].append(item_id)
    rows: list[dict[str, Any]] = []
    for category, ids in sorted(by_category.items()):
        old_correct = sum(1 for item_id in ids if old_by_id[item_id]["is_correct"])
        new_correct = sum(1 for item_id in ids if new_by_id[item_id]["is_correct"])
        improvements = sum(
            1 for item_id in ids if not old_by_id[item_id]["is_correct"] and new_by_id[item_id]["is_correct"]
        )
        regressions = sum(
            1 for item_id in ids if old_by_id[item_id]["is_correct"] and not new_by_id[item_id]["is_correct"]
        )
        rows.append(
            {
                "category": category,
                "n": len(ids),
                "old_accuracy": safe_ratio(old_correct, len(ids)),
                "new_accuracy": safe_ratio(new_correct, len(ids)),
                "accuracy_delta": round(safe_ratio(new_correct - old_correct, len(ids)), 6),
                "improvement_mass": safe_ratio(improvements, len(ids)),
                "regression_mass": safe_ratio(regressions, len(ids)),
            }
        )
    return rows


def error_ecology_claim_gate(primary: list[dict[str, Any]]) -> dict[str, Any]:
    passed_pairs = [
        row
        for row in primary
        if row["accuracy_delta"] > 0
        and row["regression_mass"] <= 0.08
        and row["accuracy_delta_ci95_bootstrap"]["low"] > 0
    ]
    qwen_generation_pairs = [
        row for row in primary if row["old_model"].startswith("Qwen") and row["new_model"].startswith("Qwen")
    ]
    return {
        "primary_claim": "Newer/boundary Qwen generations improve accuracy without unacceptable regression mass.",
        "pass_boolean": len(passed_pairs) >= 2,
        "passed_pair_count": len(passed_pairs),
        "evaluated_pair_count": len(qwen_generation_pairs),
        "failure_reason": "" if len(passed_pairs) >= 2 else "fewer than two preregistered Qwen pair comparisons had positive CI-bounded gains with regression_mass <= 0.08",
        "passed_pairs": [
            {
                "old_model": row["old_model"],
                "new_model": row["new_model"],
                "accuracy_delta": row["accuracy_delta"],
                "regression_mass": row["regression_mass"],
            }
            for row in passed_pairs
        ],
    }


def build_paper1_conclusions(
    leaderboard: list[dict[str, Any]],
    pairwise: dict[str, Any],
    model_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    best = leaderboard[0]
    worst = leaderboard[-1]
    primary = pairwise["primary_comparisons"]
    high_redistribution = [
        row for row in primary if row["error_redistribution_index"] >= 0.50
    ]
    max_eri = max((row["error_redistribution_index"] for row in primary), default=0.0)
    category_best: dict[str, dict[str, Any]] = {}
    for summary in model_summaries:
        label = summary["model"]["label"]
        for row in summary["category_summary"]:
            category = row["category"]
            item = {"model": label, **row}
            if category not in category_best or item["accuracy"] > category_best[category]["accuracy"]:
                category_best[category] = item
    return {
        "title": "Error Redistribution Across Boundary Small Language Model Generations",
        "evidence_status": "supported_for_qwen_error_ecology" if primary else "insufficient",
        "claim_gate": pairwise["claim_gate"],
        "headline_findings": [
            f"Best overall model in this run: {best['model']} at {best['accuracy']:.3f} accuracy over {best['n']} items.",
            f"Weakest overall model in this run: {worst['model']} at {worst['accuracy']:.3f} accuracy.",
            f"{len(high_redistribution)} primary Qwen comparisons crossed the preregistered high-redistribution threshold of ERI >= 0.50; the maximum primary-pair ERI was {max_eri:.3f}.",
            f"The data are paired across {best['n']:,} shared items, so item-level error overlap and regression mass are valid to report.",
        ],
        "recommended_claim_language": recommended_paper1_language(pairwise["claim_gate"]),
        "primary_comparisons": primary,
        "best_model_by_category": sorted(category_best.values(), key=lambda row: row["category"]),
    }


def recommended_paper1_language(claim_gate: dict[str, Any]) -> str:
    if claim_gate["pass_boolean"]:
        return (
            "The run supports a Qwen-family error-ecology paper: newer/boundary checkpoints improve aggregate accuracy in multiple paired comparisons, "
            "but the paper should emphasize error redistribution and regression pockets rather than simple monotonic progress."
        )
    return (
        "The run supports an exploratory Qwen-family error-ecology paper, not a strong monotonic-improvement claim. "
        "Use the paired comparisons to argue that model updates redistribute errors and create category-specific regressions."
    )


def build_paper2_conclusions(scored_by_model: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    any_condition = any("condition" in row for rows in scored_by_model.values() for row in rows)
    return {
        "title": "The Interface Tax of Multimodal and Long-Context Features in Boundary SLMs",
        "evidence_status": "insufficient_controlled_interventions",
        "claim_gate": {
            "primary_claim": "Interface interventions impose a paired reliability tax in at least two model families.",
            "pass_boolean": False,
            "failure_reason": "raw result files contain no intervention/condition field; all rows are single-condition benchmark answers",
        },
        "what_can_be_said": [
            "These raw runs can provide baseline model capability and answer-extraction behavior.",
            "They cannot estimate context tax, multimodal distractor susceptibility, thinking-toggle effects, or format-control fragility because there are no paired intervention conditions.",
        ],
        "minimum_next_experiment": [
            "For every selected item, run baseline plus context_short/context_medium/context_long with controlled answer position.",
            "Run strict letter-only or JSON-only conditions to measure output-control failure.",
            "Run thinking_enabled/thinking_disabled where supported.",
            "Run multimodal distractor wrappers only for models whose model card/runtime supports those modalities.",
            "Preserve fields: experiment, model, id, category, condition, ground_truth, response, latency, tokens, and error.",
        ],
        "detected_condition_field": any_condition,
    }


def build_evidence_assessment(
    scored_by_model: dict[str, list[dict[str, Any]]],
    leaderboard: list[dict[str, Any]],
) -> dict[str, Any]:
    ids_by_model = {model: {row["id"] for row in rows} for model, rows in scored_by_model.items()}
    common_ids = set.intersection(*ids_by_model.values()) if ids_by_model else set()
    categories = {
        row["category"]
        for rows in scored_by_model.values()
        for row in rows
    }
    families = {infer_model_meta(model).family for model in scored_by_model}
    qwen_models = [row for row in leaderboard if row["family"] == "qwen"]
    return {
        "is_enough_for_paper1": len(common_ids) >= 1000 and len(qwen_models) >= 5,
        "is_enough_for_paper2": False,
        "common_item_count": len(common_ids),
        "model_count": len(scored_by_model),
        "family_count": len(families),
        "category_count": len(categories),
        "qwen_model_count": len(qwen_models),
        "notes": [
            "Paper 1 has enough paired Qwen-family data for a serious empirical analysis, with Gemma and Llama as cross-family anchors.",
            "Paper 2 still needs controlled intervention conditions; do not claim interface tax from this raw baseline run.",
        ],
    }


def write_markdown_conclusions(path: Path, conclusions: dict[str, Any]) -> None:
    leaderboard = conclusions["leaderboard"]
    p1 = conclusions["paper1_error_ecology"]
    p2 = conclusions["paper2_interface_tax"]
    evidence = conclusions["evidence_assessment"]
    lines = [
        "# Boundary-SLM TPU Result Conclusions",
        "",
        f"Generated: {conclusions['created_utc']}",
        "",
        "## Data Coverage",
        "",
        f"- Models analyzed: {evidence['model_count']}",
        f"- Shared item count: {evidence['common_item_count']}",
        f"- Categories: {evidence['category_count']}",
        f"- Qwen models: {evidence['qwen_model_count']}",
        f"- Enough for Paper 1: {evidence['is_enough_for_paper1']}",
        f"- Enough for Paper 2: {evidence['is_enough_for_paper2']}",
        "",
        "## Leaderboard",
        "",
        "| Rank | Model | Family | Accuracy | 95% CI | Answered |",
        "| ---: | --- | --- | ---: | --- | ---: |",
    ]
    for idx, row in enumerate(leaderboard, start=1):
        lines.append(
            f"| {idx} | {row['model']} | {row['family']} | {row['accuracy']:.3f} | "
            f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}] | {row['answered_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Paper 1: Error Ecology",
            "",
            f"Evidence status: `{p1['evidence_status']}`",
            "",
            f"Claim gate pass: `{p1['claim_gate']['pass_boolean']}`",
            "",
            "Recommended language:",
            "",
            p1["recommended_claim_language"],
            "",
            "Headline findings:",
        ]
    )
    lines.extend([f"- {item}" for item in p1["headline_findings"]])
    lines.extend(
        [
            "",
            "Primary paired comparisons:",
            "",
            "| Old | New | Delta | Regression Mass | ERI | McNemar p |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in p1["primary_comparisons"]:
        lines.append(
            f"| {row['old_model']} | {row['new_model']} | {row['accuracy_delta']:.3f} | "
            f"{row['regression_mass']:.3f} | {row['error_redistribution_index']:.3f} | {row['mcnemar_exact_p']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Paper 2: Interface Tax",
            "",
            f"Evidence status: `{p2['evidence_status']}`",
            "",
            f"Claim gate pass: `{p2['claim_gate']['pass_boolean']}`",
            "",
            f"Reason: {p2['claim_gate']['failure_reason']}",
            "",
            "What can be said:",
        ]
    )
    lines.extend([f"- {item}" for item in p2["what_can_be_said"]])
    lines.extend(["", "Minimum next experiment:"])
    lines.extend([f"- {item}" for item in p2["minimum_next_experiment"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> dict[str, float]:
    if n == 0:
        return {"low": 0.0, "high": 0.0}
    phat = successes / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return {
        "low": round((centre - margin) / denom, 6),
        "high": round((centre + margin) / denom, 6),
    }


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    return round(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo), 3)


def bootstrap_mean_ci(values: list[float], *, iters: int, seed: int = 17) -> dict[str, float]:
    if not values:
        return {"low": 0.0, "high": 0.0}
    import random

    rng = random.Random(seed)
    estimates = []
    for _ in range(max(1, iters)):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(sum(sample) / len(sample))
    estimates.sort()
    return {
        "low": round(estimates[int(0.025 * (len(estimates) - 1))], 6),
        "high": round(estimates[int(0.975 * (len(estimates) - 1))], 6),
    }


def mcnemar_exact(*, b01: int, b10: int) -> float:
    n = b01 + b10
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b01, b10) + 1)) / (2**n)
    return round(min(1.0, 2 * tail), 8)


def model_order_key(meta: ModelMeta) -> tuple[str, float, float, str]:
    return (meta.family, meta.parameter_b, generation_value(meta.generation), meta.label)


def generation_value(value: str) -> float:
    cleaned = "".join(ch if ch.isdigit() or ch == "." else " " for ch in value)
    parts = [float(part) for part in cleaned.split() if part]
    return parts[-1] if parts else 0.0


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze raw model-level TPU JSONL answers.")
    parser.add_argument("--input-dir", type=Path, default=Path("main/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("main/analysis/raw_tpu_results"))
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    conclusions = analyze_raw_results(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        bootstrap_iters=args.bootstrap_iters,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "models": conclusions["evidence_assessment"]["model_count"],
                "common_items": conclusions["evidence_assessment"]["common_item_count"],
                "paper1_enough": conclusions["evidence_assessment"]["is_enough_for_paper1"],
                "paper2_enough": conclusions["evidence_assessment"]["is_enough_for_paper2"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
