from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import math
from statistics import mean, median
from typing import Any, Iterable

import pandas as pd

from boundary_slm.io import write_csv, write_json
from boundary_slm.raw_results import (
    bootstrap_mean_ci,
    infer_model_meta,
    mcnemar_exact,
    model_order_key,
    safe_ratio,
)


WILD_REPO = "kensho/WILD"
WILD_URL = "https://huggingface.co/datasets/kensho/WILD"
DEFAULT_INPUT_URI = "hf://datasets/kensho/WILD/data.parquet"
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "llama-3.2-1b",
    "llama-3.2-3b",
]
DEFAULT_TASKS = [
    "mmlu_pro",
    "mmlu",
    "arc_challenge",
    "gsm8k",
    "hellaswag",
    "piqa",
    "boolq",
    "ifeval",
    "bbh",
    "math",
]
WILD_COLUMNS = ["model", "task", "subtask", "item_id", "score", "input_tokens", "output_tokens"]


def safe_ratio_or_none(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value in {None, ""}:
            continue
        try:
            value_float = float(value)
        except Exception:
            continue
        if math.isfinite(value_float):
            values.append(value_float)
    return values


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return round(ordered[lo], 6)
    return round(ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo), 6)


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 6) if values else None


def median_or_none(values: list[float]) -> float | None:
    return round(median(values), 6) if values else None


def pct(value: Any, *, signed: bool = False) -> str:
    if value in {None, ""}:
        return "--"
    try:
        value_float = float(value) * 100.0
    except Exception:
        return "--"
    sign = "+" if signed else ""
    return f"{value_float:{sign}.1f}\\%"


def count_text(value: Any) -> str:
    if value in {None, ""}:
        return "--"
    try:
        value_float = float(value)
    except Exception:
        return "--"
    if value_float.is_integer():
        return str(int(value_float))
    return f"{value_float:.1f}"


def short_model(value: str) -> str:
    return value.split("/")[-1]


def normalize_records(frame: pd.DataFrame, *, source_id: str = "wild") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        score_value = int(raw["score"])
        if score_value not in {0, 1}:
            continue
        model = str(raw["model"])
        task = str(raw["task"])
        subtask = str(raw["subtask"])
        item_id = str(raw["item_id"])
        meta = infer_model_meta(model)
        rows.append(
            {
                "source_id": source_id,
                "model": model,
                "family": meta.family,
                "generation": meta.generation,
                "parameter_b": meta.parameter_b,
                "boundary_role": meta.boundary_role,
                "task": task,
                "subtask": subtask,
                "item_id": item_id,
                "id": f"{task}:{subtask}:{item_id}",
                "category": task,
                "score": score_value,
                "is_correct": bool(score_value),
                "input_tokens": int(raw.get("input_tokens", 0) or 0),
                "output_tokens": int(raw.get("output_tokens", 0) or 0),
            }
        )
    return rows


def load_wild_rows(
    input_uri: str,
    *,
    models: list[str],
    tasks: list[str],
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    frame = pd.read_parquet(input_uri, columns=WILD_COLUMNS)
    filtered = frame[frame["model"].isin(models) & frame["task"].isin(tasks)]
    if max_rows is not None:
        filtered = filtered.head(max_rows)
    return normalize_records(filtered)


def index_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in rows}


def compare_records(
    old_model: str,
    old_rows: list[dict[str, Any]],
    new_model: str,
    new_rows: list[dict[str, Any]],
    *,
    task_scope: str,
    bootstrap_iters: int,
) -> dict[str, Any] | None:
    old_by_id = index_by_id(old_rows)
    new_by_id = index_by_id(new_rows)
    common = sorted(set(old_by_id) & set(new_by_id))
    if not common:
        return None
    old_errors = {item_id for item_id in common if not old_by_id[item_id]["is_correct"]}
    new_errors = {item_id for item_id in common if not new_by_id[item_id]["is_correct"]}
    improvements = {
        item_id
        for item_id in common
        if not old_by_id[item_id]["is_correct"] and new_by_id[item_id]["is_correct"]
    }
    regressions = {
        item_id
        for item_id in common
        if old_by_id[item_id]["is_correct"] and not new_by_id[item_id]["is_correct"]
    }
    persistent_errors = old_errors & new_errors
    persistent_correct = {
        item_id
        for item_id in common
        if old_by_id[item_id]["is_correct"] and new_by_id[item_id]["is_correct"]
    }
    old_acc = safe_ratio(len(common) - len(old_errors), len(common))
    new_acc = safe_ratio(len(common) - len(new_errors), len(common))
    old_error_rate = safe_ratio(len(old_errors), len(common))
    churn_count = len(improvements) + len(regressions)
    if bootstrap_iters > 0:
        deltas = [
            (1.0 if new_by_id[item_id]["is_correct"] else 0.0)
            - (1.0 if old_by_id[item_id]["is_correct"] else 0.0)
            for item_id in common
        ]
        delta_ci = bootstrap_mean_ci(deltas, iters=bootstrap_iters)
    else:
        delta_ci = {"low": 0.0, "high": 0.0}
    old_meta = infer_model_meta(old_model)
    new_meta = infer_model_meta(new_model)
    return {
        "source_id": "wild",
        "comparison_id": f"{task_scope}:{old_model} -> {new_model}",
        "task_scope": task_scope,
        "family": old_meta.family,
        "old_model": old_model,
        "new_model": new_model,
        "old_generation": old_meta.generation,
        "new_generation": new_meta.generation,
        "old_parameter_b": old_meta.parameter_b,
        "new_parameter_b": new_meta.parameter_b,
        "n_common": len(common),
        "old_accuracy": old_acc,
        "new_accuracy": new_acc,
        "accuracy_delta": round(new_acc - old_acc, 6),
        "accuracy_delta_ci95_low": delta_ci["low"],
        "accuracy_delta_ci95_high": delta_ci["high"],
        "old_error_count": len(old_errors),
        "new_error_count": len(new_errors),
        "persistent_correct_count": len(persistent_correct),
        "persistent_error_count": len(persistent_errors),
        "improvement_count": len(improvements),
        "regression_count": len(regressions),
        "improvement_mass": safe_ratio(len(improvements), len(common)),
        "regression_mass": safe_ratio(len(regressions), len(common)),
        "churn_mass": safe_ratio(churn_count, len(common)),
        "error_persistence": safe_ratio_or_none(len(persistent_errors), len(old_errors)),
        "correction_rate": safe_ratio_or_none(len(improvements), len(old_errors)),
        "normalized_regression_burden": safe_ratio_or_none(safe_ratio(len(regressions), len(common)), old_error_rate),
        "error_jaccard": safe_ratio(len(persistent_errors), len(old_errors | new_errors)),
        "error_redistribution_index": round(1.0 - safe_ratio(len(persistent_errors), len(old_errors | new_errors)), 6),
        "mcnemar_exact_p": mcnemar_exact(b01=len(improvements), b10=len(regressions)),
        "claim_use": "external_item_level_correctness_replication",
        "claim_boundary": "WILD provides binary correctness only; no raw-response parser or prompt-format claims are made.",
    }


def pairwise_metrics(records: list[dict[str, Any]], *, bootstrap_iters: int) -> list[dict[str, Any]]:
    by_scope_model: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_scope_model[(str(row["task"]), str(row["model"]))].append(row)
        by_scope_model[("__all_selected__", str(row["model"]))].append(row)

    scopes = sorted({scope for scope, _model in by_scope_model})
    output: list[dict[str, Any]] = []
    for scope in scopes:
        models = sorted(
            {model for row_scope, model in by_scope_model if row_scope == scope},
            key=lambda value: model_order_key(infer_model_meta(value)),
        )
        for old_idx, old_model in enumerate(models):
            for new_model in models[old_idx + 1 :]:
                if infer_model_meta(old_model).family != infer_model_meta(new_model).family:
                    continue
                row = compare_records(
                    old_model,
                    by_scope_model[(scope, old_model)],
                    new_model,
                    by_scope_model[(scope, new_model)],
                    task_scope=scope,
                    bootstrap_iters=bootstrap_iters,
                )
                if row:
                    output.append(row)
    return output


def model_task_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["model"]), str(row["task"]))].append(row)
    output: list[dict[str, Any]] = []
    for (model, task), rows in sorted(grouped.items()):
        meta = infer_model_meta(model)
        output.append(
            {
                "source_id": "wild",
                "model": model,
                "family": meta.family,
                "generation": meta.generation,
                "parameter_b": meta.parameter_b,
                "task": task,
                "n": len(rows),
                "accuracy": round(mean([1.0 if row["is_correct"] else 0.0 for row in rows]), 6),
                "mean_input_tokens": round(mean([int(row["input_tokens"]) for row in rows]), 3),
                "mean_output_tokens": round(mean([int(row["output_tokens"]) for row in rows]), 3),
            }
        )
    return output


def task_family_summary(pairwise_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pairwise_rows:
        grouped[(str(row["task_scope"]), str(row["family"]))].append(row)
    output: list[dict[str, Any]] = []
    for (task_scope, family), rows in sorted(grouped.items()):
        output.append(
            {
                "source_id": "wild",
                "task_scope": task_scope,
                "family": family,
                "pair_count": len(rows),
                "mean_accuracy_delta": mean_or_none(numeric_values(rows, "accuracy_delta")),
                "mean_churn_mass": mean_or_none(numeric_values(rows, "churn_mass")),
                "mean_regression_mass": mean_or_none(numeric_values(rows, "regression_mass")),
                "mean_error_persistence": mean_or_none(numeric_values(rows, "error_persistence")),
                "mean_normalized_regression_burden": mean_or_none(numeric_values(rows, "normalized_regression_burden")),
            }
        )
    return output


def task_dispersion_summary(pairwise_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scoped = [row for row in pairwise_rows if row["task_scope"] != "__all_selected__"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scoped:
        grouped[str(row["family"])].append(row)
        grouped["all_selected_families"].append(row)
    output: list[dict[str, Any]] = []
    for family, rows in sorted(grouped.items()):
        churn = numeric_values(rows, "churn_mass")
        regression = numeric_values(rows, "regression_mass")
        persistence = numeric_values(rows, "error_persistence")
        nrb = numeric_values(rows, "normalized_regression_burden")
        shared_items = numeric_values(rows, "n_common")
        task_means: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            task_means[str(row["task_scope"])].append(row)
        ranked_churn = sorted(
            (
                (task, mean_or_none(numeric_values(vals, "churn_mass")))
                for task, vals in task_means.items()
            ),
            key=lambda item: (item[1] is None, 0.0 if item[1] is None else -item[1], item[0]),
        )
        ranked_regression = sorted(
            (
                (task, mean_or_none(numeric_values(vals, "regression_mass")))
                for task, vals in task_means.items()
            ),
            key=lambda item: (item[1] is None, 0.0 if item[1] is None else -item[1], item[0]),
        )
        output.append(
            {
                "source_id": "wild",
                "family": family,
                "task_count": len(task_means),
                "pair_count": len(rows),
                "median_shared_items": median_or_none(shared_items),
                "min_shared_items": min(shared_items) if shared_items else None,
                "max_shared_items": max(shared_items) if shared_items else None,
                "median_churn_mass": median_or_none(churn),
                "churn_mass_iqr_low": percentile(churn, 0.25),
                "churn_mass_iqr_high": percentile(churn, 0.75),
                "min_churn_mass": min(churn) if churn else None,
                "max_churn_mass": max(churn) if churn else None,
                "median_regression_mass": median_or_none(regression),
                "median_error_persistence": median_or_none(persistence),
                "median_normalized_regression_burden": median_or_none(nrb),
                "highest_churn_task": ranked_churn[0][0] if ranked_churn else "",
                "highest_churn_task_mean": ranked_churn[0][1] if ranked_churn else None,
                "highest_regression_task": ranked_regression[0][0] if ranked_regression else "",
                "highest_regression_task_mean": ranked_regression[0][1] if ranked_regression else None,
            }
        )
    return output


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
    return "".join(replacements.get(char, char) for char in text)


def write_summary_table(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("External source", "WILD"),
        ("Evidence tier", "B: item-level correctness"),
        ("Models retained", report["model_count"]),
        ("Tasks retained", report["task_count"]),
        ("Normalized records", report["normalized_record_count"]),
        ("Pairwise comparisons", f"{report['pairwise_comparison_count']} ({report['all_selected_pair_count']} all-selected + {report['task_level_pair_count']} task-level)"),
        ("All-selected Qwen mean churn", f"{100 * report['qwen_all_selected_mean_churn']:.1f}%"),
        ("WILD source scale", "65 models; 109,564 items; 163 tasks; 27 datasets"),
    ]
    lines = [r"\begin{tabular}{ll}", r"\toprule", r"Field & Value \\", r"\midrule"]
    for key, value in rows:
        lines.append(f"{latex_escape(key)} & {latex_escape(value)} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_top_pair_table(path: Path, pairwise_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = [
        row
        for row in pairwise_rows
        if row["task_scope"] == "__all_selected__" and row["family"] in {"qwen", "llama"}
    ][:6]
    lines = [
        r"\begin{tabular}{lllrrrrrrrr}",
        r"\toprule",
        r"Family & Current & Candidate & $N$ & $\Delta$ & $I$ & $R$ & Churn & Persist. & Corr. & NRB \\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(
            f"{latex_escape(row['family'])} & {latex_escape(short_model(row['old_model']))} & "
            f"{latex_escape(short_model(row['new_model']))} & {int(row['n_common'])} & "
            f"{pct(row['accuracy_delta'], signed=True)} & "
            f"{pct(row['improvement_mass'])} & {pct(row['regression_mass'])} & "
            f"{pct(row['churn_mass'])} & {pct(row['error_persistence'])} & "
            f"{pct(row['correction_rate'])} & {pct(row['normalized_regression_burden'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_task_dispersion_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = [row for row in rows if row["family"] in {"all_selected_families", "qwen", "llama"}]
    lines = [
        r"\begin{tabular}{lrrrrrrrl}",
        r"\toprule",
        r"Scope & Tasks & Pairs & Median $N$ & Median churn & Churn IQR & Median $R$ & Median persist. & Highest-churn task \\",
        r"\midrule",
    ]
    for row in selected:
        iqr = f"[{pct(row['churn_mass_iqr_low'])}, {pct(row['churn_mass_iqr_high'])}]"
        lines.append(
            f"{latex_escape(row['family'])} & {row['task_count']} & {row['pair_count']} & "
            f"{count_text(row['median_shared_items'])} & "
            f"{pct(row['median_churn_mass'])} & {iqr} & {pct(row['median_regression_mass'])} & "
            f"{pct(row['median_error_persistence'])} & "
            f"{latex_escape(row['highest_churn_task'])} ({pct(row['highest_churn_task_mean'])}) \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_records_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(path, index=False)


def generate(
    *,
    input_uri: str,
    output_dir: Path,
    models: list[str],
    tasks: list[str],
    bootstrap_iters: int = 500,
    max_rows: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_wild_rows(input_uri, models=models, tasks=tasks, max_rows=max_rows)
    pairwise_rows = pairwise_metrics(records, bootstrap_iters=bootstrap_iters)
    model_summary = model_task_summary(records)
    family_summary = task_family_summary(pairwise_rows)
    dispersion_summary = task_dispersion_summary(pairwise_rows)
    qwen_all = [
        row for row in pairwise_rows if row["task_scope"] == "__all_selected__" and row["family"] == "qwen"
    ]
    all_selected_pair_count = sum(1 for row in pairwise_rows if row["task_scope"] == "__all_selected__")
    task_level_pair_count = sum(1 for row in pairwise_rows if row["task_scope"] != "__all_selected__")
    all_selected_n_values = [
        int(row["n_common"]) for row in pairwise_rows if row["task_scope"] == "__all_selected__"
    ]
    task_level_n_values = [
        int(row["n_common"]) for row in pairwise_rows if row["task_scope"] != "__all_selected__"
    ]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_id": "wild",
        "source_url": WILD_URL,
        "source_repo": WILD_REPO,
        "source_input_uri": input_uri,
        "license": "apache-2.0",
        "evidence_tier": "B",
        "claim_use": "item_level_correctness_replication",
        "models_requested": models,
        "tasks_requested": tasks,
        "model_count": len({row["model"] for row in records}),
        "task_count": len({row["task"] for row in records}),
        "normalized_record_count": len(records),
        "pairwise_comparison_count": len(pairwise_rows),
        "all_selected_pair_count": all_selected_pair_count,
        "task_level_pair_count": task_level_pair_count,
        "all_selected_shared_item_count_min": min(all_selected_n_values) if all_selected_n_values else 0,
        "all_selected_shared_item_count_median": int(median(all_selected_n_values)) if all_selected_n_values else 0,
        "all_selected_shared_item_count_max": max(all_selected_n_values) if all_selected_n_values else 0,
        "task_level_shared_item_count_min": min(task_level_n_values) if task_level_n_values else 0,
        "task_level_shared_item_count_median": int(median(task_level_n_values)) if task_level_n_values else 0,
        "task_level_shared_item_count_max": max(task_level_n_values) if task_level_n_values else 0,
        "qwen_all_selected_pair_count": len(qwen_all),
        "qwen_all_selected_mean_churn": round(mean([float(row["churn_mass"]) for row in qwen_all]), 6) if qwen_all else 0.0,
        "wild_paper_source_scale": {
            "models": 65,
            "unique_items": 109564,
            "tasks": 163,
            "datasets": 27,
        },
        "public_text_policy": "Only hashed WILD item ids and binary correctness scores are written; no prompt text or model responses are emitted.",
        "claim_boundary": "WILD supports external item-level correctness replication only, not raw-response parsing claims.",
        "claim_ready": bool(records and pairwise_rows and qwen_all),
    }
    write_records_parquet(output_dir / "wild_normalized_records.parquet", records)
    write_json(output_dir / "wild_pairwise_replacement_metrics.json", pairwise_rows)
    write_csv(output_dir / "wild_pairwise_replacement_metrics.csv", pairwise_rows)
    write_csv(output_dir / "wild_model_task_summary.csv", model_summary)
    write_csv(output_dir / "wild_task_family_summary.csv", family_summary)
    write_csv(output_dir / "wild_task_dispersion_summary.csv", dispersion_summary)
    write_json(output_dir / "source_coverage_report.json", report)
    write_json(output_dir / "external_claim_check.json", {
        "source_id": "wild",
        "claim_ready": report["claim_ready"],
        "claim_use": report["claim_use"],
        "claim_boundary": report["claim_boundary"],
        "pairwise_comparison_count": report["pairwise_comparison_count"],
        "qwen_all_selected_pair_count": report["qwen_all_selected_pair_count"],
    })
    write_summary_table(output_dir / "tables" / "wild_replication_summary.tex", report)
    write_top_pair_table(output_dir / "tables" / "wild_all_selected_pairs.tex", pairwise_rows)
    write_task_dispersion_table(output_dir / "tables" / "wild_task_dispersion.tex", dispersion_summary)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest WILD item-level correctness as external replication evidence.")
    parser.add_argument("--input-uri", default=DEFAULT_INPUT_URI)
    parser.add_argument("--output-dir", type=Path, default=Path("main/analysis/external_evidence"))
    parser.add_argument("--model", action="append", dest="models", default=None)
    parser.add_argument("--task", action="append", dest="tasks", default=None)
    parser.add_argument("--bootstrap-iters", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    report = generate(
        input_uri=args.input_uri,
        output_dir=args.output_dir,
        models=args.models or DEFAULT_MODELS,
        tasks=args.tasks or DEFAULT_TASKS,
        bootstrap_iters=args.bootstrap_iters,
        max_rows=args.max_rows,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
