from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from boundary_slm.io import write_csv, write_json


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def f(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def ratio_or_none(numerator: float, denominator: float) -> float | None:
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


def pct(value: Any, *, signed: bool = False) -> str:
    if value in {None, ""}:
        return "--"
    try:
        value_float = 100.0 * float(value)
    except Exception:
        return "--"
    sign = "+" if signed else ""
    return f"{value_float:{sign}.1f}"


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


def generation_number(value: Any) -> float:
    text = str(value).lower()
    numeric = "".join(ch for ch in text if ch.isdigit() or ch == ".")
    if numeric.count(".") > 1:
        first, *rest = numeric.split(".")
        numeric = first + "." + "".join(rest)
    return f(numeric)


def path_interpretation(row: dict[str, Any]) -> str:
    if f(row.get("accuracy_delta")) <= 0:
        return "negative swap"
    old_gen = generation_number(row.get("old_generation"))
    new_gen = generation_number(row.get("new_generation"))
    old_param = f(row.get("old_parameter_b"))
    new_param = f(row.get("new_parameter_b"))
    if new_gen > old_gen and new_param > old_param:
        return "newer+larger"
    if new_gen > old_gen and new_param >= old_param:
        return "newer"
    if math.isclose(new_gen, old_gen) and new_param > old_param:
        return "same-gen larger"
    if math.isclose(new_gen, old_gen) and math.isclose(new_param, old_param):
        return "same-size variant"
    if new_param > old_param:
        return "larger; chronology unclear"
    return "all-pairs only"


def review_flag(row: dict[str, Any]) -> str:
    if f(row.get("accuracy_delta")) <= 0:
        return "not upgrade"
    flags = 0
    if f(row.get("churn_mass")) >= 0.25:
        flags += 1
    if f(row.get("regression_mass")) >= 0.08:
        flags += 1
    if f(row.get("normalized_regression_burden")) >= 0.10:
        flags += 1
    if abs(f(row.get("accuracy_delta"))) <= 0.05:
        flags += 1
    if flags >= 2:
        return "manual required"
    if flags == 1:
        return "manual advised"
    return "routine"


def enrich_pairwise(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        n = f(row["n_common"])
        old_error_count = f(row["old_error_count"])
        old_correct_count = n - old_error_count
        old_error_rate = old_error_count / n if n else 0.0
        improvement_mass = f(row["improvement_mass"])
        regression_mass = f(row["regression_mass"])
        churn_mass = improvement_mass + regression_mass
        accuracy_delta = f(row["accuracy_delta"])
        persistent_error_count = f(row["persistent_error_count"])
        regression_count = f(row["regression_count"])
        improvement_count = f(row["improvement_count"])
        normalized_regression_burden = ratio_or_none(regression_mass, old_error_rate)
        error_persistence = ratio_or_none(persistent_error_count, old_error_count)
        correction_rate = ratio_or_none(improvement_count, old_error_count)
        enriched = dict(row)
        enriched.update(
            {
                "n_common": int(n),
                "old_accuracy": f(row["old_accuracy"]),
                "new_accuracy": f(row["new_accuracy"]),
                "accuracy_delta": accuracy_delta,
                "old_error_rate": round(old_error_rate, 6),
                "error_persistence": error_persistence,
                "correction_rate": correction_rate,
                "regression_rate_on_old_correct": round(regression_count / old_correct_count, 6) if old_correct_count else 0.0,
                "churn_mass": round(churn_mass, 6),
                "normalized_regression_burden": normalized_regression_burden,
                "regression_gap": normalized_regression_burden,
                "net_gain_per_changed_item": round(accuracy_delta / churn_mass, 6) if churn_mass else 0.0,
                "improvement_to_regression_ratio": round(improvement_mass / regression_mass, 6) if regression_mass else math.inf,
                "positive_delta": accuracy_delta > 0,
                "path_interpretation": path_interpretation(row),
                "review_flag": review_flag(
                    {
                        **row,
                        "churn_mass": churn_mass,
                        "normalized_regression_burden": normalized_regression_burden,
                    }
                ),
            }
        )
        out.append(enriched)
    return out


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def bootstrap_ci(values: list[float], *, seed: int = 20260617, iters: int = 5000) -> dict[str, float]:
    if not values:
        return {"low": 0.0, "high": 0.0}
    rng = RandomLCG(seed)
    estimates = []
    for _ in range(iters):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(mean(sample))
    return {
        "low": round(percentile(estimates, 0.025), 6),
        "high": round(percentile(estimates, 0.975), 6),
    }


class RandomLCG:
    def __init__(self, seed: int) -> None:
        self.state = seed & 0x7FFFFFFF

    def randrange(self, limit: int) -> int:
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return self.state % limit


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def fisher_r_ci(r: float, n: int, z: float = 1.959963984540054) -> dict[str, float]:
    """Approximate Pearson-r interval; descriptive here because pairs share models."""
    if n <= 3:
        return {"low": 0.0, "high": 0.0}
    if r >= 0.999999:
        return {"low": 1.0, "high": 1.0}
    if r <= -0.999999:
        return {"low": -1.0, "high": -1.0}
    clipped = max(min(r, 0.999999), -0.999999)
    center = math.atanh(clipped)
    se = 1.0 / math.sqrt(n - 3)
    return {
        "low": round(math.tanh(center - z * se), 6),
        "high": round(math.tanh(center + z * se), 6),
    }


def family_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family"])].append(row)
    out = []
    for family, vals in sorted(grouped.items()):
        persistence = numeric_values(vals, "error_persistence")
        regression_gap = numeric_values(vals, "regression_gap")
        churn = numeric_values(vals, "churn_mass")
        delta = numeric_values(vals, "accuracy_delta")
        old_accuracy = numeric_values(vals, "old_accuracy")
        new_accuracy = numeric_values(vals, "new_accuracy")
        regression_mass = numeric_values(vals, "regression_mass")
        improvement_mass = numeric_values(vals, "improvement_mass")
        correction_rate = numeric_values(vals, "correction_rate")
        persistence_ci = bootstrap_ci(persistence)
        regression_gap_ci = bootstrap_ci(regression_gap, seed=20260618)
        churn_ci = bootstrap_ci(churn, seed=20260619)
        out.append(
            {
                "family": family,
                "pair_count": len(vals),
                "model_count_in_pairs": len({row["old_model"] for row in vals} | {row["new_model"] for row in vals}),
                "mean_old_accuracy": round(mean(old_accuracy), 6),
                "mean_new_accuracy": round(mean(new_accuracy), 6),
                "mean_accuracy_delta": round(mean(delta), 6),
                "mean_error_persistence": round(mean(persistence), 6),
                "error_persistence_sd": round(stdev(persistence), 6),
                "error_persistence_ci_low": persistence_ci["low"],
                "error_persistence_ci_high": persistence_ci["high"],
                "mean_regression_gap": round(mean(regression_gap), 6),
                "mean_normalized_regression_burden": round(mean(regression_gap), 6),
                "regression_gap_ci_low": regression_gap_ci["low"],
                "regression_gap_ci_high": regression_gap_ci["high"],
                "normalized_regression_burden_ci_low": regression_gap_ci["low"],
                "normalized_regression_burden_ci_high": regression_gap_ci["high"],
                "mean_churn_mass": round(mean(churn), 6),
                "churn_mass_ci_low": churn_ci["low"],
                "churn_mass_ci_high": churn_ci["high"],
                "mean_regression_mass": round(mean(regression_mass), 6),
                "mean_improvement_mass": round(mean(improvement_mass), 6),
                "mean_correction_rate": round(mean(correction_rate), 6),
                "accuracy_delta_to_regression_gap_r": round(pearson(delta, regression_gap), 6),
                "positive_delta_pairs": sum(1 for row in vals if row["positive_delta"]),
            }
        )
    return out


def model_level_metric(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get(metric) in {None, ""}:
            continue
        value = f(row[metric])
        buckets[(row["family"], row["old_model"])].append(value)
        buckets[(row["family"], row["new_model"])].append(value)
    return [
        {
            "family": family,
            "model": model,
            f"mean_{metric}_across_incident_pairs": round(mean(values), 6),
            "incident_pair_count": len(values),
        }
        for (family, model), values in sorted(buckets.items())
    ]


def exact_family_permutation_p(model_rows: list[dict[str, Any]], metric_key: str, family_a: str, family_b: str) -> dict[str, Any]:
    values_a = [f(row[metric_key]) for row in model_rows if row["family"] == family_a]
    values_b = [f(row[metric_key]) for row in model_rows if row["family"] == family_b]
    values = values_a + values_b
    n_a = len(values_a)
    if not values_a or not values_b:
        return {"observed_difference": 0.0, "exact_p": 1.0, "n_permutations": 0}
    observed = mean(values_a) - mean(values_b)
    extreme = 0
    total = 0
    indices = range(len(values))
    for combo in itertools.combinations(indices, n_a):
        combo = set(combo)
        perm_a = [values[idx] for idx in indices if idx in combo]
        perm_b = [values[idx] for idx in indices if idx not in combo]
        diff = mean(perm_a) - mean(perm_b)
        if abs(diff) >= abs(observed) - 1e-12:
            extreme += 1
        total += 1
    return {
        "family_a": family_a,
        "family_b": family_b,
        "metric": metric_key,
        "family_a_model_count": len(values_a),
        "family_b_model_count": len(values_b),
        "family_a_mean": round(mean(values_a), 6),
        "family_b_mean": round(mean(values_b), 6),
        "observed_difference": round(observed, 6),
        "exact_p": round(extreme / total, 6) if total else 1.0,
        "n_permutations": total,
    }


def leave_one_model_out_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    for family, vals in sorted(grouped.items()):
        models = sorted({row["old_model"] for row in vals} | {row["new_model"] for row in vals})
        for metric in ["regression_gap", "churn_mass", "improvement_mass", "regression_mass", "error_persistence"]:
            metric_vals = [row for row in vals if row.get(metric) not in {None, ""}]
            full_r = pearson([f(row["accuracy_delta"]) for row in metric_vals], [f(row[metric]) for row in metric_vals])
            ci = fisher_r_ci(full_r, len(metric_vals))
            loo = []
            for model in models:
                kept = [
                    row
                    for row in metric_vals
                    if row["old_model"] != model and row["new_model"] != model
                ]
                if len(kept) >= 3:
                    loo.append(pearson([f(row["accuracy_delta"]) for row in kept], [f(row[metric]) for row in kept]))
            out.append(
                {
                    "family": family,
                    "metric": metric,
                    "model_count": len(models),
                    "pair_count": len(metric_vals),
                    "full_pairwise_r": round(full_r, 6),
                    "fisher_ci95_low": ci["low"],
                    "fisher_ci95_high": ci["high"],
                    "leave_one_model_out_min_r": round(min(loo), 6) if loo else 0.0,
                    "leave_one_model_out_max_r": round(max(loo), 6) if loo else 0.0,
                    "leave_one_model_out_runs": len(loo),
                }
            )
    return out


def successful_upgrade_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successful = [row for row in rows if row["positive_delta"]]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in successful:
        grouped[row["family"]].append(row)
    out = []
    for family, vals in sorted(grouped.items()):
        out.append(
            {
                "family": family,
                "positive_delta_pair_count": len(vals),
                "mean_accuracy_delta": round(mean([f(row["accuracy_delta"]) for row in vals]), 6),
                "mean_churn_mass": round(mean([f(row["churn_mass"]) for row in vals]), 6),
                "mean_improvement_mass": round(mean([f(row["improvement_mass"]) for row in vals]), 6),
                "mean_regression_mass": round(mean([f(row["regression_mass"]) for row in vals]), 6),
                "mean_error_persistence": round(mean(numeric_values(vals, "error_persistence")), 6) if numeric_values(vals, "error_persistence") else 0.0,
                "mean_correction_rate": round(mean(numeric_values(vals, "correction_rate")), 6) if numeric_values(vals, "correction_rate") else 0.0,
                "mean_regression_gap": round(mean(numeric_values(vals, "regression_gap")), 6) if numeric_values(vals, "regression_gap") else 0.0,
                "mean_net_gain_per_changed_item": round(mean([f(row["net_gain_per_changed_item"]) for row in vals]), 6),
                "accuracy_delta_to_churn_mass_r": round(
                    pearson([f(row["accuracy_delta"]) for row in vals], [f(row["churn_mass"]) for row in vals]),
                    6,
                ),
            }
        )
    return out


def near_parity_rows(rows: list[dict[str, Any]], threshold: float = 0.05) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if abs(f(row["accuracy_delta"])) <= threshold:
            out.append(
                {
                    "comparison_id": row["comparison_id"],
                    "family": row["family"],
                    "accuracy_delta": row["accuracy_delta"],
                    "churn_mass": row["churn_mass"],
                    "improvement_mass": row["improvement_mass"],
                    "regression_mass": row["regression_mass"],
                    "error_persistence": row["error_persistence"],
                    "error_redistribution_index": row["error_redistribution_index"],
                }
            )
    return sorted(out, key=lambda item: (-f(item["churn_mass"]), item["comparison_id"]))


def category_summary(full_comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for comparison in full_comparisons:
        family = comparison["family"]
        for row in comparison["category_deltas"]:
            buckets[(family, row["category"])].append(row)
    out = []
    for (family, category), vals in sorted(buckets.items()):
        out.append(
            {
                "family": family,
                "category": category,
                "pair_count": len(vals),
                "mean_accuracy_delta": round(mean([f(row["accuracy_delta"]) for row in vals]), 6),
                "mean_improvement_mass": round(mean([f(row["improvement_mass"]) for row in vals]), 6),
                "mean_regression_mass": round(mean([f(row["regression_mass"]) for row in vals]), 6),
            }
        )
    return out


def extraction_coverage(input_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((input_dir / "per_model").glob("*.json")):
        payload = read_json(path)
        counts = payload.get("extraction_method_counts", {})
        model = payload.get("model", {})
        raw_prediction = int(counts.get("raw_prediction", 0))
        explicit_marker = sum(
            int(counts.get(key, 0))
            for key in ["answer_is", "correct_answer", "final_answer", "option_is"]
        )
        fallback_letter = sum(
            int(counts.get(key, 0))
            for key in ["single_letter", "tail_option", "last_standalone_letter"]
        )
        unanswered = sum(
            int(counts.get(key, 0))
            for key in ["empty", "none", "prompt_echo_without_completion"]
        )
        n = int(payload.get("n", 0))
        rows.append(
            {
                "model": model.get("label", path.stem),
                "family": model.get("family", "unknown"),
                "n": n,
                "raw_prediction_rows": raw_prediction,
                "explicit_marker_rows": explicit_marker,
                "fallback_letter_rows": fallback_letter,
                "unanswered_rows": unanswered,
                "answered_rows": max(0, n - unanswered),
                "has_saved_prediction_field": raw_prediction > 0,
            }
        )
    return rows


def extraction_coverage_by_family(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family"])].append(row)
    out = []
    for family, vals in sorted(grouped.items()):
        n = sum(int(row["n"]) for row in vals)
        out.append(
            {
                "family": family,
                "model_count": len(vals),
                "row_count": n,
                "models_with_saved_prediction_field": sum(1 for row in vals if row["has_saved_prediction_field"]),
                "raw_prediction_rows": sum(int(row["raw_prediction_rows"]) for row in vals),
                "explicit_marker_rows": sum(int(row["explicit_marker_rows"]) for row in vals),
                "fallback_letter_rows": sum(int(row["fallback_letter_rows"]) for row in vals),
                "unanswered_rows": sum(int(row["unanswered_rows"]) for row in vals),
            }
        )
    return out


def model_run_metadata_rows(input_dir: Path, leaderboard: list[dict[str, str]]) -> list[dict[str, Any]]:
    evidence_path = input_dir / "evidence_audit.json"
    evidence = read_json(evidence_path) if evidence_path.exists() else {}
    field_by_model = {
        row.get("model", ""): set(row.get("fields", []))
        for row in evidence.get("per_model", [])
    }
    output = []
    for row in sorted(leaderboard, key=lambda item: (item["family"], item["model"])):
        fields = field_by_model.get(row["model"], set())
        output.append(
            {
                "model": row["model"],
                "family": row["family"],
                "n": int(f(row.get("n", 0))),
                "captured_item_id": "yes" if "id" in fields else "no",
                "captured_category": "yes" if "category" in fields else "no",
                "captured_ground_truth": "yes" if "ground_truth" in fields else "no",
                "captured_raw_response": "yes" if "response" in fields else "no",
                "captured_saved_prediction": "yes" if "prediction" in fields else "no",
                "captured_gpu_worker_id": "yes" if "gpu" in fields else "no",
                "captured_hf_repo": "no",
                "captured_revision_hash": "no",
                "captured_prompt_template": "no",
                "captured_chat_template": "no",
                "captured_decoding_parameters": "no",
                "captured_runtime_versions": "no",
                "captured_run_date": "no",
                "metadata_completeness": "incomplete",
                "notes": "Raw run rows did not preserve model repository, revision, prompt, decoding, package, or run-date metadata.",
            }
        )
    return output


def model_run_metadata_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        ("Item id", "captured_item_id"),
        ("Category", "captured_category"),
        ("Ground-truth label", "captured_ground_truth"),
        ("Raw response", "captured_raw_response"),
        ("Saved prediction", "captured_saved_prediction"),
        ("GPU worker id", "captured_gpu_worker_id"),
        ("HF repository", "captured_hf_repo"),
        ("Model revision hash", "captured_revision_hash"),
        ("Prompt template", "captured_prompt_template"),
        ("Chat template", "captured_chat_template"),
        ("Decoding parameters", "captured_decoding_parameters"),
        ("Runtime/package versions", "captured_runtime_versions"),
        ("Run date", "captured_run_date"),
    ]
    n_models = len(rows)
    return [
        {
            "metadata_field": label,
            "models_with_field": sum(1 for row in rows if row[key] == "yes"),
            "model_count": n_models,
            "status": "complete" if n_models and all(row[key] == "yes" for row in rows) else "incomplete",
        }
        for label, key in fields
    ]


def top_category_regression_rows(category_rows: list[dict[str, Any]], top_k: int = 3) -> list[dict[str, Any]]:
    out = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in category_rows:
        grouped[str(row["family"])].append(row)
    for family, vals in sorted(grouped.items()):
        ranked = sorted(vals, key=lambda row: f(row["mean_regression_mass"]), reverse=True)[:top_k]
        for rank, row in enumerate(ranked, start=1):
            out.append(
                {
                    "family": family,
                    "rank": rank,
                    "category": row["category"],
                    "pair_count": row["pair_count"],
                    "mean_accuracy_delta": row["mean_accuracy_delta"],
                    "mean_improvement_mass": row["mean_improvement_mass"],
                    "mean_regression_mass": row["mean_regression_mass"],
                }
            )
    return out


def quality_gate_sensitivity(
    all_rows: list[dict[str, Any]],
    leaderboard: list[dict[str, str]],
    thresholds: Iterable[float | None] = (None, 0.70, 0.80, 0.90),
) -> list[dict[str, Any]]:
    out = []
    primary_families = {"qwen", "gemma"}
    for threshold in thresholds:
        if threshold is None:
            eligible_models = {row["model"] for row in leaderboard}
            gate_label = "none"
        else:
            eligible_models = {
                row["model"]
                for row in leaderboard
                if f(row.get("answered_rate", 0.0)) >= threshold
            }
            gate_label = f"{threshold:.2f}"

        excluded = sorted(
            row["model"]
            for row in leaderboard
            if row["family"] in primary_families and row["model"] not in eligible_models
        )
        kept = [
            row
            for row in all_rows
            if row["family"] in primary_families
            and row["old_model"] in eligible_models
            and row["new_model"] in eligible_models
        ]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in kept:
            grouped[row["family"]].append(row)

        sensitivity_row: dict[str, Any] = {
            "answerability_gate": gate_label,
            "excluded_primary_family_models": "; ".join(excluded) if excluded else "none",
        }
        for family in ["qwen", "gemma"]:
            vals = grouped.get(family, [])
            sensitivity_row[f"{family}_pair_count"] = len(vals)
            sensitivity_row[f"{family}_mean_error_persistence"] = round(
                mean(numeric_values(vals, "error_persistence")), 6
            ) if numeric_values(vals, "error_persistence") else 0.0
            sensitivity_row[f"{family}_mean_regression_gap"] = round(
                mean(numeric_values(vals, "regression_gap")), 6
            ) if numeric_values(vals, "regression_gap") else 0.0
            sensitivity_row[f"{family}_mean_normalized_regression_burden"] = sensitivity_row[
                f"{family}_mean_regression_gap"
            ]
            sensitivity_row[f"{family}_mean_churn_mass"] = round(
                mean([f(row["churn_mass"]) for row in vals]), 6
            ) if vals else 0.0
        out.append(sensitivity_row)
    return out


def write_latex_tables(output_dir: Path, summaries: dict[str, Any]) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lrrrrrrrrrrr}",
        "\\toprule",
        "Family & Models & Pairs & Old acc. & New acc. & $\\Delta$ & $I$ & $R$ & Churn & Err. persist. & Corr. rate & NRB \\\\",
        "\\midrule",
    ]
    for row in summaries["family_summary"]:
        lines.append(
            f"{row['family'].title()} & {row['model_count_in_pairs']} & {row['pair_count']} & "
            f"{pct(row['mean_old_accuracy'])} & {pct(row['mean_new_accuracy'])} & "
            f"{pct(row['mean_accuracy_delta'])} & {pct(row['mean_improvement_mass'])} & "
            f"{pct(row['mean_regression_mass'])} & {pct(row['mean_churn_mass'])} & "
            f"{pct(row['mean_error_persistence'])} & {pct(row['mean_correction_rate'])} & "
            f"{pct(row['mean_regression_gap'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (tables_dir / "family_summary.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Family & Metric & Pairs & Pairwise $r$ & 95\\% CI & LOO range \\\\",
        "\\midrule",
    ]
    metric_names = {
        "churn_mass": "Churn mass",
        "error_persistence": "Persistence",
        "regression_gap": "Norm. burden",
    }
    for row in summaries["leave_one_model_out_correlations"]:
        if row["metric"] not in metric_names:
            continue
        lines.append(
            f"{row['family'].title()} & {metric_names[row['metric']]} & {row['pair_count']} & "
            f"{row['full_pairwise_r']:.3f} & "
            f"[{row['fisher_ci95_low']:.3f}, {row['fisher_ci95_high']:.3f}] & "
            f"[{row['leave_one_model_out_min_r']:.3f}, {row['leave_one_model_out_max_r']:.3f}] \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (tables_dir / "correlation_stability.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        "\\begin{tabular}{lp{0.24\\linewidth}rrrrrrrr}",
        "\\toprule",
        "Gate & Excluded models & Qwen pairs & Gemma pairs & Qwen persist. & Gemma persist. & Qwen NRB & Gemma NRB & Qwen churn & Gemma churn \\\\",
        "\\midrule",
    ]
    for row in summaries["quality_gate_sensitivity"]:
        gate = "none" if row["answerability_gate"] == "none" else f"{100 * f(row['answerability_gate']):.0f}\\%"
        lines.append(
            f"{gate} & {row['excluded_primary_family_models']} & "
            f"{row['qwen_pair_count']} & {row['gemma_pair_count']} & "
            f"{pct(row['qwen_mean_error_persistence'])} & "
            f"{pct(row['gemma_mean_error_persistence'])} & "
            f"{pct(row['qwen_mean_regression_gap'])} & "
            f"{pct(row['gemma_mean_regression_gap'])} & "
            f"{pct(row['qwen_mean_churn_mass'])} & "
            f"{pct(row['gemma_mean_churn_mass'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (tables_dir / "quality_gate_sensitivity.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Family & Models & Rows & Saved pred. & Explicit marker & Fallback & Unanswered & Saved-field models \\\\",
        "\\midrule",
    ]
    for row in summaries["extraction_coverage_by_family"]:
        lines.append(
            f"{row['family'].title()} & {row['model_count']} & {row['row_count']} & "
            f"{row['raw_prediction_rows']} & {row['explicit_marker_rows']} & "
            f"{row['fallback_letter_rows']} & {row['unanswered_rows']} & "
            f"{row['models_with_saved_prediction_field']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (tables_dir / "extraction_coverage.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Family & Category & Pairs & $\\Delta$ & $I$ & $R$ \\\\",
        "\\midrule",
    ]
    for row in summaries["top_category_regressions"]:
        lines.append(
            f"{row['family'].title()} & {row['category']} & {row['pair_count']} & "
            f"{pct(row['mean_accuracy_delta'])} & "
            f"{pct(row['mean_improvement_mass'])} & "
            f"{pct(row['mean_regression_mass'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (tables_dir / "top_category_regressions.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Metadata field & Captured for all models? & Models with field & Total models & Complete & Missing & Note \\\\",
        "\\midrule",
    ]
    for row in summaries["model_run_metadata_summary"]:
        complete = int(row["models_with_field"])
        total = int(row["model_count"])
        status = "yes" if complete == total and total else "no"
        note = "captured" if status == "yes" else "missing/incomplete"
        lines.append(
            f"{latex_escape(row['metadata_field'])} & {status} & {complete} & {total} & "
            f"{complete} & {total - complete} & {note} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (tables_dir / "model_run_metadata_status.tex").write_text("\n".join(lines), encoding="utf-8")

    def pairwise_table_lines(rows: list[dict[str, Any]]) -> list[str]:
        lines = [
            "\\begin{tabular}{lllrrrrrrrl}",
            "\\toprule",
            "Family & Directed pair & Path note & $\\Delta$ & $I$ & $R$ & Churn & Persist. & Corr. & NRB & Review \\\\",
            "\\midrule",
        ]
        for row in rows:
            pair = f"{latex_escape(row['old_model'])} $\\rightarrow$ {latex_escape(row['new_model'])}"
            lines.append(
                f"{latex_escape(row['family'].title())} & {pair} & "
                f"{latex_escape(row['path_interpretation'])} & "
                f"{pct(row['accuracy_delta'])} & "
                f"{pct(row['improvement_mass'])} & "
                f"{pct(row['regression_mass'])} & "
                f"{pct(row['churn_mass'])} & "
                f"{pct(row['error_persistence'])} & "
                f"{pct(row['correction_rate'])} & "
                f"{pct(row['normalized_regression_burden'])} & "
                f"{latex_escape(row['review_flag'])} \\\\"
            )
        lines.extend(["\\bottomrule", "\\end{tabular}", ""])
        return lines

    lines = pairwise_table_lines(summaries["enriched_pairwise_rows"])
    (tables_dir / "all_pairwise_comparisons.tex").write_text("\n".join(lines), encoding="utf-8")
    for family in sorted({row["family"] for row in summaries["enriched_pairwise_rows"]}):
        family_rows = [row for row in summaries["enriched_pairwise_rows"] if row["family"] == family]
        (tables_dir / f"all_pairwise_comparisons_{family}.tex").write_text(
            "\n".join(pairwise_table_lines(family_rows)),
            encoding="utf-8",
        )

def analyze(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    pairwise_csv = input_dir / "standardized_pairwise_error_ecology_all_models.csv"
    pairwise_json = input_dir / "standardized_pairwise_error_ecology_all_models.json"
    leaderboard = read_csv(input_dir / "leaderboard.csv")
    min_answered_rate = 0.80
    eligible_models = {
        row["model"]
        for row in leaderboard
        if f(row.get("answered_rate", 0.0)) >= min_answered_rate
    }
    excluded_models = [
        {
            "model": row["model"],
            "family": row["family"],
            "answered_rate": f(row.get("answered_rate", 0.0)),
            "parse_failure_rate": f(row.get("parse_failure_rate", 0.0)),
            "reason": f"answered_rate below primary-analysis threshold {min_answered_rate}",
        }
        for row in leaderboard
        if row["model"] not in eligible_models
    ]
    all_rows = enrich_pairwise(read_csv(pairwise_csv))
    rows = [
        row
        for row in all_rows
        if row["old_model"] in eligible_models and row["new_model"] in eligible_models
    ]
    pairwise_payload = read_json(pairwise_json)
    primary_full_comparisons = [
        row
        for row in pairwise_payload["full_comparisons"]
        if row["old_model"] in eligible_models and row["new_model"] in eligible_models
    ]
    model_persistence = model_level_metric(rows, "error_persistence")
    model_gap = model_level_metric(rows, "regression_gap")
    extraction_rows = extraction_coverage(input_dir)
    run_metadata_rows = model_run_metadata_rows(input_dir, leaderboard)
    category_rows = category_summary(primary_full_comparisons)
    summaries = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_pairwise_csv": str(pairwise_csv),
        "primary_min_answered_rate": min_answered_rate,
        "eligible_models": sorted(eligible_models),
        "excluded_models": excluded_models,
        "all_pair_count_before_quality_gate": len(all_rows),
        "pair_count": len(rows),
        "families": sorted({row["family"] for row in rows}),
        "family_summary": family_summary(rows),
        "successful_upgrade_summary": successful_upgrade_summary(rows),
        "near_parity_comparisons_abs_delta_le_0_05": near_parity_rows(rows),
        "model_level_error_persistence": model_persistence,
        "model_level_regression_gap": model_gap,
        "quality_gate_sensitivity": quality_gate_sensitivity(all_rows, leaderboard),
        "dependency_aware_family_tests": [
            exact_family_permutation_p(model_persistence, "mean_error_persistence_across_incident_pairs", "qwen", "gemma"),
            exact_family_permutation_p(model_gap, "mean_regression_gap_across_incident_pairs", "qwen", "gemma"),
        ],
        "leave_one_model_out_correlations": leave_one_model_out_correlations(rows),
        "category_summary": category_rows,
        "top_category_regressions": top_category_regression_rows(category_rows),
        "extraction_coverage": extraction_rows,
        "extraction_coverage_by_family": extraction_coverage_by_family(extraction_rows),
        "model_run_metadata": run_metadata_rows,
        "model_run_metadata_summary": model_run_metadata_summary(run_metadata_rows),
        "enriched_pairwise_rows": rows,
        "all_enriched_pairwise_rows_before_quality_gate": all_rows,
        "notes": [
            "Pairwise comparisons are not independent because each model appears in multiple pairs.",
            "Dependency-aware family tests aggregate metrics at the model level, then use exact label permutation across Qwen and Gemma models.",
            "Phi-2 is excluded from pairwise family analysis because only one Phi model is available.",
            f"Primary pairwise analysis excludes models with answered_rate < {min_answered_rate}.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "paper_metrics.json", summaries)
    write_csv(output_dir / "enriched_pairwise_metrics.csv", rows)
    write_csv(output_dir / "enriched_pairwise_metrics_all_before_quality_gate.csv", all_rows)
    write_csv(output_dir / "excluded_models_quality_gate.csv", excluded_models)
    write_csv(output_dir / "family_summary.csv", summaries["family_summary"])
    write_csv(output_dir / "successful_upgrade_summary.csv", summaries["successful_upgrade_summary"])
    write_csv(output_dir / "near_parity_comparisons.csv", summaries["near_parity_comparisons_abs_delta_le_0_05"])
    write_csv(output_dir / "model_level_error_persistence.csv", model_persistence)
    write_csv(output_dir / "model_level_regression_gap.csv", model_gap)
    write_csv(output_dir / "quality_gate_sensitivity.csv", summaries["quality_gate_sensitivity"])
    write_csv(output_dir / "dependency_aware_family_tests.csv", summaries["dependency_aware_family_tests"])
    write_csv(output_dir / "leave_one_model_out_correlations.csv", summaries["leave_one_model_out_correlations"])
    write_csv(output_dir / "category_summary.csv", summaries["category_summary"])
    write_csv(output_dir / "top_category_regressions.csv", summaries["top_category_regressions"])
    write_csv(output_dir / "extraction_coverage.csv", summaries["extraction_coverage"])
    write_csv(output_dir / "extraction_coverage_by_family.csv", summaries["extraction_coverage_by_family"])
    write_csv(output_dir / "model_run_metadata.csv", summaries["model_run_metadata"])
    write_csv(output_dir / "model_run_metadata_summary.csv", summaries["model_run_metadata_summary"])
    write_latex_tables(output_dir, summaries)
    return summaries


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready metrics from Boundary-SLM pairwise outputs.")
    parser.add_argument("--input-dir", type=Path, default=Path("main/analysis/raw_tpu_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("main/analysis/paper_metrics"))
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    summaries = analyze(args.input_dir, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "pair_count": summaries["pair_count"],
                "families": summaries["families"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
