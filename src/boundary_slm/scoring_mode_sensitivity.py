from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from boundary_slm.io import read_jsonl, write_csv, write_json
from boundary_slm.raw_results import (
    extract_answer,
    extract_answer_from_raw,
    infer_model_meta,
    looks_like_prompt_echo_without_completion,
    model_order_key,
    raw_item_id,
    safe_ratio,
)


VALID_OPTIONS = set("ABCDEFGHIJ")


def ratio_or_none(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def numeric_delta(new_value: Any, old_value: Any) -> float | None:
    if new_value in {None, ""} or old_value in {None, ""}:
        return None
    return round(float(new_value) - float(old_value), 6)


def pct(value: Any, *, signed: bool = False) -> str:
    if value in {None, ""}:
        return "--"
    sign = "+" if signed else ""
    return f"{100.0 * float(value):{sign}.1f}"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def has_saved_prediction(row: dict[str, Any]) -> bool:
    prediction = str(row.get("prediction", "")).strip().upper()
    return len(prediction) == 1 and prediction in VALID_OPTIONS


def raw_response_only_prediction(row: dict[str, Any]) -> tuple[str | None, str, float]:
    response = str(row.get("response", row.get("response_text", "")))
    if looks_like_prompt_echo_without_completion(response):
        return None, "prompt_echo_without_completion", 0.0
    return extract_answer(response)


def score_rows(rows: list[dict[str, Any]], *, mode: str) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        if mode == "saved_prediction_primary":
            prediction, method, confidence = extract_answer_from_raw(row)
        elif mode == "raw_response_only":
            prediction, method, confidence = raw_response_only_prediction(row)
        else:
            raise ValueError(f"Unknown scoring mode: {mode}")
        truth = str(row.get("ground_truth", row.get("expected", ""))).strip().upper()
        scored.append(
            {
                "id": raw_item_id(row),
                "ground_truth": truth,
                "prediction": prediction or "",
                "answered": prediction is not None,
                "is_correct": prediction == truth,
                "extraction_method": method,
                "extraction_confidence": confidence,
            }
        )
    return scored


def summarize_model(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    saved_scored = score_rows(rows, mode="saved_prediction_primary")
    raw_scored = score_rows(rows, mode="raw_response_only")
    meta = infer_model_meta(model)
    by_id_saved = {row["id"]: row for row in saved_scored}
    by_id_raw = {row["id"]: row for row in raw_scored}
    common = sorted(set(by_id_saved) & set(by_id_raw))
    saved_prediction_rows = sum(1 for row in rows if has_saved_prediction(row))
    comparable_prediction_rows = 0
    agreement_rows = 0
    disagreement_rows = 0
    raw_unparseable_saved_rows = 0
    saved_correct_only = 0
    raw_correct_only = 0
    same_correctness = 0
    changed_correctness = 0
    for raw_row in rows:
        if not has_saved_prediction(raw_row):
            continue
        item_id = raw_item_id(raw_row)
        saved_prediction = str(raw_row.get("prediction", "")).strip().upper()
        raw_prediction = by_id_raw[item_id]["prediction"]
        if raw_prediction:
            comparable_prediction_rows += 1
            if raw_prediction == saved_prediction:
                agreement_rows += 1
            else:
                disagreement_rows += 1
        else:
            raw_unparseable_saved_rows += 1
    for item_id in common:
        saved_correct = bool(by_id_saved[item_id]["is_correct"])
        raw_correct = bool(by_id_raw[item_id]["is_correct"])
        if saved_correct == raw_correct:
            same_correctness += 1
        else:
            changed_correctness += 1
            if saved_correct:
                saved_correct_only += 1
            if raw_correct:
                raw_correct_only += 1
    n = len(common)
    saved_correct_count = sum(1 for row in saved_scored if row["is_correct"])
    raw_correct_count = sum(1 for row in raw_scored if row["is_correct"])
    saved_answered = sum(1 for row in saved_scored if row["answered"])
    raw_answered = sum(1 for row in raw_scored if row["answered"])
    return {
        "model": model,
        "family": meta.family,
        "n": n,
        "saved_prediction_rows": saved_prediction_rows,
        "saved_prediction_row_rate": safe_ratio(saved_prediction_rows, len(rows)),
        "saved_primary_accuracy": safe_ratio(saved_correct_count, len(saved_scored)),
        "raw_response_only_accuracy": safe_ratio(raw_correct_count, len(raw_scored)),
        "accuracy_delta_raw_minus_saved": round(
            safe_ratio(raw_correct_count, len(raw_scored)) - safe_ratio(saved_correct_count, len(saved_scored)),
            6,
        ),
        "saved_primary_answered_rate": safe_ratio(saved_answered, len(saved_scored)),
        "raw_response_only_answered_rate": safe_ratio(raw_answered, len(raw_scored)),
        "prediction_comparable_rows": comparable_prediction_rows,
        "prediction_agreement_rows": agreement_rows,
        "prediction_disagreement_rows": disagreement_rows,
        "raw_unparseable_saved_rows": raw_unparseable_saved_rows,
        "agreement_rate_on_comparable_predictions": safe_ratio(agreement_rows, comparable_prediction_rows),
        "same_correctness_rows": same_correctness,
        "changed_correctness_rows": changed_correctness,
        "changed_correctness_rate": safe_ratio(changed_correctness, n),
        "saved_primary_correct_only_rows": saved_correct_only,
        "raw_response_only_correct_only_rows": raw_correct_only,
    }


def compare_scored_models(
    old_model: str,
    old_rows: list[dict[str, Any]],
    new_model: str,
    new_rows: list[dict[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    old_by_id = {row["id"]: row for row in score_rows(old_rows, mode=mode)}
    new_by_id = {row["id"]: row for row in score_rows(new_rows, mode=mode)}
    common = sorted(set(old_by_id) & set(new_by_id))
    old_errors = [item_id for item_id in common if not old_by_id[item_id]["is_correct"]]
    regressions = [
        item_id
        for item_id in common
        if old_by_id[item_id]["is_correct"] and not new_by_id[item_id]["is_correct"]
    ]
    improvements = [
        item_id
        for item_id in common
        if not old_by_id[item_id]["is_correct"] and new_by_id[item_id]["is_correct"]
    ]
    persistent_errors = [
        item_id
        for item_id in common
        if not old_by_id[item_id]["is_correct"] and not new_by_id[item_id]["is_correct"]
    ]
    old_acc = safe_ratio(sum(1 for item_id in common if old_by_id[item_id]["is_correct"]), len(common))
    new_acc = safe_ratio(sum(1 for item_id in common if new_by_id[item_id]["is_correct"]), len(common))
    old_error_rate = safe_ratio(len(old_errors), len(common))
    regression_mass = safe_ratio(len(regressions), len(common))
    return {
        "mode": mode,
        "family": infer_model_meta(old_model).family,
        "old_model": old_model,
        "new_model": new_model,
        "comparison_id": f"{old_model} -> {new_model}",
        "n_common": len(common),
        "old_accuracy": old_acc,
        "new_accuracy": new_acc,
        "accuracy_delta": round(new_acc - old_acc, 6),
        "improvement_mass": safe_ratio(len(improvements), len(common)),
        "regression_mass": regression_mass,
        "churn_mass": safe_ratio(len(improvements) + len(regressions), len(common)),
        "error_persistence": ratio_or_none(len(persistent_errors), len(old_errors)),
        "normalized_regression_burden": ratio_or_none(regression_mass, old_error_rate),
    }


def pairwise_mode_rows(rows_by_model: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    models = sorted(
        rows_by_model,
        key=lambda label: model_order_key(infer_model_meta(label)),
    )
    out: list[dict[str, Any]] = []
    for old_index, old_model in enumerate(models):
        for new_model in models[old_index + 1 :]:
            if infer_model_meta(old_model).family != infer_model_meta(new_model).family:
                continue
            for mode in ["saved_prediction_primary", "raw_response_only"]:
                out.append(
                    compare_scored_models(
                        old_model,
                        rows_by_model[old_model],
                        new_model,
                        rows_by_model[new_model],
                        mode=mode,
                    )
                )
    return out


def pairwise_mode_delta_rows(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in pair_rows:
        by_pair[row["comparison_id"]][row["mode"]] = row
    out: list[dict[str, Any]] = []
    metrics = [
        "old_accuracy",
        "new_accuracy",
        "accuracy_delta",
        "improvement_mass",
        "regression_mass",
        "churn_mass",
        "error_persistence",
        "normalized_regression_burden",
    ]
    for comparison_id, modes in sorted(by_pair.items()):
        if "saved_prediction_primary" not in modes or "raw_response_only" not in modes:
            continue
        saved = modes["saved_prediction_primary"]
        raw = modes["raw_response_only"]
        row = {
            "comparison_id": comparison_id,
            "family": saved["family"],
            "old_model": saved["old_model"],
            "new_model": saved["new_model"],
            "n_common": saved["n_common"],
        }
        for metric in metrics:
            row[f"saved_{metric}"] = saved[metric]
            row[f"raw_{metric}"] = raw[metric]
            row[f"raw_minus_saved_{metric}"] = numeric_delta(raw[metric], saved[metric])
        out.append(row)
    return out


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


def write_latex_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Model & Saved acc. & Raw-only acc. & $\\Delta$ & Saved answered & Raw answered & Changed corr. \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(row['model'])} & "
            f"{pct(row['saved_primary_accuracy'])} & "
            f"{pct(row['raw_response_only_accuracy'])} & "
            f"{pct(row['accuracy_delta_raw_minus_saved'], signed=True)} & "
            f"{pct(row['saved_primary_answered_rate'])} & "
            f"{pct(row['raw_response_only_answered_rate'])} & "
            f"{pct(row['changed_correctness_rate'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def generate(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(input_dir.glob("*.jsonl")):
        rows = read_jsonl(path)
        if not rows:
            continue
        if not any(has_saved_prediction(row) for row in rows):
            continue
        model = str(rows[0].get("model") or rows[0].get("model_label") or path.stem)
        rows_by_model[model] = rows

    model_rows = [summarize_model(model, rows) for model, rows in sorted(rows_by_model.items())]
    pair_rows = pairwise_mode_rows(rows_by_model)
    pair_delta_rows = pairwise_mode_delta_rows(pair_rows)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "models_with_saved_prediction_fields": sorted(rows_by_model),
        "model_count": len(rows_by_model),
        "model_sensitivity": model_rows,
        "pairwise_mode_metrics": pair_rows,
        "pairwise_mode_delta": pair_delta_rows,
        "claim_use": "Scoring-mode robustness only; not a substitute for a completed manual parser audit.",
        "manual_audit_gate": "Strong MMLU-Pro raw-output claims remain gated until human parser-audit labels are completed.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "scoring_mode_sensitivity.csv", model_rows)
    write_csv(output_dir / "scoring_mode_pairwise_metrics.csv", pair_rows)
    write_csv(output_dir / "scoring_mode_pairwise_delta.csv", pair_delta_rows)
    write_json(output_dir / "scoring_mode_sensitivity.json", report)
    write_latex_table(output_dir / "scoring_mode_sensitivity.tex", model_rows)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare saved-prediction scoring with raw-response-only parser scoring.")
    parser.add_argument("--input-dir", type=Path, default=Path("main/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("main/analysis/parser_audit"))
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    report = generate(args.input_dir, args.output_dir)
    print(json.dumps({"model_count": report["model_count"], "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
