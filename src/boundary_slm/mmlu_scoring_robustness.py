from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from boundary_slm.io import read_jsonl, write_csv, write_json
from boundary_slm.raw_results import (
    extract_answer_from_raw,
    infer_model_meta,
    model_order_key,
    raw_item_id,
    safe_ratio,
)
from boundary_slm.scoring_mode_sensitivity import has_saved_prediction, raw_response_only_prediction


SCORING_MODES = [
    "default",
    "raw_only",
    "exclude_saved_prediction_models",
    "exclude_low_confidence_fallback",
    "explicit_marker_only",
]
EXPLICIT_OR_SAVED_METHODS = {
    "raw_prediction",
    "single_letter",
    "final_answer",
    "correct_answer",
    "therefore_answer",
    "answer_is",
    "option_is",
}
LOW_CONFIDENCE_METHODS = {"tail_option", "last_standalone_letter", "none", "empty", "prompt_echo_without_completion"}


def ratio_or_none(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = row.get(key)
        if value in {None, ""}:
            continue
        try:
            out.append(float(value))
        except Exception:
            continue
    return out


def avg(rows: list[dict[str, Any]], key: str) -> float | None:
    values = numeric_values(rows, key)
    return round(mean(values), 6) if values else None


def pct(value: Any, *, signed: bool = False) -> str:
    if value in {None, ""}:
        return "--"
    sign = "+" if signed else ""
    try:
        return f"{100.0 * float(value):{sign}.1f}"
    except Exception:
        return "--"


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


def load_rows_by_model(input_dir: Path) -> dict[str, list[dict[str, Any]]]:
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(input_dir.glob("*.jsonl")):
        rows = read_jsonl(path)
        if not rows:
            continue
        model = str(rows[0].get("model") or rows[0].get("model_label") or path.stem)
        rows_by_model[model] = rows
    return rows_by_model


def score_row(raw: dict[str, Any], *, mode: str) -> dict[str, Any] | None:
    if mode == "raw_only":
        prediction, method, confidence = raw_response_only_prediction(raw)
    else:
        prediction, method, confidence = extract_answer_from_raw(raw)

    if mode == "exclude_low_confidence_fallback" and (method in LOW_CONFIDENCE_METHODS or confidence < 0.62):
        return None
    if mode == "explicit_marker_only" and method not in EXPLICIT_OR_SAVED_METHODS:
        return None

    truth = str(raw.get("ground_truth", raw.get("expected", ""))).strip().upper()
    return {
        "id": raw_item_id(raw),
        "ground_truth": truth,
        "prediction": prediction or "",
        "answered": prediction is not None,
        "is_correct": prediction == truth,
        "extraction_method": method,
        "extraction_confidence": confidence,
    }


def scored_models_for_mode(rows_by_model: dict[str, list[dict[str, Any]]], mode: str) -> dict[str, list[dict[str, Any]]]:
    scored: dict[str, list[dict[str, Any]]] = {}
    for model, rows in rows_by_model.items():
        if mode == "exclude_saved_prediction_models" and any(has_saved_prediction(row) for row in rows):
            continue
        scored_rows = [score_row(row, mode=mode) for row in rows]
        kept = [row for row in scored_rows if row is not None]
        if kept:
            scored[model] = kept
    return scored


def compare_scored(
    old_model: str,
    old_rows: list[dict[str, Any]],
    new_model: str,
    new_rows: list[dict[str, Any]],
    *,
    mode: str,
) -> dict[str, Any] | None:
    old_by_id = {str(row["id"]): row for row in old_rows}
    new_by_id = {str(row["id"]): row for row in new_rows}
    common = sorted(set(old_by_id) & set(new_by_id))
    if not common:
        return None
    old_errors = {item_id for item_id in common if not old_by_id[item_id]["is_correct"]}
    new_errors = {item_id for item_id in common if not new_by_id[item_id]["is_correct"]}
    improvements = {
        item_id for item_id in common if not old_by_id[item_id]["is_correct"] and new_by_id[item_id]["is_correct"]
    }
    regressions = {
        item_id for item_id in common if old_by_id[item_id]["is_correct"] and not new_by_id[item_id]["is_correct"]
    }
    persistent_errors = old_errors & new_errors
    old_acc = safe_ratio(len(common) - len(old_errors), len(common))
    new_acc = safe_ratio(len(common) - len(new_errors), len(common))
    old_error_rate = safe_ratio(len(old_errors), len(common))
    regression_mass = safe_ratio(len(regressions), len(common))
    return {
        "mode": mode,
        "family": infer_model_meta(old_model).family,
        "old_model": old_model,
        "new_model": new_model,
        "comparison_id": f"{old_model} -> {new_model}",
        "n_common": len(common),
        "old_error_count": len(old_errors),
        "old_accuracy": old_acc,
        "new_accuracy": new_acc,
        "accuracy_delta": round(new_acc - old_acc, 6),
        "improvement_mass": safe_ratio(len(improvements), len(common)),
        "regression_mass": regression_mass,
        "churn_mass": safe_ratio(len(improvements) + len(regressions), len(common)),
        "error_persistence": ratio_or_none(len(persistent_errors), len(old_errors)),
        "correction_rate": ratio_or_none(len(improvements), len(old_errors)),
        "normalized_regression_burden": ratio_or_none(regression_mass, old_error_rate),
    }


def pairwise_for_mode(scored_by_model: dict[str, list[dict[str, Any]]], mode: str) -> list[dict[str, Any]]:
    models = sorted(scored_by_model, key=lambda label: model_order_key(infer_model_meta(label)))
    out: list[dict[str, Any]] = []
    for old_idx, old_model in enumerate(models):
        for new_model in models[old_idx + 1 :]:
            if infer_model_meta(old_model).family != infer_model_meta(new_model).family:
                continue
            row = compare_scored(old_model, scored_by_model[old_model], new_model, scored_by_model[new_model], mode=mode)
            if row is not None:
                out.append(row)
    return out


def mode_family_summary(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[(str(row["mode"]), str(row["family"]))].append(row)
    out: list[dict[str, Any]] = []
    for (mode, family), rows in sorted(grouped.items()):
        out.append(
            {
                "mode": mode,
                "family": family,
                "pair_count": len(rows),
                "median_n_common": int(median([int(row["n_common"]) for row in rows])) if rows else 0,
                "mean_accuracy_delta": avg(rows, "accuracy_delta"),
                "mean_improvement_mass": avg(rows, "improvement_mass"),
                "mean_regression_mass": avg(rows, "regression_mass"),
                "mean_churn_mass": avg(rows, "churn_mass"),
                "mean_error_persistence": avg(rows, "error_persistence"),
                "mean_correction_rate": avg(rows, "correction_rate"),
                "mean_normalized_regression_burden": avg(rows, "normalized_regression_burden"),
            }
        )
    return out


def family_contrast_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_mode: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        by_mode[str(row["mode"])][str(row["family"])] = row
    out: list[dict[str, Any]] = []
    for mode, families in sorted(by_mode.items()):
        qwen = families.get("qwen")
        gemma = families.get("gemma")
        if not qwen or not gemma:
            out.append(
                {
                    "mode": mode,
                    "has_qwen_gemma_contrast": False,
                    "reason": "missing qwen or gemma family rows after this robustness filter",
                }
            )
            continue
        qwen_p = qwen.get("mean_error_persistence")
        gemma_p = gemma.get("mean_error_persistence")
        qwen_nrb = qwen.get("mean_normalized_regression_burden")
        gemma_nrb = gemma.get("mean_normalized_regression_burden")
        out.append(
            {
                "mode": mode,
                "has_qwen_gemma_contrast": True,
                "qwen_pair_count": qwen["pair_count"],
                "gemma_pair_count": gemma["pair_count"],
                "qwen_mean_error_persistence": qwen_p,
                "gemma_mean_error_persistence": gemma_p,
                "qwen_minus_gemma_error_persistence": round(float(qwen_p) - float(gemma_p), 6)
                if qwen_p is not None and gemma_p is not None
                else None,
                "qwen_mean_nrb": qwen_nrb,
                "gemma_mean_nrb": gemma_nrb,
                "qwen_minus_gemma_nrb": round(float(qwen_nrb) - float(gemma_nrb), 6)
                if qwen_nrb is not None and gemma_nrb is not None
                else None,
            }
        )
    return out


def load_parser_gate(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"claim_ready": False, "status": "missing_parser_audit_claim_gate", "parser_validated": False}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"claim_ready": False, "status": "unreadable_parser_audit_claim_gate", "parser_validated": False}


def robustness_gate(contrast_rows: list[dict[str, Any]], parser_gate: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    parser_validated = bool(parser_gate.get("claim_ready") or parser_gate.get("parser_validated"))
    if not parser_validated:
        blockers.append("manual parser audit is incomplete or below threshold")
    primary = next((row for row in contrast_rows if row.get("mode") == "default" and row.get("has_qwen_gemma_contrast")), None)
    material_change = False
    if not primary:
        blockers.append("default qwen/gemma family contrast is unavailable")
    else:
        primary_diff = primary.get("qwen_minus_gemma_error_persistence")
        for row in contrast_rows:
            if not row.get("has_qwen_gemma_contrast") or row.get("mode") == "default":
                continue
            diff = row.get("qwen_minus_gemma_error_persistence")
            if diff is None or primary_diff is None:
                continue
            if (float(primary_diff) < 0 <= float(diff)) or (float(primary_diff) > 0 >= float(diff)):
                material_change = True
        if material_change:
            blockers.append("qwen/gemma persistence contrast changes direction under at least one robustness mode")
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mmlu_pro_confirmatory": not blockers,
        "claim_ready": not blockers,
        "status": "ready" if not blockers else "blocked",
        "parser_validated": parser_validated,
        "family_contrast_material_change": material_change,
        "blockers": blockers,
    }


def write_summary_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Mode & Family & Pairs & Mean $N$ & $\Delta$ & $R$ & Churn & Persist. \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(row['mode'])} & {latex_escape(row['family'])} & {row['pair_count']} & "
            f"{row['median_n_common']} & {pct(row['mean_accuracy_delta'], signed=True)} & "
            f"{pct(row['mean_regression_mass'])} & {pct(row['mean_churn_mass'])} & "
            f"{pct(row['mean_error_persistence'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_contrast_tex(path: Path, rows: list[dict[str, Any]], gate: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Mode & Qwen pairs & Gemma pairs & Qwen persist. & Gemma persist. & Difference \\",
        r"\midrule",
    ]
    for row in rows:
        if not row.get("has_qwen_gemma_contrast"):
            lines.append(f"{latex_escape(row['mode'])} & 0 & 0 & -- & -- & -- \\\\")
            continue
        lines.append(
            f"{latex_escape(row['mode'])} & {row['qwen_pair_count']} & {row['gemma_pair_count']} & "
            f"{pct(row['qwen_mean_error_persistence'])} & {pct(row['gemma_mean_error_persistence'])} & "
            f"{pct(row['qwen_minus_gemma_error_persistence'], signed=True)} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            f"Claim gate & \\multicolumn{{5}}{{l}}{{{latex_escape(gate['status'])}: {latex_escape('; '.join(gate['blockers']) or 'ready')}}} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def generate(input_dir: Path, output_dir: Path, parser_gate_path: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_model = load_rows_by_model(input_dir)
    all_pair_rows: list[dict[str, Any]] = []
    mode_model_counts: list[dict[str, Any]] = []
    for mode in SCORING_MODES:
        scored = scored_models_for_mode(rows_by_model, mode)
        all_pair_rows.extend(pairwise_for_mode(scored, mode))
        mode_model_counts.append(
            {
                "mode": mode,
                "model_count": len(scored),
                "scored_row_count": sum(len(rows) for rows in scored.values()),
            }
        )
    summary_rows = mode_family_summary(all_pair_rows)
    contrasts = family_contrast_rows(summary_rows)
    parser_gate = load_parser_gate(parser_gate_path)
    gate = robustness_gate(contrasts, parser_gate)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "parser_gate_path": str(parser_gate_path),
        "mode_model_counts": mode_model_counts,
        "mmlu_scoring_robustness": summary_rows,
        "family_contrast_robustness": contrasts,
        "mmlu_claim_gate": gate,
        "claim_boundary": "MMLU-Pro remains a parser-dependent archived-output diagnostic unless parser and scoring robustness gates pass.",
    }
    write_csv(output_dir / "mmlu_scoring_robustness.csv", summary_rows)
    write_json(output_dir / "mmlu_scoring_robustness.json", report)
    write_csv(output_dir / "family_contrast_robustness.csv", contrasts)
    write_json(output_dir / "family_contrast_robustness.json", contrasts)
    write_json(output_dir / "mmlu_claim_gate.json", gate)
    write_summary_tex(output_dir / "mmlu_scoring_robustness.tex", summary_rows)
    write_contrast_tex(output_dir / "family_contrast_robustness.tex", contrasts, gate)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MMLU-Pro scoring robustness gates over archived raw-output logs.")
    parser.add_argument("--input-dir", type=Path, default=Path("main/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("main/analysis/parser_audit"))
    parser.add_argument("--parser-gate-path", type=Path, default=Path("main/analysis/parser_audit/parser_audit_claim_gate.json"))
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    report = generate(args.input_dir, args.output_dir, args.parser_gate_path)
    print(json.dumps(report["mmlu_claim_gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
