from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boundary_slm.raw_results import (
    analyze_raw_results,
    build_raw_evidence_audit,
    extract_answer,
    extract_answer_from_raw,
    infer_model_meta,
)
from boundary_slm.replacement_audit_cards import generate_cards


def test_extract_answer_prefers_final_answer() -> None:
    text = "A. maybe\nB. maybe\nTherefore, the correct answer is: H. final."
    prediction, method, confidence = extract_answer(text)
    assert prediction == "H"
    assert method == "correct_answer"
    assert confidence >= 0.8


def test_extract_answer_handles_parenthesized_answer() -> None:
    prediction, method, _ = extract_answer("Therefore, the answer is (J).")
    assert prediction == "J"
    assert method == "answer_is"


def test_infer_model_meta_qwen35() -> None:
    meta = infer_model_meta("Qwen3.5-2B")
    assert meta.family == "qwen"
    assert meta.generation == "qwen3.5"
    assert meta.parameter_b == 2.0


def test_infer_model_meta_handles_gemma_million_and_phi() -> None:
    gemma = infer_model_meta("Gemma-3-270M-Instruct")
    phi = infer_model_meta("Phi-2")
    assert gemma.family == "gemma"
    assert gemma.generation == "gemma3"
    assert gemma.parameter_b == 0.27
    assert phi.family == "phi"
    assert phi.generation == "phi2"
    assert phi.parameter_b == 2.7


def test_extract_answer_from_raw_prefers_prediction_field() -> None:
    prediction, method, confidence = extract_answer_from_raw(
        {"prediction": "J", "response": "A. distractor B. distractor"}
    )
    assert prediction == "J"
    assert method == "raw_prediction"
    assert confidence == 1.0


def test_extract_answer_from_raw_rejects_prompt_echo() -> None:
    prediction, method, confidence = extract_answer_from_raw(
        {
            "response": (
                "Question: Which choice is correct?\n"
                "Options:\nA. first\nB. second\n"
                "Answer with only the final option letter.\n"
                "model"
            )
        }
    )
    assert prediction is None
    assert method == "prompt_echo_without_completion"
    assert confidence == 0.0


def test_analyze_raw_results_writes_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "results"
    input_dir.mkdir()
    rows_a = [
        {"id": 1, "category": "math", "ground_truth": "A", "model": "Qwen2-0.5B-Instruct", "response": "Final answer: A"},
        {"id": 2, "category": "physics", "ground_truth": "B", "model": "Qwen2-0.5B-Instruct", "response": "Final answer: C"},
    ]
    rows_b = [
        {"id": 1, "category": "math", "ground_truth": "A", "model": "Qwen2.5-0.5B-Instruct", "response": "Final answer: A"},
        {"id": 2, "category": "physics", "ground_truth": "B", "model": "Qwen2.5-0.5B-Instruct", "response": "Final answer: B"},
    ]
    for name, rows in [("a.jsonl", rows_a), ("b.jsonl", rows_b)]:
        (input_dir / name).write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    output_dir = tmp_path / "analysis"
    conclusions = analyze_raw_results(input_dir=input_dir, output_dir=output_dir, bootstrap_iters=50)
    assert (output_dir / "leaderboard.csv").exists()
    assert (output_dir / "paper_conclusions.json").exists()
    assert (output_dir / "evidence_audit.json").exists()
    assert conclusions["evidence_assessment"]["model_count"] == 2
    assert conclusions["evidence_audit"]["verdict"] == "not_provable_missing_condition_field"


def test_evidence_audit_detects_paired_conditions() -> None:
    audit = build_raw_evidence_audit(
        {
            "Qwen-test": [
                {"model": "Qwen-test", "item_id": "1", "condition": "baseline", "response_text": "A", "expected": "A"},
                {"model": "Qwen-test", "item_id": "1", "condition": "context_long_middle", "response_text": "B", "expected": "A"},
            ]
        }
    )
    assert audit["paper2_pairing_detected"] is True
    assert audit["pairs_with_multiple_conditions_same_model_item"] == 1


def test_replacement_audit_card_flags_near_parity_churn() -> None:
    rows = [
        {
            "comparison_id": "old -> new",
            "family": "qwen",
            "old_model": "old",
            "new_model": "new",
            "n_common": "100",
            "old_accuracy": "0.50",
            "new_accuracy": "0.52",
            "accuracy_delta": "0.02",
            "improvement_mass": "0.16",
            "regression_mass": "0.14",
            "churn_mass": "0.30",
            "error_persistence": "0.68",
            "correction_rate": "0.32",
            "normalized_regression_burden": "0.28",
            "net_gain_per_changed_item": "0.066667",
            "improvement_to_regression_ratio": "1.142857",
            "top_improving_categories": "math:+0.10",
            "top_regressing_categories": "history:-0.08",
        }
    ]
    cards = generate_cards(rows)
    assert cards[0]["status"] == "positive_delta_candidate"
    assert cards[0]["review_priority"] == "manual_review_required"
    assert "near_parity_churn" in cards[0]["risk_flags"]
    assert cards[0]["churn_mass_pct"] == 30.0
