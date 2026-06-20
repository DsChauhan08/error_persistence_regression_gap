from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boundary_slm.artifact_manifest import build_manifest
from boundary_slm.artifact_release_status import generate as generate_release_status
from boundary_slm.cpu_mmlu_pro_rerun import run as run_cpu_rerun
from boundary_slm.external_benchmark_context import generate as generate_external_context, validate_rows
from boundary_slm.mmlu_scoring_robustness import generate as generate_mmlu_robustness
from boundary_slm.mmlu_pro_manifest import (
    SourceRow,
    build_manifest_rows,
    generate as generate_mmlu_manifest,
    source_row_hash,
)
from boundary_slm.parser_audit_impact import generate as generate_parser_impact
from boundary_slm.parser_audit_labeler import apply_label, next_indices, progress as label_progress
from boundary_slm.parser_audit_label_server import complete_html, row_html
from boundary_slm.parser_audit_report import manual_claim_gate, summarize_manual_audit
from boundary_slm.parser_audit import generate as generate_parser_audit
from boundary_slm.parser_audit_second_pass import generate as generate_second_pass_audit
from boundary_slm.parser_audit_workbook import generate as generate_parser_workbook
from boundary_slm.public_release_hygiene import scan_release
from boundary_slm.scoring_mode_sensitivity import generate as generate_scoring_sensitivity


def write_items_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_source_parquet(path: Path, count: int = 5) -> None:
    import pandas as pd

    rows = []
    for idx in range(1, count + 1):
        rows.append(
            {
                "question_id": idx,
                "question": f"Question {idx}?",
                "options": [f"choice {letter}" for letter in "ABCDEFGHIJ"],
                "answer": "A" if idx % 2 else "B",
                "answer_index": 0 if idx % 2 else 1,
                "cot_content": "",
                "category": "math" if idx % 2 else "physics",
                "src": "unit-test",
            }
        )
    pd.DataFrame(rows).to_parquet(path)


def test_mmlu_manifest_matches_toy_rows_and_hashes_are_redacted() -> None:
    source = {
        "1": SourceRow("1", "Secret question text", ["secret A", "secret B"], "A", "math", "toy"),
    }
    current = {
        "1": {
            "id": "1",
            "categories": {"math": 1},
            "ground_truths": {"A": 1},
            "models": {"m1"},
            "row_count": 1,
        }
    }
    rows = build_manifest_rows(current, source, source_repo="repo", source_revision="rev")
    assert rows[0]["status"] == "matched"
    assert rows[0]["question_sha256"]
    assert "Secret question text" not in json.dumps(rows[0])
    assert source_row_hash(source["1"]) == source_row_hash(source["1"])


def test_mmlu_manifest_detects_answer_mismatch() -> None:
    source = {"1": SourceRow("1", "Q", ["A"], "B", "math", "toy")}
    current = {
        "1": {
            "id": "1",
            "categories": {"math": 1},
            "ground_truths": {"A": 1},
            "models": {"m1"},
            "row_count": 1,
        }
    }
    rows = build_manifest_rows(current, source, source_repo="repo", source_revision="rev")
    assert rows[0]["status"] == "mismatch"
    assert "Ground-truth answer differs" in rows[0]["notes"]


def test_mmlu_manifest_command_with_local_parquet(tmp_path: Path) -> None:
    per_model = tmp_path / "per_model"
    write_items_csv(
        per_model / "model_items.csv",
        [
            {
                "model": "m",
                "id": "1",
                "category": "math",
                "ground_truth": "A",
                "prediction": "A",
                "is_correct": "True",
                "answered": "True",
                "extraction_method": "final_answer",
            }
        ],
    )
    parquet = tmp_path / "source.parquet"
    write_source_parquet(parquet, count=1)
    report = generate_mmlu_manifest(input_dir=per_model, output_dir=tmp_path / "out", parquet_path=parquet)
    assert report["claim_ready"]
    assert (tmp_path / "out" / "mmlu_pro_source_manifest.csv").exists()


def test_parser_audit_report_computes_false_rates() -> None:
    rows = [
        {
            "family": "qwen",
            "extraction_method": "final_answer",
            "parser_answered": "true",
            "parser_prediction": "A",
            "ground_truth": "A",
            "human_answered": "true",
            "human_prediction": "A",
            "human_parser_correct": "true",
        },
        {
            "family": "qwen",
            "extraction_method": "tail_option",
            "parser_answered": "true",
            "parser_prediction": "B",
            "ground_truth": "B",
            "human_answered": "true",
            "human_prediction": "C",
            "human_parser_correct": "false",
        },
        {
            "family": "qwen",
            "extraction_method": "none",
            "parser_answered": "false",
            "parser_prediction": "",
            "ground_truth": "D",
            "human_answered": "true",
            "human_prediction": "D",
            "human_parser_correct": "false",
        },
    ]
    summary = summarize_manual_audit(rows)
    by_method = {row["extraction_method"]: row for row in summary}
    assert by_method["tail_option"]["false_extraction_rows"] == 1
    assert by_method["none"]["false_unanswered_rows"] == 1
    gate = manual_claim_gate(summary, min_rows=3, min_agreement=0.5)
    assert gate["completed_manual_rows"] == 3
    assert gate["correctness_changed_by_human_label_rows"] == 2


def test_parser_audit_impact_counts_false_regression_and_improvement(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    rows = [
        {
            "model": "Qwen2-0.5B-Instruct",
            "family": "qwen",
            "item_id": "i1",
            "category": "math",
            "ground_truth": "A",
            "parser_prediction": "A",
            "parser_answered": "true",
            "extraction_method": "final_answer",
            "extraction_confidence": "0.9",
            "audit_source": "high_risk",
            "risk_reason": "toy",
            "human_prediction": "A",
            "human_answered": "true",
            "human_parser_correct": "true",
            "human_notes": "",
        },
        {
            "model": "Qwen2.5-0.5B-Instruct",
            "family": "qwen",
            "item_id": "i1",
            "category": "math",
            "ground_truth": "A",
            "parser_prediction": "B",
            "parser_answered": "true",
            "extraction_method": "tail_option",
            "extraction_confidence": "0.62",
            "audit_source": "high_risk",
            "risk_reason": "toy",
            "human_prediction": "A",
            "human_answered": "true",
            "human_parser_correct": "false",
            "human_notes": "",
        },
        {
            "model": "Qwen2-0.5B-Instruct",
            "family": "qwen",
            "item_id": "i2",
            "category": "math",
            "ground_truth": "C",
            "parser_prediction": "D",
            "parser_answered": "true",
            "extraction_method": "tail_option",
            "extraction_confidence": "0.62",
            "audit_source": "stratified",
            "risk_reason": "toy",
            "human_prediction": "C",
            "human_answered": "true",
            "human_parser_correct": "false",
            "human_notes": "",
        },
        {
            "model": "Qwen2.5-0.5B-Instruct",
            "family": "qwen",
            "item_id": "i2",
            "category": "math",
            "ground_truth": "C",
            "parser_prediction": "C",
            "parser_answered": "true",
            "extraction_method": "final_answer",
            "extraction_confidence": "0.9",
            "audit_source": "stratified",
            "risk_reason": "toy",
            "human_prediction": "D",
            "human_answered": "true",
            "human_parser_correct": "false",
            "human_notes": "",
        },
    ]
    write_items_csv(sample, rows)
    report = generate_parser_impact(
        sample_csv=sample,
        output_dir=tmp_path / "out",
        required_rows=4,
        required_high_risk_rows=2,
        min_agreement=0.1,
    )
    gate = report["claim_gate"]
    assert gate["false_regression_impact_cases"] == 1
    assert gate["false_improvement_impact_cases"] == 1
    assert not gate["claim_ready"]
    assert (tmp_path / "out" / "false_regression_improvement_impact.csv").exists()


def test_parser_audit_second_pass_reports_consistency_and_keeps_private_sample(tmp_path: Path) -> None:
    sample = tmp_path / "parser_audit_sample.csv"
    rows = [
        {
            "model": "Qwen2-0.5B-Instruct",
            "family": "qwen",
            "item_id": "i1",
            "category": "math",
            "ground_truth": "A",
            "parser_prediction": "A",
            "parser_answered": "true",
            "extraction_method": "final_answer",
            "extraction_confidence": "0.9",
            "response_excerpt": "private raw tail",
            "audit_source": "high_risk",
            "risk_reason": "toy",
            "human_prediction": "A",
            "human_answered": "true",
            "human_parser_correct": "true",
            "human_notes": "",
        },
        {
            "model": "Qwen2.5-0.5B-Instruct",
            "family": "qwen",
            "item_id": "i2",
            "category": "math",
            "ground_truth": "B",
            "parser_prediction": "C",
            "parser_answered": "true",
            "extraction_method": "tail_option",
            "extraction_confidence": "0.62",
            "response_excerpt": "private raw tail",
            "audit_source": "stratified",
            "risk_reason": "",
            "human_prediction": "B",
            "human_answered": "true",
            "human_parser_correct": "false",
            "human_notes": "",
        },
    ]
    write_items_csv(sample, rows)
    out = tmp_path / "audit"
    first_report = generate_second_pass_audit(sample_csv=sample, output_dir=out, sample_size=2, seed=1)
    second_sample = out / "second_pass_parser_audit_sample.csv"
    assert first_report["status"] == "pending_labels"
    assert second_sample.exists()
    private_rows = list(csv.DictReader(second_sample.open(newline="", encoding="utf-8")))
    assert "response_excerpt" in private_rows[0]
    assert "human_prediction" not in private_rows[0]

    private_rows[0]["second_human_prediction"] = "A"
    private_rows[0]["second_human_answered"] = "true"
    private_rows[0]["second_human_parser_correct"] = "true"
    private_rows[1]["second_human_prediction"] = "B"
    private_rows[1]["second_human_answered"] = "true"
    private_rows[1]["second_human_parser_correct"] = "false"
    write_items_csv(second_sample, private_rows)
    report = generate_second_pass_audit(sample_csv=sample, output_dir=out, sample_size=2, seed=1)
    assert report["status"] == "ready"
    assert report["consistency"]["completed_second_pass_rows"] == 2
    assert report["consistency"]["prediction_agreement_rate"] == 1.0
    assert (out / "second_pass_parser_audit_public_summary.json").exists()


def test_parser_audit_workbook_writes_private_batches_and_redacted_progress(tmp_path: Path) -> None:
    sample = tmp_path / "parser_audit_sample.csv"
    rows = [
        {
            "model": "Qwen2-0.5B-Instruct",
            "family": "qwen",
            "item_id": f"i{idx}",
            "category": "math",
            "ground_truth": "A",
            "parser_prediction": "A",
            "parser_answered": "true",
            "extraction_method": "final_answer",
            "extraction_confidence": "0.9",
            "response_excerpt": "private raw tail",
            "audit_source": "high_risk" if idx == 1 else "stratified",
            "risk_reason": "toy" if idx == 1 else "",
            "human_prediction": "A" if idx == 1 else "",
            "human_answered": "true" if idx == 1 else "",
            "human_parser_correct": "true" if idx == 1 else "",
            "human_notes": "",
        }
        for idx in range(1, 4)
    ]
    write_items_csv(sample, rows)
    report = generate_parser_workbook(sample_csv=sample, output_dir=tmp_path / "audit", batch_size=2)
    assert report["progress"]["total_rows"] == 3
    assert report["progress"]["completed_rows"] == 1
    assert report["progress"]["completed_high_risk_rows"] == 1
    assert len(report["batch_manifest"]) == 2
    batch = tmp_path / "audit" / "labeling_batches" / "parser_audit_batch_001.csv"
    assert batch.exists()
    assert "private raw tail" in batch.read_text(encoding="utf-8")
    public_report = json.loads((tmp_path / "audit" / "parser_audit_labeling_progress.json").read_text(encoding="utf-8"))
    assert "private raw tail" not in json.dumps(public_report)


def test_parser_audit_includes_high_risk_saved_raw_disagreement(tmp_path: Path) -> None:
    input_dir = tmp_path / "results"
    input_dir.mkdir()
    rows = [
        {
            "id": "1",
            "category": "math",
            "ground_truth": "A",
            "model": "Gemma-3-1B-Instruct",
            "prediction": "A",
            "response": "Final answer: B",
        },
        {
            "id": "2",
            "category": "math",
            "ground_truth": "B",
            "model": "Gemma-3-1B-Instruct",
            "response": "The answer is B.",
        },
    ]
    (input_dir / "gemma.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = generate_parser_audit(input_dir, tmp_path / "audit", per_stratum=0, high_risk_rows=5, seed=1)
    sample = list(csv.DictReader((tmp_path / "audit" / "parser_audit_sample.csv").open(newline="", encoding="utf-8")))
    assert report["high_risk_sampled_rows"] == 1
    assert sample[0]["audit_source"] == "high_risk"
    assert "saved_raw_disagreement" in sample[0]["risk_reason"]


def test_scoring_mode_sensitivity_compares_saved_and_raw_modes(tmp_path: Path) -> None:
    input_dir = tmp_path / "results"
    input_dir.mkdir()
    rows_a = [
        {
            "id": "1",
            "category": "math",
            "ground_truth": "A",
            "model": "Gemma-3-270M-Instruct",
            "prediction": "A",
            "response": "Final answer: B",
        },
        {
            "id": "2",
            "category": "math",
            "ground_truth": "B",
            "model": "Gemma-3-270M-Instruct",
            "prediction": "B",
            "response": "Final answer: B",
        },
    ]
    rows_b = [
        {
            "id": "1",
            "category": "math",
            "ground_truth": "A",
            "model": "Gemma-3-1B-Instruct",
            "prediction": "A",
            "response": "Final answer: A",
        },
        {
            "id": "2",
            "category": "math",
            "ground_truth": "B",
            "model": "Gemma-3-1B-Instruct",
            "prediction": "A",
            "response": "Final answer: B",
        },
    ]
    for name, rows in [("a.jsonl", rows_a), ("b.jsonl", rows_b)]:
        (input_dir / name).write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = generate_scoring_sensitivity(input_dir, tmp_path / "out")
    assert report["model_count"] == 2
    model_rows = {row["model"]: row for row in report["model_sensitivity"]}
    assert model_rows["Gemma-3-270M-Instruct"]["changed_correctness_rows"] == 1
    assert (tmp_path / "out" / "scoring_mode_sensitivity.csv").exists()
    assert (tmp_path / "out" / "scoring_mode_pairwise_delta.csv").exists()
    assert report["pairwise_mode_delta"][0]["raw_minus_saved_accuracy_delta"] != 0


def test_mmlu_scoring_robustness_blocks_without_parser_gate(tmp_path: Path) -> None:
    input_dir = tmp_path / "results"
    input_dir.mkdir()
    rows_a = [
        {"id": "1", "ground_truth": "A", "model": "Qwen2-0.5B-Instruct", "response": "Final answer: A"},
        {"id": "2", "ground_truth": "B", "model": "Qwen2-0.5B-Instruct", "response": "Final answer: A"},
    ]
    rows_b = [
        {"id": "1", "ground_truth": "A", "model": "Qwen2.5-0.5B-Instruct", "response": "Final answer: B"},
        {"id": "2", "ground_truth": "B", "model": "Qwen2.5-0.5B-Instruct", "response": "Final answer: B"},
    ]
    for name, rows in [("a.jsonl", rows_a), ("b.jsonl", rows_b)]:
        (input_dir / name).write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = generate_mmlu_robustness(
        input_dir=input_dir,
        output_dir=tmp_path / "robust",
        parser_gate_path=tmp_path / "missing_gate.json",
    )
    assert report["mmlu_claim_gate"]["status"] == "blocked"
    assert not report["mmlu_claim_gate"]["mmlu_pro_confirmatory"]
    assert (tmp_path / "robust" / "mmlu_scoring_robustness.csv").exists()


def test_cpu_rerun_mock_writes_outputs_and_resumes(tmp_path: Path) -> None:
    per_model = tmp_path / "per_model"
    rows = []
    for idx in range(1, 6):
        rows.append(
            {
                "model": "m",
                "id": str(idx),
                "category": "math" if idx % 2 else "physics",
                "ground_truth": "A" if idx % 2 else "B",
                "prediction": "A",
                "is_correct": "True",
                "answered": "True",
                "extraction_method": "final_answer",
            }
        )
    write_items_csv(per_model / "model_items.csv", rows)
    parquet = tmp_path / "source.parquet"
    write_source_parquet(parquet, count=5)
    output = tmp_path / "cpu"
    claim = run_cpu_rerun(
        output_dir=output,
        per_model_dir=per_model,
        models=["mock-a", "mock-b"],
        backend="mock",
        sample_size=5,
        stress_size=2,
        seed=1,
        max_new_tokens=4,
        parquet_path=parquet,
    )
    first_count = len((output / "records.jsonl").read_text(encoding="utf-8").strip().splitlines())
    rerun_claim = run_cpu_rerun(
        output_dir=output,
        per_model_dir=per_model,
        models=["mock-a", "mock-b"],
        backend="mock",
        sample_size=5,
        stress_size=2,
        seed=1,
        max_new_tokens=4,
        parquet_path=parquet,
    )
    second_count = len((output / "records.jsonl").read_text(encoding="utf-8").strip().splitlines())
    assert claim["claim_ready"]
    assert rerun_claim["claim_ready"]
    assert first_count == second_count == 10
    for name in [
        "records.jsonl",
        "run_manifest.json",
        "environment.json",
        "selected_item_manifest.csv",
        "summary.csv",
        "pairwise_cpu_audit.csv",
        "claim_check.json",
    ]:
        assert (output / name).exists(), name
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["prompt_template"]
    assert manifest["prompt_template_sha256"]
    assert manifest["selected_item_manifest"] == "selected_item_manifest.csv"
    assert manifest["completed_models"] == ["mock-a", "mock-b"]
    assert all(row["loader_metadata"]["chat_template_sha256"] == "mock" for row in manifest["models"])
    item_manifest = list(csv.DictReader((output / "selected_item_manifest.csv").open(newline="", encoding="utf-8")))
    assert len(item_manifest) == 5
    assert {"question_sha256", "options_sha256", "answer_sha256", "item_sha256"} <= set(item_manifest[0])


def test_external_context_rejects_missing_source_url(tmp_path: Path) -> None:
    bad = [
        {
            "model_or_family": "x",
            "benchmark": "MMLU",
            "setting": "5-shot",
            "source_url": "",
            "paper_use": "context_only",
            "caveat": "none",
        }
    ]
    try:
        validate_rows(bad)
    except ValueError as exc:
        assert "source_url" in str(exc)
    else:
        raise AssertionError("validate_rows should reject missing source_url")
    report = generate_external_context(tmp_path / "external")
    assert report["paper_use"] == "context_only"
    assert (tmp_path / "external" / "informal_benchmark_context.csv").exists()
    informal = list(csv.DictReader((tmp_path / "external" / "informal_benchmark_context.csv").open(newline="", encoding="utf-8")))
    assert informal
    assert all(row["claim_use"] == "context_only" for row in informal)
    tex = (tmp_path / "external" / "tables" / "external_benchmark_context.tex").read_text(encoding="utf-8")
    assert "URL" in tex
    assert "Access" in tex


def test_public_release_hygiene_flags_response_tail_column(tmp_path: Path) -> None:
    release = tmp_path / "release"
    leak = release / "analysis" / "raw_tpu_results" / "per_model"
    leak.mkdir(parents=True)
    (leak / "model_items.csv").write_text("id,response_tail\n1,secret text\n", encoding="utf-8")
    report = scan_release(release)
    assert not report["passed"]
    assert any("response_tail" in finding["issue"] for finding in report["findings"])


def test_public_release_hygiene_flags_unredacted_response_excerpt_anywhere(tmp_path: Path) -> None:
    release = tmp_path / "release"
    leak = release / "analysis" / "parser_audit"
    leak.mkdir(parents=True)
    (leak / "custom_public_summary.csv").write_text("id,response_excerpt\n1,secret raw text\n", encoding="utf-8")
    report = scan_release(release)
    assert not report["passed"]
    assert any("response_excerpt" in finding["issue"] for finding in report["findings"])


def test_public_release_hygiene_flags_labeling_batches_path(tmp_path: Path) -> None:
    release = tmp_path / "release"
    leak = release / "analysis" / "parser_audit" / "labeling_batches"
    leak.mkdir(parents=True)
    (leak / "parser_audit_batch_001.csv").write_text("id,human_prediction\n1,A\n", encoding="utf-8")
    report = scan_release(release)
    assert not report["passed"]
    assert any("labeling_batches" in finding["issue"] or "forbidden raw-output path component" in finding["issue"] for finding in report["findings"])


def test_artifact_manifest_excludes_private_parser_sample(tmp_path: Path) -> None:
    sample = tmp_path / "main" / "analysis" / "parser_audit" / "parser_audit_sample.csv"
    sample.parent.mkdir(parents=True)
    sample.write_text("response_excerpt\nsecret raw response\n", encoding="utf-8")
    second_sample = sample.parent / "second_pass_parser_audit_sample.csv"
    second_sample.write_text("response_excerpt\nsecret raw response\n", encoding="utf-8")
    batch = sample.parent / "labeling_batches" / "parser_audit_batch_001.csv"
    batch.parent.mkdir(parents=True)
    batch.write_text("response_excerpt\nsecret raw response\n", encoding="utf-8")
    summary = sample.parent / "parser_audit_public_summary.json"
    summary.write_text("{}\n", encoding="utf-8")
    manifest = build_manifest(tmp_path, ["main/analysis/parser_audit"])
    paths = {row["path"] for row in manifest["files"]}
    assert "main/analysis/parser_audit/parser_audit_sample.csv" not in paths
    assert "main/analysis/parser_audit/second_pass_parser_audit_sample.csv" not in paths
    assert "main/analysis/parser_audit/labeling_batches/parser_audit_batch_001.csv" not in paths
    assert "main/analysis/parser_audit/parser_audit_public_summary.json" in paths


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_artifact_release_status_reports_github_ready_without_parser_validation(tmp_path: Path) -> None:
    out = tmp_path / "status"
    parser_gate = tmp_path / "parser_gate.json"
    mmlu_gate = tmp_path / "mmlu_gate.json"
    wild_gate = tmp_path / "wild_gate.json"
    source_manifest = tmp_path / "source_manifest.json"
    hygiene = tmp_path / "hygiene.json"
    public_manifest = tmp_path / "manifest.json"
    write_json(parser_gate, {"claim_ready": False, "parser_validated": False, "status": "blocked"})
    write_json(mmlu_gate, {"claim_ready": False, "mmlu_pro_confirmatory": False, "status": "blocked"})
    write_json(wild_gate, {"claim_ready": True, "pairwise_comparison_count": 44})
    write_json(
        source_manifest,
        {
            "claim_ready": True,
            "evaluated_item_count": 3008,
            "matched_item_count": 3008,
            "mismatch_item_count": 0,
            "missing_source_item_count": 0,
        },
    )
    write_json(hygiene, {"passed": True, "findings": []})
    write_json(public_manifest, {"file_count": 10, "missing": 0, "mismatches": 0})
    status = generate_release_status(
        root=tmp_path,
        output_dir=out,
        repository_url="https://github.com/DsChauhan08/error_persistence_regression_gap",
        latest_test_result="42 passed in 4.75s",
        parser_gate_path=parser_gate,
        mmlu_gate_path=mmlu_gate,
        wild_claim_path=wild_gate,
        source_manifest_path=source_manifest,
        hygiene_report_path=hygiene,
        public_manifest_path=public_manifest,
    )
    assert status["github_public_ready"]
    assert status["core_claim_ready"]
    assert not status["journal_ready"]
    assert not status["parser_validated"]
    assert not status["mmlu_pro_confirmatory"]
    assert not any("missing release metadata" in blocker for blocker in status["blockers"])
    assert (out / "artifact_release_status.json").exists()


def test_artifact_release_status_blocks_github_ready_without_wild_claim(tmp_path: Path) -> None:
    out = tmp_path / "status"
    parser_gate = tmp_path / "parser_gate.json"
    mmlu_gate = tmp_path / "mmlu_gate.json"
    wild_gate = tmp_path / "wild_gate.json"
    source_manifest = tmp_path / "source_manifest.json"
    hygiene = tmp_path / "hygiene.json"
    public_manifest = tmp_path / "manifest.json"
    write_json(parser_gate, {"claim_ready": False})
    write_json(mmlu_gate, {"claim_ready": False})
    write_json(wild_gate, {"claim_ready": False, "pairwise_comparison_count": 0})
    write_json(
        source_manifest,
        {
            "claim_ready": True,
            "evaluated_item_count": 3008,
            "matched_item_count": 3008,
            "mismatch_item_count": 0,
            "missing_source_item_count": 0,
        },
    )
    write_json(hygiene, {"passed": True, "findings": []})
    write_json(public_manifest, {"file_count": 10, "missing": 0, "mismatches": 0})
    status = generate_release_status(
        root=tmp_path,
        output_dir=out,
        repository_url="https://github.com/DsChauhan08/error_persistence_regression_gap",
        latest_test_result="42 passed in 4.75s",
        parser_gate_path=parser_gate,
        mmlu_gate_path=mmlu_gate,
        wild_claim_path=wild_gate,
        source_manifest_path=source_manifest,
        hygiene_report_path=hygiene,
        public_manifest_path=public_manifest,
    )
    assert not status["github_public_ready"]
    assert not status["core_claim_ready"]
    assert any("WILD item-level correctness" in blocker for blocker in status["blockers"])


def test_artifact_release_status_can_be_journal_ready_when_all_gates_pass(tmp_path: Path) -> None:
    out = tmp_path / "status"
    parser_gate = tmp_path / "parser_gate.json"
    mmlu_gate = tmp_path / "mmlu_gate.json"
    wild_gate = tmp_path / "wild_gate.json"
    source_manifest = tmp_path / "source_manifest.json"
    hygiene = tmp_path / "hygiene.json"
    public_manifest = tmp_path / "manifest.json"
    write_json(parser_gate, {"claim_ready": True, "parser_validated": True, "status": "ready"})
    write_json(mmlu_gate, {"claim_ready": True, "mmlu_pro_confirmatory": True, "status": "ready"})
    write_json(wild_gate, {"claim_ready": True, "pairwise_comparison_count": 44})
    write_json(
        source_manifest,
        {
            "claim_ready": True,
            "evaluated_item_count": 3008,
            "matched_item_count": 3008,
            "mismatch_item_count": 0,
            "missing_source_item_count": 0,
        },
    )
    write_json(hygiene, {"passed": True, "findings": []})
    write_json(public_manifest, {"file_count": 10, "missing": 0, "mismatches": 0})
    status = generate_release_status(
        root=tmp_path,
        output_dir=out,
        repository_url="https://github.com/DsChauhan08/error_persistence_regression_gap",
        latest_test_result="42 passed in 4.75s",
        parser_gate_path=parser_gate,
        mmlu_gate_path=mmlu_gate,
        wild_claim_path=wild_gate,
        source_manifest_path=source_manifest,
        hygiene_report_path=hygiene,
        public_manifest_path=public_manifest,
        archive_identifier="swh:1:dir:0123456789abcdef",
    )
    assert status["github_public_ready"]
    assert status["parser_validated"]
    assert status["mmlu_pro_confirmatory"]
    assert status["archive_ready"]
    assert status["journal_ready"]
    assert not status["blockers"]


def test_parser_audit_labeler_computes_parser_correct_and_progress() -> None:
    rows = [
        {
            "parser_prediction": "B",
            "parser_answered": "true",
            "audit_source": "high_risk",
            "human_prediction": "",
            "human_answered": "",
            "human_parser_correct": "",
        },
        {
            "parser_prediction": "C",
            "parser_answered": "true",
            "audit_source": "stratified",
            "human_prediction": "C",
            "human_answered": "true",
            "human_parser_correct": "true",
        },
    ]
    labeled_match = apply_label(rows[0], human_prediction="B", human_answered=True, human_notes="clear")
    assert labeled_match["human_parser_correct"] == "true"
    assert labeled_match["human_prediction"] == "B"
    labeled_miss = apply_label(rows[0], human_prediction="A", human_answered=True, human_notes="")
    assert labeled_miss["human_parser_correct"] == "false"
    rows[0] = labeled_match
    report = label_progress(rows)
    assert report["completed_rows"] == 2
    assert report["completed_high_risk_rows"] == 1
    assert next_indices(rows, high_risk_first=True, limit=None) == []


def test_parser_audit_label_server_renders_without_raw_html_execution() -> None:
    row = {
        "model": "Qwen-test",
        "item_id": "42",
        "category": "math",
        "ground_truth": "A",
        "parser_prediction": "B",
        "parser_answered": "true",
        "extraction_method": "answer_is",
        "extraction_confidence": "0.9",
        "audit_source": "high_risk",
        "risk_reason": "unit",
        "response_excerpt": "<script>alert('x')</script> final answer is B",
    }
    status = {
        "completed_rows": 1,
        "total_rows": 2,
        "completed_high_risk_rows": 1,
        "high_risk_rows": 1,
    }
    html = row_html(row, 0, 2, status)
    assert "Parser Audit Labeler" in html
    assert "&lt;script&gt;" in html
    assert "<script>alert" not in html
    done = complete_html(status)
    assert "Parser Audit Complete" in done
