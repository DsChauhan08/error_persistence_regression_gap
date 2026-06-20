from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boundary_slm.experiment import run_error_ecology, run_interface
from boundary_slm.interventions import InterventionSpec, apply_intervention
from boundary_slm.manifest import build_legacy_manifest
from boundary_slm.metrics import bootstrap_ci, error_ecology_stats, interface_stats
from boundary_slm.models import load_model_registry
from boundary_slm.scoring import score_response
from boundary_slm.tasks import EvalItem, build_task_items


def test_model_registry_smoke_scope() -> None:
    models = load_model_registry(smoke=True)
    labels = {model.label for model in models}
    assert "Qwen3.5-0.8B" in labels
    assert "Gemma-3-1B-It" in labels
    assert all(model.primary for model in models)


def test_task_generation_is_deterministic() -> None:
    first = build_task_items(smoke=True, seed=123)
    second = build_task_items(smoke=True, seed=123)
    assert [item.to_record() for item in first] == [item.to_record() for item in second]
    assert {item.task_family for item in first} == {"math", "multiple_choice", "instruction", "code"}


def test_scoring_number_choice_and_json() -> None:
    number = EvalItem("n", "t", "math", "p", "42", "number")
    choice = EvalItem("c", "t", "multiple_choice", "p", "B", "multiple_choice")
    json_item = EvalItem("j", "t", "instruction", "p", "COBALT", "json_value", metadata={"json_key": "selected", "allowed_keys": ["selected"]})
    assert score_response(number, "The answer is 42.")["is_correct"]
    assert score_response(choice, "B")["is_correct"]
    assert score_response(json_item, '{"selected":"COBALT"}')["is_correct"]
    assert not score_response(json_item, '{"selected":"COBALT","extra":1}')["format_ok"]


def test_intervention_adds_context_metadata() -> None:
    item = EvalItem("x", "t", "math", "What is 1+1?", "2", "number")
    prompt, metadata = apply_intervention(item, InterventionSpec("context_long_middle", "context"))
    assert "unrelated context" in prompt.lower()
    assert "What is 1+1?" in prompt
    assert metadata["context_length"] == "long"
    assert metadata["answer_position"] == "middle"


def test_bootstrap_ci_bounds_mean() -> None:
    low, high = bootstrap_ci([1.0, 1.0, 0.0, 0.0], iters=100, seed=1)
    assert low <= 0.5 <= high


def test_error_ecology_gate_detects_family_pass() -> None:
    records = []
    for family in ["qwen", "gemma"]:
        for item_id in ["a", "b", "c", "d"]:
            records.append({"condition": "baseline", "family": family, "parameter_b": 1.0, "generation": "old1", "item_id": item_id, "is_correct": item_id in {"a", "b"}})
            records.append({"condition": "baseline", "family": family, "parameter_b": 1.0, "generation": "new2", "item_id": item_id, "is_correct": item_id in {"a", "b", "c"}})
    stats = error_ecology_stats(records)
    assert stats["claim_check"]["pass_boolean"]


def test_interface_gate_detects_two_family_tax() -> None:
    records = []
    for family in ["qwen", "gemma"]:
        for model in [f"{family}-m"]:
            for idx in range(20):
                records.append({"family": family, "model_label": model, "item_id": str(idx), "condition": "baseline", "is_correct": True, "answered": True, "format_ok": True, "elapsed_seconds": 1.0, "tokens_per_second": 1.0})
                records.append({"family": family, "model_label": model, "item_id": str(idx), "condition": "context_long_middle", "is_correct": idx < 10, "answered": True, "format_ok": True, "elapsed_seconds": 1.0, "tokens_per_second": 1.0})
    stats = interface_stats(records, bootstrap_iters=100)
    assert stats["claim_check"]["pass_boolean"]


def test_mock_runs_write_required_outputs(tmp_path: Path) -> None:
    error_root = tmp_path / "error"
    interface_root = tmp_path / "interface"
    run_error_ecology(output_root=error_root, backend_name="mock", smoke=True)
    run_interface(output_root=interface_root, backend_name="mock", smoke=True)
    for root in [error_root, interface_root]:
        for name in ["run_manifest.json", "records.jsonl", "summary.csv", "stats.json", "claim_check.json", "environment.json"]:
            assert (root / name).exists(), name
        claim = json.loads((root / "claim_check.json").read_text(encoding="utf-8"))
        assert "pass_boolean" in claim


def test_manifest_builder_hashes_files(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "a.txt").write_text("hello", encoding="utf-8")
    manifest = build_legacy_manifest(legacy)
    assert manifest["file_count"] == 1
    assert manifest["files"][0]["path"] == "a.txt"
    assert manifest["files"][0]["sha256"]

