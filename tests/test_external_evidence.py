from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from boundary_slm.external_evidence_registry import generate as generate_registry
from boundary_slm.external_evidence_registry import validate_sources
from boundary_slm.external_wild_ingest import generate as generate_wild
from boundary_slm.external_wild_ingest import normalize_records, pairwise_metrics
from boundary_slm.open_llm_details_ingest import generate as generate_open_llm_details


def test_registry_rejects_aggregate_churn_claim() -> None:
    rows = [
        {
            "source_id": "bad",
            "source_name": "Bad aggregate",
            "source_url": "https://example.com",
            "license": "unknown",
            "access_date": "2026-06-18",
            "artifact_type": "aggregate_scores",
            "evidence_tier": "C",
            "claim_use": "item_level_churn",
            "access_status": "public",
            "compatibility_status": "aggregate_only",
        }
    ]
    try:
        validate_sources(rows)
    except ValueError as exc:
        assert "aggregate-only" in str(exc)
    else:
        raise AssertionError("aggregate-only source should not be allowed to support churn")


def test_registry_generation_writes_manifest(tmp_path: Path) -> None:
    report = generate_registry(tmp_path / "external")
    assert report["public_first_source"] == "wild"
    assert (tmp_path / "external" / "external_evidence_map.csv").exists()
    assert (tmp_path / "external" / "tables" / "external_claim_use_table.tex").exists()
    rows = json.loads((tmp_path / "external" / "external_evidence_map.json").read_text())
    informal = [row for row in rows if row.get("source_quality") == "informal"]
    assert informal
    assert all(row["claim_use"] == "context_only" for row in informal)
    assert all(row["excluded_from_statistical_tests"] for row in informal)


def test_wild_normalization_preserves_schema() -> None:
    frame = pd.DataFrame(
        [
            {
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "task": "mmlu_pro",
                "subtask": "math",
                "item_id": "abc123",
                "score": 1,
                "input_tokens": 10,
                "output_tokens": 2,
            }
        ]
    )
    rows = normalize_records(frame)
    assert rows[0]["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert rows[0]["task"] == "mmlu_pro"
    assert rows[0]["subtask"] == "math"
    assert rows[0]["item_id"] == "abc123"
    assert rows[0]["score"] == 1
    assert rows[0]["is_correct"] is True


def test_wild_pairwise_metrics_match_toy_counts() -> None:
    records = []
    old_scores = {"i1": 1, "i2": 0, "i3": 1, "i4": 0}
    new_scores = {"i1": 1, "i2": 1, "i3": 0, "i4": 0}
    for model, scores in [
        ("Qwen/Qwen2.5-0.5B-Instruct", old_scores),
        ("Qwen/Qwen2.5-1.5B-Instruct", new_scores),
    ]:
        for item_id, score in scores.items():
            records.append(
                {
                    "source_id": "wild",
                    "model": model,
                    "family": "qwen",
                    "generation": "qwen2.5",
                    "parameter_b": 0.5 if "0.5B" in model else 1.5,
                    "boundary_role": "strict_sub_4b",
                    "task": "mmlu_pro",
                    "subtask": "math",
                    "item_id": item_id,
                    "id": f"mmlu_pro:math:{item_id}",
                    "category": "mmlu_pro",
                    "score": score,
                    "is_correct": bool(score),
                    "input_tokens": 1,
                    "output_tokens": 1,
                }
            )
    rows = pairwise_metrics(records, bootstrap_iters=10)
    main = [row for row in rows if row["task_scope"] == "__all_selected__"][0]
    assert main["n_common"] == 4
    assert main["persistent_correct_count"] == 1
    assert main["persistent_error_count"] == 1
    assert main["improvement_count"] == 1
    assert main["regression_count"] == 1
    assert main["churn_mass"] == 0.5
    assert main["error_persistence"] == 0.5
    assert main["correction_rate"] == 0.5
    assert main["normalized_regression_burden"] == 0.5


def test_wild_pairwise_zero_old_errors_marks_old_error_ratios_na() -> None:
    records = []
    old_scores = {"i1": 1, "i2": 1}
    new_scores = {"i1": 1, "i2": 0}
    for model, scores in [
        ("Qwen/Qwen2.5-0.5B-Instruct", old_scores),
        ("Qwen/Qwen2.5-1.5B-Instruct", new_scores),
    ]:
        for item_id, score in scores.items():
            records.append(
                {
                    "source_id": "wild",
                    "model": model,
                    "family": "qwen",
                    "generation": "qwen2.5",
                    "parameter_b": 0.5 if "0.5B" in model else 1.5,
                    "boundary_role": "strict_sub_4b",
                    "task": "mmlu_pro",
                    "subtask": "math",
                    "item_id": item_id,
                    "id": f"mmlu_pro:math:{item_id}",
                    "category": "mmlu_pro",
                    "score": score,
                    "is_correct": bool(score),
                    "input_tokens": 1,
                    "output_tokens": 1,
                }
            )
    main = [row for row in pairwise_metrics(records, bootstrap_iters=0) if row["task_scope"] == "__all_selected__"][0]
    assert main["old_error_count"] == 0
    assert main["error_persistence"] is None
    assert main["correction_rate"] is None
    assert main["normalized_regression_burden"] is None


def test_wild_smoke_ingest_from_local_parquet(tmp_path: Path) -> None:
    source = tmp_path / "wild.parquet"
    rows = []
    for item_idx in range(6):
        for model, offset in [
            ("Qwen/Qwen2.5-0.5B-Instruct", 0),
            ("Qwen/Qwen2.5-1.5B-Instruct", 1),
        ]:
            rows.append(
                {
                    "model": model,
                    "task": "mmlu_pro",
                    "subtask": "math",
                    "item_id": f"item-{item_idx}",
                    "score": (item_idx + offset) % 2,
                    "input_tokens": 10,
                    "output_tokens": 1,
                }
            )
    pd.DataFrame(rows).to_parquet(source, index=False)
    report = generate_wild(
        input_uri=str(source),
        output_dir=tmp_path / "out",
        models=["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"],
        tasks=["mmlu_pro"],
        bootstrap_iters=10,
    )
    assert report["claim_ready"]
    assert report["pairwise_comparison_count"] == 2
    assert report["all_selected_pair_count"] == 1
    assert report["task_level_pair_count"] == 1
    assert (tmp_path / "out" / "wild_normalized_records.parquet").exists()
    assert (tmp_path / "out" / "wild_pairwise_replacement_metrics.csv").exists()
    assert (tmp_path / "out" / "wild_task_dispersion_summary.csv").exists()
    assert (tmp_path / "out" / "tables" / "wild_task_dispersion.tex").exists()
    pair_table = (tmp_path / "out" / "tables" / "wild_all_selected_pairs.tex").read_text(encoding="utf-8")
    dispersion_table = (tmp_path / "out" / "tables" / "wild_task_dispersion.tex").read_text(encoding="utf-8")
    assert "$N$" in pair_table
    assert "Median $N$" in dispersion_table


def test_open_llm_details_stub_writes_exclusions(tmp_path: Path) -> None:
    report = generate_open_llm_details(
        tmp_path / "open_llm",
        try_download=False,
        targets=[{"model": "x", "repo_id": "open-llm-leaderboard/x-details"}],
    )
    assert report["exclusion_count"] == 1
    exclusions = json.loads((tmp_path / "open_llm" / "open_llm_details_exclusions.json").read_text())
    assert exclusions[0]["status"] == "not_downloaded"
