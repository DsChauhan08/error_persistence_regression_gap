from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from boundary_slm.io import write_csv, write_json


def current_access_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


DEFAULT_ROWS: list[dict[str, Any]] = [
    {
        "model_or_family": "google/gemma-2-2b-it",
        "reported_model_scope": "Gemma 2 PT 2B model-card aggregate",
        "benchmark": "MMLU",
        "setting": "5-shot, top-1",
        "reported_score": 51.3,
        "score_unit": "accuracy_percent",
        "source_url": "https://huggingface.co/google/gemma-2-2b-it",
        "paper_use": "context_only",
        "claim_use": "context_only",
        "source_type": "official_model_card",
        "source_quality": "official_aggregate",
        "evidence_tier": "C",
        "setting_complete": True,
        "model_revision_known": False,
        "excluded_from_statistical_tests": True,
        "caveat": "Model-card aggregate score; not paired item-level output from this study.",
    },
    {
        "model_or_family": "google/gemma-3-4b-it",
        "reported_model_scope": "Gemma 3 PT 4B model-card aggregate",
        "benchmark": "MMLU",
        "setting": "5-shot",
        "reported_score": 59.6,
        "score_unit": "accuracy_percent",
        "source_url": "https://huggingface.co/google/gemma-3-4b-it",
        "paper_use": "context_only",
        "claim_use": "context_only",
        "source_type": "official_model_card",
        "source_quality": "official_aggregate",
        "evidence_tier": "C",
        "setting_complete": True,
        "model_revision_known": False,
        "excluded_from_statistical_tests": True,
        "caveat": "Pretrained-model aggregate from model card; not the same prompt/run as this paper.",
    },
    {
        "model_or_family": "google/gemma-3-4b-it",
        "reported_model_scope": "Gemma 3 PT 4B model-card aggregate",
        "benchmark": "MMLU-Pro",
        "setting": "5-shot, CoT",
        "reported_score": 29.2,
        "score_unit": "accuracy_percent",
        "source_url": "https://huggingface.co/google/gemma-3-4b-it",
        "paper_use": "context_only",
        "claim_use": "context_only",
        "source_type": "official_model_card",
        "source_quality": "official_aggregate",
        "evidence_tier": "C",
        "setting_complete": True,
        "model_revision_known": False,
        "excluded_from_statistical_tests": True,
        "caveat": "Aggregate public benchmark score; cannot reveal corrected errors or regressions.",
    },
    {
        "model_or_family": "meta-llama/Llama-3.2-1B-Instruct",
        "reported_model_scope": "Llama 3.2 1B base model-card aggregate",
        "benchmark": "MMLU",
        "setting": "5-shot macro_avg/acc_char",
        "reported_score": 32.2,
        "score_unit": "accuracy_percent",
        "source_url": "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct",
        "paper_use": "context_only",
        "claim_use": "context_only",
        "source_type": "official_model_card",
        "source_quality": "official_aggregate",
        "evidence_tier": "C",
        "setting_complete": True,
        "model_revision_known": False,
        "excluded_from_statistical_tests": True,
        "caveat": "Base-model aggregate in public card; not paired item-level output from this study.",
    },
    {
        "model_or_family": "meta-llama/Llama-3.2-1B-Instruct",
        "reported_model_scope": "Llama 3.2 1B base model-card aggregate",
        "benchmark": "ARC-Challenge",
        "setting": "25-shot acc_char",
        "reported_score": 32.8,
        "score_unit": "accuracy_percent",
        "source_url": "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct",
        "paper_use": "context_only",
        "claim_use": "context_only",
        "source_type": "official_model_card",
        "source_quality": "official_aggregate",
        "evidence_tier": "C",
        "setting_complete": True,
        "model_revision_known": False,
        "excluded_from_statistical_tests": True,
        "caveat": "Different benchmark and setting; useful only as aggregate context.",
    },
    {
        "model_or_family": "Qwen/Qwen2-0.5B",
        "reported_model_scope": "Qwen2 technical report small-model table",
        "benchmark": "MMLU",
        "setting": "report setting; aggregate",
        "reported_score": 45.4,
        "score_unit": "accuracy_percent",
        "source_url": "https://arxiv.org/abs/2407.10671",
        "paper_use": "context_only",
        "claim_use": "context_only",
        "source_type": "technical_report",
        "source_quality": "official_aggregate",
        "evidence_tier": "C",
        "setting_complete": False,
        "model_revision_known": False,
        "excluded_from_statistical_tests": True,
        "caveat": "Aggregate report score; useful for positioning only.",
    },
    {
        "model_or_family": "Qwen/Qwen2-0.5B",
        "reported_model_scope": "Qwen2 technical report small-model table",
        "benchmark": "MMLU-Pro",
        "setting": "report setting; aggregate",
        "reported_score": 14.7,
        "score_unit": "accuracy_percent",
        "source_url": "https://arxiv.org/abs/2407.10671",
        "paper_use": "context_only",
        "claim_use": "context_only",
        "source_type": "technical_report",
        "source_quality": "official_aggregate",
        "evidence_tier": "C",
        "setting_complete": False,
        "model_revision_known": False,
        "excluded_from_statistical_tests": True,
        "caveat": "Aggregate report score; cannot identify corrected errors or regressions.",
    },
    {
        "model_or_family": "Qwen/Qwen2-1.5B",
        "reported_model_scope": "Qwen2 technical report small-model table",
        "benchmark": "MMLU-Pro",
        "setting": "report setting; aggregate",
        "reported_score": 21.8,
        "score_unit": "accuracy_percent",
        "source_url": "https://arxiv.org/abs/2407.10671",
        "paper_use": "context_only",
        "claim_use": "context_only",
        "source_type": "technical_report",
        "source_quality": "official_aggregate",
        "evidence_tier": "C",
        "setting_complete": False,
        "model_revision_known": False,
        "excluded_from_statistical_tests": True,
        "caveat": "Aggregate report score; cannot identify item-level replacement churn.",
    },
    {
        "model_or_family": "Phi-2",
        "reported_model_scope": "Qwen2 technical report comparison table",
        "benchmark": "MMLU",
        "setting": "report setting; aggregate",
        "reported_score": 52.7,
        "score_unit": "accuracy_percent",
        "source_url": "https://arxiv.org/abs/2407.10671",
        "paper_use": "context_only",
        "claim_use": "context_only",
        "source_type": "technical_report",
        "source_quality": "third_party_aggregate_in_report",
        "evidence_tier": "C",
        "setting_complete": False,
        "model_revision_known": False,
        "excluded_from_statistical_tests": True,
        "caveat": "Aggregate comparison score from a report table; not item-level output.",
    },
    {
        "model_or_family": "Qwen small models",
        "reported_model_scope": "Qwen2.5/Qwen3 technical reports and model cards",
        "benchmark": "mixed public benchmarks",
        "setting": "varies by report",
        "reported_score": "",
        "score_unit": "not_transcribed",
        "source_url": "https://arxiv.org/abs/2412.15115",
        "paper_use": "context_only",
        "claim_use": "context_only",
        "source_type": "technical_report",
        "source_quality": "official_aggregate_pointer",
        "evidence_tier": "C",
        "setting_complete": False,
        "model_revision_known": False,
        "excluded_from_statistical_tests": True,
        "caveat": "Included as a source pointer only; no cross-paper numeric comparison is made.",
    },
    {
        "model_or_family": "Qwen3 small models",
        "reported_model_scope": "Qwen3 technical report and model cards",
        "benchmark": "mixed public benchmarks",
        "setting": "varies by report",
        "reported_score": "",
        "score_unit": "not_transcribed",
        "source_url": "https://arxiv.org/abs/2505.09388",
        "paper_use": "context_only",
        "claim_use": "context_only",
        "source_type": "technical_report",
        "source_quality": "official_aggregate_pointer",
        "evidence_tier": "C",
        "setting_complete": False,
        "model_revision_known": False,
        "excluded_from_statistical_tests": True,
        "caveat": "Included as a source pointer only; item-level churn requires paired outputs.",
    },
]

INFORMAL_ROWS: list[dict[str, Any]] = [
    {
        "model_or_family": "google/gemma-4-E4B",
        "reported_model_scope": "informal user benchmark report",
        "benchmark": "MMLU-Pro Computer Science",
        "setting": "consumer-hardware report; exact harness and revision not archived here",
        "reported_score": 68.8,
        "score_unit": "accuracy_percent",
        "source_url": "https://www.linkedin.com/posts/nicholasacarroll_gemma-4-small-model-benchmarks-activity-7446994230763393024-QZLZ",
        "paper_use": "context_only",
        "claim_use": "context_only",
        "source_type": "community_benchmark_report",
        "source_quality": "informal",
        "evidence_tier": "D",
        "setting_complete": False,
        "model_revision_known": False,
        "excluded_from_statistical_tests": True,
        "caveat": "Informal single-subject report; not used for paper claims or statistical tests.",
    },
    {
        "model_or_family": "Llama 3.2 1B/3B",
        "reported_model_scope": "informal blog hardware/performance discussion",
        "benchmark": "BABILong-style long-context examples",
        "setting": "blog report; exact harness not archived here",
        "reported_score": "",
        "score_unit": "not_transcribed",
        "source_url": "https://medium.com/pythoneers/llama-3-2-1b-and-3b-small-but-mighty-23648ca7a431",
        "paper_use": "context_only",
        "claim_use": "context_only",
        "source_type": "community_blog",
        "source_quality": "informal",
        "evidence_tier": "D",
        "setting_complete": False,
        "model_revision_known": False,
        "excluded_from_statistical_tests": True,
        "caveat": "Logged only to show informal evidence was screened; no numerical claim is made from it.",
    },
]

REQUIRED_FIELDS = ["model_or_family", "benchmark", "setting", "source_url", "paper_use", "claim_use", "caveat"]


def validate_rows(rows: list[dict[str, Any]]) -> None:
    problems: list[str] = []
    for idx, row in enumerate(rows):
        for field in REQUIRED_FIELDS:
            if not str(row.get(field, "")).strip():
                problems.append(f"row {idx} missing {field}")
        if str(row.get("paper_use")) != "context_only":
            problems.append(f"row {idx} must be context_only")
        if str(row.get("claim_use")) != "context_only":
            problems.append(f"row {idx} claim_use must be context_only")
        if str(row.get("evidence_tier", "C")) in {"C", "D"} and not bool(row.get("excluded_from_statistical_tests", True)):
            problems.append(f"row {idx} aggregate/informal context must be excluded from statistical tests")
    if problems:
        raise ValueError("; ".join(problems))


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


def source_label(row: dict[str, Any]) -> str:
    value = str(row.get("source_type", "")).replace("_", " ")
    return value or "source"


def write_latex_table(path: Path, rows: list[dict[str, Any]], *, max_rows: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = rows[:max_rows]
    lines = [
        "\\begin{tabular}{llp{0.20\\linewidth}llp{0.20\\linewidth}l}",
        "\\toprule",
        "Model/source & Benchmark & Setting & Score & Source & URL & Access \\\\",
        "\\midrule",
    ]
    for row in selected:
        score = row["reported_score"] if str(row["reported_score"]) else "--"
        lines.append(
            f"{latex_escape(row['model_or_family'])} & {latex_escape(row['benchmark'])} & "
            f"{latex_escape(row['setting'])} & {latex_escape(score)} & "
            f"{latex_escape(source_label(row))} & \\url{{{row['source_url']}}} & "
            f"{latex_escape(row.get('access_date', ''))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def generate(output_dir: Path, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    access_date = current_access_date()
    context_rows = [dict(row, access_date=row.get("access_date", access_date)) for row in (rows or DEFAULT_ROWS)]
    informal_rows = [dict(row, access_date=row.get("access_date", access_date)) for row in (INFORMAL_ROWS if rows is None else [])]
    validate_rows(context_rows)
    validate_rows(informal_rows)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": len(context_rows),
        "informal_context_row_count": len(informal_rows),
        "access_date": access_date,
        "paper_use": "context_only",
        "claim_boundary": "External aggregate benchmark scores are not used as item-level churn evidence.",
    }
    write_csv(output_dir / "external_benchmark_context.csv", context_rows)
    write_json(output_dir / "external_benchmark_context.json", context_rows)
    write_csv(output_dir / "informal_benchmark_context.csv", informal_rows)
    write_json(output_dir / "informal_benchmark_context.json", informal_rows)
    write_json(output_dir / "external_benchmark_context_report.json", report)
    write_latex_table(output_dir / "tables" / "external_benchmark_context.tex", context_rows)
    write_latex_table(output_dir / "tables" / "informal_benchmark_context.tex", informal_rows)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write external aggregate benchmark context table.")
    parser.add_argument("--output-dir", type=Path, default=Path("main/analysis/external_benchmark_context"))
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    report = generate(args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
