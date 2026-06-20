from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from boundary_slm.io import write_csv, write_json


EVIDENCE_TIERS = {"A", "B", "C", "D"}
CHURN_CLAIM_USES = {"item_level_churn", "item_level_correctness_replication"}
REQUIRED_FIELDS = [
    "source_id",
    "source_name",
    "source_url",
    "license",
    "access_date",
    "artifact_type",
    "evidence_tier",
    "claim_use",
    "access_status",
    "compatibility_status",
]


def utc_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def default_sources(access_date: str | None = None) -> list[dict[str, Any]]:
    date_value = access_date or utc_date()
    return [
        {
            "source_id": "wild",
            "source_name": "WILD: Wide-scale Item Level Dataset",
            "source_url": "https://huggingface.co/datasets/kensho/WILD",
            "license": "apache-2.0",
            "access_date": date_value,
            "source_type": "public_dataset",
            "source_quality": "formal_item_level",
            "artifact_type": "item_level_correctness",
            "benchmark_scope": "27 benchmarks; correctness matrix only",
            "evidence_tier": "B",
            "claim_use": "item_level_correctness_replication",
            "has_item_ids": True,
            "has_raw_outputs": False,
            "has_predictions": False,
            "has_aggregate_scores": True,
            "access_status": "public",
            "compatibility_status": "usable_public_replication",
            "setting_complete": True,
            "model_revision_known": False,
            "excluded_from_statistical_tests": False,
            "exclusion_reason": "",
            "notes": "Supports item-level churn over binary correctness scores, but not parser or raw-response claims.",
        },
        {
            "source_id": "wild_raw",
            "source_name": "WILD raw responses",
            "source_url": "https://huggingface.co/datasets/michaelkrumdickkensho/WILD-raw",
            "license": "apache-2.0",
            "access_date": date_value,
            "source_type": "public_dataset",
            "source_quality": "formal_raw_item_level",
            "artifact_type": "raw_item_level_outputs",
            "benchmark_scope": "raw conversations, targets, model answers, scorer output",
            "evidence_tier": "A",
            "claim_use": "optional_private_or_license_checked",
            "has_item_ids": True,
            "has_raw_outputs": True,
            "has_predictions": True,
            "has_aggregate_scores": False,
            "access_status": "public_large",
            "compatibility_status": "not_in_public_release_by_default",
            "setting_complete": True,
            "model_revision_known": False,
            "excluded_from_statistical_tests": True,
            "exclusion_reason": "Raw WILD responses are not required for correctness-only replication and are not redistributed by default.",
            "notes": "Raw responses are not needed for the public WILD replication and are not redistributed by this artifact.",
        },
        {
            "source_id": "open_llm_leaderboard_results",
            "source_name": "Open LLM Leaderboard aggregate results",
            "source_url": "https://huggingface.co/datasets/open-llm-leaderboard/results",
            "license": "see-source-card",
            "access_date": date_value,
            "source_type": "public_leaderboard",
            "source_quality": "aggregate_only",
            "artifact_type": "aggregate_scores",
            "benchmark_scope": "leaderboard aggregate task metrics",
            "evidence_tier": "C",
            "claim_use": "context_only",
            "has_item_ids": False,
            "has_raw_outputs": False,
            "has_predictions": False,
            "has_aggregate_scores": True,
            "access_status": "public",
            "compatibility_status": "aggregate_only",
            "setting_complete": False,
            "model_revision_known": False,
            "excluded_from_statistical_tests": True,
            "exclusion_reason": "Aggregate-only source cannot identify corrected errors or regressions.",
            "notes": "Useful for positioning, not for corrected-error or regression accounting.",
        },
        {
            "source_id": "open_llm_leaderboard_details",
            "source_name": "Open LLM Leaderboard per-model detail datasets",
            "source_url": "https://huggingface.co/datasets/open-llm-leaderboard/Qwen__Qwen2.5-0.5B-Instruct-details",
            "license": "gated-terms",
            "access_date": date_value,
            "source_type": "gated_dataset",
            "source_quality": "potential_item_level_if_accessible",
            "artifact_type": "item_level_samples",
            "benchmark_scope": "per-model sample files, including MMLU-Pro for several exact study models",
            "evidence_tier": "D",
            "tier_if_accessible": "A",
            "claim_use": "excluded_until_access_granted",
            "has_item_ids": True,
            "has_raw_outputs": True,
            "has_predictions": True,
            "has_aggregate_scores": True,
            "access_status": "gated_terms_required",
            "compatibility_status": "optional_skip_without_hf_access",
            "setting_complete": False,
            "model_revision_known": False,
            "excluded_from_statistical_tests": True,
            "exclusion_reason": "Access terms must be accepted before use.",
            "notes": "The public repository is visible but file access requires accepting dataset conditions.",
        },
        {
            "source_id": "opencompass_predictions",
            "source_name": "OpenCompass academic predictions",
            "source_url": "https://huggingface.co/datasets/opencompass/compass_academic_predictions",
            "license": "gated-terms",
            "access_date": date_value,
            "source_type": "gated_dataset",
            "source_quality": "potential_item_level_if_accessible",
            "artifact_type": "item_level_predictions",
            "benchmark_scope": "large prediction archive across many tasks and models",
            "evidence_tier": "D",
            "tier_if_accessible": "A",
            "claim_use": "excluded_until_access_granted",
            "has_item_ids": True,
            "has_raw_outputs": True,
            "has_predictions": True,
            "has_aggregate_scores": False,
            "access_status": "gated_terms_required",
            "compatibility_status": "optional_skip_without_hf_access",
            "setting_complete": False,
            "model_revision_known": False,
            "excluded_from_statistical_tests": True,
            "exclusion_reason": "Access terms must be accepted before use.",
            "notes": "Large optional archive; not part of public-first evidence.",
        },
        {
            "source_id": "model_cards_reports",
            "source_name": "Model cards and technical reports",
            "source_url": "https://huggingface.co/models",
            "license": "varies-by-model-card",
            "access_date": date_value,
            "source_type": "official_model_documentation",
            "source_quality": "official_aggregate",
            "artifact_type": "aggregate_scores_and_source_pointers",
            "benchmark_scope": "reported benchmark scores and model documentation",
            "evidence_tier": "C",
            "claim_use": "context_only",
            "has_item_ids": False,
            "has_raw_outputs": False,
            "has_predictions": False,
            "has_aggregate_scores": True,
            "access_status": "public_or_model-gated",
            "compatibility_status": "aggregate_only",
            "setting_complete": False,
            "model_revision_known": False,
            "excluded_from_statistical_tests": True,
            "exclusion_reason": "Aggregate-only documentation cannot support item-level replacement claims.",
            "notes": "Useful citation context; cannot support item-level replacement claims.",
        },
        {
            "source_id": "community_mmlu_pro_small_models",
            "source_name": "Community small-model MMLU-Pro discussions",
            "source_url": "https://www.reddit.com/r/LocalLLaMA/comments/1gii24g/mmlupro_scores_of_small_models_5b/",
            "license": "user-generated-platform",
            "access_date": date_value,
            "source_type": "community_discussion",
            "source_quality": "informal",
            "artifact_type": "informal_aggregate_or_anecdotal_scores",
            "benchmark_scope": "community-reported small-model impressions and occasional benchmark numbers",
            "evidence_tier": "D",
            "claim_use": "context_only",
            "has_item_ids": False,
            "has_raw_outputs": False,
            "has_predictions": False,
            "has_aggregate_scores": True,
            "access_status": "public_web",
            "compatibility_status": "informal_context_only",
            "setting_complete": False,
            "model_revision_known": False,
            "excluded_from_statistical_tests": True,
            "exclusion_reason": "Informal community source; settings and revisions are not controlled.",
            "notes": "Logged for source-quality transparency only; not used in paper claims or tests.",
        },
        {
            "source_id": "informal_gemma4_e4b_mmlu_pro_cs_report",
            "source_name": "Informal Gemma 4 E4B MMLU-Pro Computer Science report",
            "source_url": "https://www.linkedin.com/posts/nicholasacarroll_gemma-4-small-model-benchmarks-activity-7446994230763393024-QZLZ",
            "license": "user-generated-platform",
            "access_date": date_value,
            "source_type": "community_benchmark_report",
            "source_quality": "informal",
            "artifact_type": "single_aggregate_score_report",
            "benchmark_scope": "MMLU-Pro Computer Science aggregate reported on consumer hardware",
            "evidence_tier": "D",
            "claim_use": "context_only",
            "has_item_ids": False,
            "has_raw_outputs": False,
            "has_predictions": False,
            "has_aggregate_scores": True,
            "access_status": "public_web",
            "compatibility_status": "informal_context_only",
            "setting_complete": False,
            "model_revision_known": False,
            "excluded_from_statistical_tests": True,
            "exclusion_reason": "Informal single-subject report; not independently reproducible from this artifact.",
            "notes": "Reported score is not used as evidence for churn or replacement accounting.",
        },
    ]


def validate_sources(rows: list[dict[str, Any]]) -> None:
    problems: list[str] = []
    for idx, row in enumerate(rows):
        for field in REQUIRED_FIELDS:
            if str(row.get(field, "")).strip() == "":
                problems.append(f"row {idx} missing {field}")
        tier = str(row.get("evidence_tier", ""))
        if tier not in EVIDENCE_TIERS:
            problems.append(f"row {idx} has invalid evidence_tier {tier!r}")
        claim_use = str(row.get("claim_use", ""))
        if tier == "C" and claim_use in CHURN_CLAIM_USES:
            problems.append(f"row {idx} aggregate-only tier C cannot support churn claim_use")
        if tier == "D" and claim_use in CHURN_CLAIM_USES:
            problems.append(f"row {idx} unavailable/gated tier D cannot support churn claim_use")
        if claim_use == "item_level_churn" and tier != "A":
            problems.append(f"row {idx} item_level_churn requires tier A")
        if claim_use == "item_level_correctness_replication" and tier not in {"A", "B"}:
            problems.append(f"row {idx} item_level_correctness_replication requires tier A or B")
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


def write_claim_use_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = rows[:6]
    lines = [
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Source & Tier & Access & Paper use \\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(
            f"{latex_escape(row['source_id'])} & {latex_escape(row['evidence_tier'])} & "
            f"{latex_escape(row['access_status'])} & {latex_escape(row['claim_use'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tier_counts = Counter(str(row["evidence_tier"]) for row in rows)
    claim_use_counts = Counter(str(row["claim_use"]) for row in rows)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_count": len(rows),
        "tier_counts": dict(sorted(tier_counts.items())),
        "claim_use_counts": dict(sorted(claim_use_counts.items())),
        "claim_boundary": (
            "Only tier A/B item-level sources can support replacement churn metrics; "
            "tier C aggregate sources are context only and tier D sources are exclusions unless access is later granted."
        ),
        "public_first_source": "wild",
    }


def generate(output_dir: Path, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = rows or default_sources()
    validate_sources(source_rows)
    report = build_report(source_rows)
    write_csv(output_dir / "external_evidence_map.csv", source_rows)
    write_json(output_dir / "external_evidence_map.json", source_rows)
    write_json(output_dir / "external_source_manifest.json", report)
    write_claim_use_table(output_dir / "tables" / "external_claim_use_table.tex", source_rows)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the external evidence availability registry.")
    parser.add_argument("--output-dir", type=Path, default=Path("main/analysis/external_evidence"))
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    report = generate(args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
