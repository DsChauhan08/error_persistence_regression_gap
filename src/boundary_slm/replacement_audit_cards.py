from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from boundary_slm.io import write_csv, write_json


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def pct(value: Any) -> float | str:
    if value in {None, ""}:
        return "NA"
    return round(100.0 * f(value), 3)


def pct_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return f"{float(value):.1f}%"


def replacement_status(row: dict[str, str]) -> str:
    delta = f(row.get("accuracy_delta"))
    if delta > 0:
        return "positive_delta_candidate"
    if delta < 0:
        return "negative_delta_swap"
    return "accuracy_tie"


def path_interpretation(row: dict[str, str]) -> str:
    value = row.get("path_interpretation")
    if value:
        return value
    return "not_classified"


def risk_flags(
    row: dict[str, str],
    *,
    churn_threshold: float,
    regression_threshold: float,
    nrb_threshold: float,
    near_parity_threshold: float,
) -> list[str]:
    flags: list[str] = []
    if f(row.get("churn_mass")) >= churn_threshold:
        flags.append("high_churn")
    if f(row.get("regression_mass")) >= regression_threshold:
        flags.append("high_regression_mass")
    if f(row.get("normalized_regression_burden")) >= nrb_threshold:
        flags.append("high_normalized_regression_burden")
    if abs(f(row.get("accuracy_delta"))) <= near_parity_threshold and f(row.get("churn_mass")) > 0:
        flags.append("near_parity_churn")
    if replacement_status(row) != "positive_delta_candidate":
        flags.append("not_a_successful_replacement")
    return flags or ["no_default_flag"]


def review_priority(flags: list[str]) -> str:
    high_flags = {
        "high_churn",
        "high_regression_mass",
        "high_normalized_regression_burden",
        "near_parity_churn",
        "not_a_successful_replacement",
    }
    if "not_a_successful_replacement" in flags:
        return "exclude_or_reframe"
    if len(high_flags.intersection(flags)) >= 2:
        return "manual_review_required"
    if high_flags.intersection(flags):
        return "manual_review_recommended"
    return "routine_monitoring"


def audit_card(
    row: dict[str, str],
    *,
    churn_threshold: float,
    regression_threshold: float,
    nrb_threshold: float,
    near_parity_threshold: float,
) -> dict[str, Any]:
    flags = risk_flags(
        row,
        churn_threshold=churn_threshold,
        regression_threshold=regression_threshold,
        nrb_threshold=nrb_threshold,
        near_parity_threshold=near_parity_threshold,
    )
    return {
        "comparison_id": row["comparison_id"],
        "family": row["family"],
        "current_model": row["old_model"],
        "candidate_model": row["new_model"],
        "status": replacement_status(row),
        "path_interpretation": path_interpretation(row),
        "review_priority": review_priority(flags),
        "risk_flags": "; ".join(flags),
        "n_items": int(f(row.get("n_common"))),
        "current_accuracy_pct": pct(row.get("old_accuracy")),
        "candidate_accuracy_pct": pct(row.get("new_accuracy")),
        "accuracy_delta_pct": pct(row.get("accuracy_delta")),
        "improvement_mass_pct": pct(row.get("improvement_mass")),
        "regression_mass_pct": pct(row.get("regression_mass")),
        "churn_mass_pct": pct(row.get("churn_mass")),
        "error_persistence_pct": pct(row.get("error_persistence")),
        "correction_rate_pct": pct(row.get("correction_rate")),
        "normalized_regression_burden_pct": pct(row.get("normalized_regression_burden")),
        "net_gain_per_changed_item": round(f(row.get("net_gain_per_changed_item")), 6),
        "improvement_to_regression_ratio": round(f(row.get("improvement_to_regression_ratio")), 6),
        "top_improving_categories": row.get("top_improving_categories", ""),
        "top_regressing_categories": row.get("top_regressing_categories", ""),
    }


def generate_cards(
    rows: list[dict[str, str]],
    *,
    churn_threshold: float = 0.25,
    regression_threshold: float = 0.08,
    nrb_threshold: float = 0.10,
    near_parity_threshold: float = 0.05,
) -> list[dict[str, Any]]:
    cards = [
        audit_card(
            row,
            churn_threshold=churn_threshold,
            regression_threshold=regression_threshold,
            nrb_threshold=nrb_threshold,
            near_parity_threshold=near_parity_threshold,
        )
        for row in rows
    ]
    return sorted(
        cards,
        key=lambda row: (
            row["review_priority"] != "exclude_or_reframe",
            row["review_priority"] != "manual_review_required",
            -row["churn_mass_pct"],
            row["comparison_id"],
        ),
    )


def write_markdown(path: Path, cards: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Replacement Audit Cards",
        "",
        "These cards are generated from quality-gated same-family pairwise outputs. Percent fields are percentages over the shared item set unless otherwise stated.",
        "",
        "## Thresholds",
        "",
        f"- High churn: >= {100 * metadata['churn_threshold']:.1f}%",
        f"- High regression mass: >= {100 * metadata['regression_threshold']:.1f}%",
        f"- High normalized regression burden: >= {100 * metadata['nrb_threshold']:.1f}%",
        f"- Near parity: absolute accuracy delta <= {100 * metadata['near_parity_threshold']:.1f}%",
        "",
        "## Cards",
        "",
    ]
    for card in cards:
        lines.extend(
            [
                f"### {card['comparison_id']}",
                "",
                f"- Status: `{card['status']}`",
                f"- Review priority: `{card['review_priority']}`",
                f"- Risk flags: `{card['risk_flags']}`",
                f"- Accuracy: {card['current_accuracy_pct']:.1f}% -> {card['candidate_accuracy_pct']:.1f}% ({card['accuracy_delta_pct']:+.1f} points)",
                f"- Improvement/regression/churn: {card['improvement_mass_pct']:.1f}% / {card['regression_mass_pct']:.1f}% / {card['churn_mass_pct']:.1f}%",
                f"- Error persistence/correction rate: {pct_text(card['error_persistence_pct'])} / {pct_text(card['correction_rate_pct'])}",
                f"- Normalized regression burden: {pct_text(card['normalized_regression_burden_pct'])}",
                f"- Top improving categories: {card['top_improving_categories']}",
                f"- Top regressing categories: {card['top_regressing_categories']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def generate(
    input_csv: Path,
    output_dir: Path,
    *,
    churn_threshold: float = 0.25,
    regression_threshold: float = 0.08,
    nrb_threshold: float = 0.10,
    near_parity_threshold: float = 0.05,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(input_csv)
    cards = generate_cards(
        rows,
        churn_threshold=churn_threshold,
        regression_threshold=regression_threshold,
        nrb_threshold=nrb_threshold,
        near_parity_threshold=near_parity_threshold,
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(input_csv),
        "card_count": len(cards),
        "churn_threshold": churn_threshold,
        "regression_threshold": regression_threshold,
        "nrb_threshold": nrb_threshold,
        "near_parity_threshold": near_parity_threshold,
    }
    write_csv(output_dir / "replacement_audit_cards.csv", cards)
    write_json(output_dir / "replacement_audit_cards.json", {"metadata": metadata, "cards": cards})
    write_markdown(output_dir / "replacement_audit_cards.md", cards, metadata)
    return {"metadata": metadata, "cards": cards}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reusable replacement-audit cards from pairwise metrics.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("main/analysis/paper_metrics/enriched_pairwise_metrics.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("main/analysis/paper_metrics"))
    parser.add_argument("--churn-threshold", type=float, default=0.25)
    parser.add_argument("--regression-threshold", type=float, default=0.08)
    parser.add_argument("--nrb-threshold", type=float, default=0.10)
    parser.add_argument("--near-parity-threshold", type=float, default=0.05)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    payload = generate(
        args.input_csv,
        args.output_dir,
        churn_threshold=args.churn_threshold,
        regression_threshold=args.regression_threshold,
        nrb_threshold=args.nrb_threshold,
        near_parity_threshold=args.near_parity_threshold,
    )
    print(json.dumps(payload["metadata"], indent=2))


if __name__ == "__main__":
    main()
