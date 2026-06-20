from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


COLORS = {
    "qwen": "#005AB5",
    "gemma": "#DC3220",
}

MARKERS = {
    "qwen": "o",
    "gemma": "s",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value: str) -> float:
    return float(value)


def compact_model_label(value: str) -> str:
    replacements = {
        "Qwen": "Q",
        "Gemma": "G",
        "-Instruct": "",
        "-Instruct-2507": "-2507",
    }
    out = value
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def plot_delta_vs_churn(rows: list[dict[str, str]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for family in sorted({row["family"] for row in rows}):
        vals = [row for row in rows if row["family"] == family]
        xs = [100 * f(row["accuracy_delta"]) for row in vals]
        ys = [100 * f(row["churn_mass"]) for row in vals]
        ax.scatter(
            xs,
            ys,
            label=family.title(),
            color=COLORS.get(family, "#444444"),
            marker=MARKERS.get(family, "o"),
            s=72,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
        )
    ax.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Accuracy delta (percentage points)")
    ax.set_ylabel("Churn mass (% of items changed)")
    ax.set_title("Accuracy gains can coincide with item-level churn")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_family_bars(rows: list[dict[str, str]], output: Path) -> None:
    families = [row["family"].title() for row in rows]
    metrics = [
        ("mean_error_persistence", "Old-error persistence"),
        ("mean_regression_gap", "Regression gap"),
        ("mean_churn_mass", "Churn mass"),
    ]
    x = range(len(families))
    width = 0.24
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for idx, (key, label) in enumerate(metrics):
        offsets = [pos + (idx - 1) * width for pos in x]
        ax.bar(offsets, [100 * f(row[key]) for row in rows], width=width, label=label)
    ax.set_xticks(list(x), families)
    ax.set_ylabel("Percent")
    ax.set_title("Family-level primary pairwise accounting")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_model_level_persistence(rows: list[dict[str, str]], output: Path) -> None:
    if not rows:
        return
    families = sorted({row["family"] for row in rows})
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for xpos, family in enumerate(families):
        vals = [row for row in rows if row["family"] == family]
        vals = sorted(vals, key=lambda row: row["model"])
        ys = [100 * f(row["mean_error_persistence_across_incident_pairs"]) for row in vals]
        offsets = [(idx - (len(vals) - 1) / 2) * 0.065 for idx in range(len(vals))]
        ax.scatter(
            [xpos + offset for offset in offsets],
            ys,
            color=COLORS.get(family, "#444444"),
            marker=MARKERS.get(family, "o"),
            s=76,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
            label=family.title(),
        )
        mean_y = sum(ys) / len(ys)
        ax.hlines(
            mean_y,
            xpos - 0.28,
            xpos + 0.28,
            color="#222222",
            linewidth=1.4,
        )
    ax.set_xticks(range(len(families)), [family.title() for family in families])
    ax.set_ylabel("Mean old-error persistence across incident pairs (%)")
    ax.set_title("Model-level persistence values used for family testing")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_near_parity(rows: list[dict[str, str]], output: Path) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda row: f(row["churn_mass"]))
    labels = [
        " ->\n".join(compact_model_label(part.strip()) for part in row["comparison_id"].split("->"))
        for row in rows
    ]
    churn = [100 * f(row["churn_mass"]) for row in rows]
    colors = [COLORS.get(row["family"], "#444444") for row in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    y = range(len(rows))
    ax.barh(list(y), churn, color=colors, alpha=0.82)
    ax.set_yticks(list(y), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Churn mass (% of items changed)")
    ax.set_title("Near-parity comparisons can still move many outcomes")
    ax.grid(True, axis="x", alpha=0.25)
    for ypos, row in zip(y, rows):
        ax.text(
            churn[ypos] + 0.7,
            ypos,
            f"delta={100 * f(row['accuracy_delta']):+.1f}pp",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def generate(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pairwise = read_csv(input_dir / "enriched_pairwise_metrics.csv")
    family = read_csv(input_dir / "family_summary.csv")
    near = read_csv(input_dir / "near_parity_comparisons.csv")
    model_persistence = read_csv(input_dir / "model_level_error_persistence.csv")
    plot_delta_vs_churn(pairwise, output_dir / "delta_vs_churn.pdf")
    plot_family_bars(family, output_dir / "family_accounting.pdf")
    plot_model_level_persistence(model_persistence, output_dir / "model_level_persistence.pdf")
    plot_near_parity(near, output_dir / "near_parity_churn.pdf")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate figures for the Boundary-SLM paper.")
    parser.add_argument("--input-dir", type=Path, default=Path("main/analysis/paper_metrics"))
    parser.add_argument("--output-dir", type=Path, default=Path("main/papers/error_persistence_regression_gap/figures"))
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    generate(args.input_dir, args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
