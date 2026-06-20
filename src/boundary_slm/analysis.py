from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from boundary_slm.io import read_jsonl, write_csv, write_json
from boundary_slm.metrics import error_ecology_stats, interface_stats, summarize_records


def analyze_output_root(input_root: Path) -> dict[str, object]:
    records = read_jsonl(input_root / "records.jsonl")
    experiments = {row.get("experiment") for row in records}
    if experiments == {"error_ecology"}:
        stats = error_ecology_stats(records)
        summary = summarize_records(records, ["family", "generation", "model_label", "task_family", "condition"])
    elif experiments == {"interface"}:
        stats = interface_stats(records)
        summary = summarize_records(records, ["family", "model_label", "task_family", "condition"])
    else:
        raise ValueError(f"Expected one experiment in records, found: {sorted(str(item) for item in experiments)}")
    write_csv(input_root / "summary.csv", summary)
    write_json(input_root / "stats.json", stats)
    write_json(input_root / "claim_check.json", stats["claim_check"])
    return {"records": len(records), "summary_rows": len(summary), "claim_check": stats["claim_check"]}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate Boundary-SLM stats from records.jsonl.")
    parser.add_argument("--input-root", type=Path, required=True)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    print(analyze_output_root(args.input_root))


if __name__ == "__main__":
    main()

