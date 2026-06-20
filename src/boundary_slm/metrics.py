from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Any, Iterable


def accuracy(rows: Iterable[dict[str, Any]]) -> float:
    items = list(rows)
    if not items:
        return 0.0
    return sum(1 for row in items if row.get("is_correct")) / len(items)


def summarize_records(records: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[tuple(record.get(key) for key in group_keys)].append(record)
    rows: list[dict[str, Any]] = []
    for key, items in sorted(buckets.items(), key=lambda item: str(item[0])):
        row = {group_keys[idx]: key[idx] for idx in range(len(group_keys))}
        row.update(
            {
                "n": len(items),
                "accuracy": round(accuracy(items), 6),
                "answered_rate": round(sum(1 for item in items if item.get("answered")) / len(items), 6),
                "format_ok_rate": round(sum(1 for item in items if item.get("format_ok")) / len(items), 6),
                "avg_elapsed_seconds": round(sum(float(item.get("elapsed_seconds", 0.0)) for item in items) / len(items), 6),
                "avg_tokens_per_second": round(sum(float(item.get("tokens_per_second", 0.0)) for item in items) / len(items), 6),
            }
        )
        rows.append(row)
    return rows


def error_ecology_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = [row for row in records if row.get("condition") == "baseline"]
    by_family_size: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in baseline:
        by_family_size[(str(row["family"]), float(row["parameter_b"]))].append(row)

    comparisons: list[dict[str, Any]] = []
    family_passes: set[str] = set()
    for (family, parameter_b), rows in sorted(by_family_size.items()):
        by_generation: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_generation[str(row["generation"])].append(row)
        ordered = sorted(by_generation, key=_generation_value)
        for old_generation, new_generation in zip(ordered, ordered[1:]):
            old_rows = _index_by_item(by_generation[old_generation])
            new_rows = _index_by_item(by_generation[new_generation])
            common = sorted(set(old_rows) & set(new_rows))
            if not common:
                continue
            old_errors = {item for item in common if not old_rows[item].get("is_correct")}
            new_errors = {item for item in common if not new_rows[item].get("is_correct")}
            union = old_errors | new_errors
            intersection = old_errors & new_errors
            regressions = {item for item in common if old_rows[item].get("is_correct") and not new_rows[item].get("is_correct")}
            improvements = {item for item in common if not old_rows[item].get("is_correct") and new_rows[item].get("is_correct")}
            old_acc = 1.0 - len(old_errors) / len(common)
            new_acc = 1.0 - len(new_errors) / len(common)
            regression_mass = len(regressions) / len(common)
            accuracy_gain = new_acc - old_acc
            passed = accuracy_gain >= 0.01 and regression_mass <= 0.08
            if passed:
                family_passes.add(family)
            comparisons.append(
                {
                    "family": family,
                    "parameter_b": parameter_b,
                    "old_generation": old_generation,
                    "new_generation": new_generation,
                    "n_common": len(common),
                    "old_accuracy": round(old_acc, 6),
                    "new_accuracy": round(new_acc, 6),
                    "accuracy_gain": round(accuracy_gain, 6),
                    "error_redistribution_index": round(1.0 - len(intersection) / len(union), 6) if union else 0.0,
                    "regression_mass": round(regression_mass, 6),
                    "improvement_mass": round(len(improvements) / len(common), 6),
                    "passed_newer_better_gate": passed,
                }
            )

    claim_pass = len(family_passes) >= 2
    return {
        "experiment": "error_ecology",
        "family_pass_count": len(family_passes),
        "family_passes": sorted(family_passes),
        "comparisons": comparisons,
        "claim_check": {
            "primary_claim": "Newer boundary SLM generations improve accuracy without unacceptable error regression mass in at least two families.",
            "pass_boolean": claim_pass,
            "failure_reason": "" if claim_pass else "fewer than two families passed the preregistered newer-better gate",
        },
    }


def interface_stats(records: list[dict[str, Any]], *, bootstrap_iters: int = 1000, seed: int = 17) -> dict[str, Any]:
    by_family_condition: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_family_condition[(str(row["family"]), str(row["condition"]))].append(row)

    family_rows: list[dict[str, Any]] = []
    family_passes: set[str] = set()
    for family in sorted({key[0] for key in by_family_condition}):
        baseline = by_family_condition.get((family, "baseline"), [])
        baseline_idx = _index_by_pair(baseline)
        base_acc = accuracy(baseline)
        worst_tax = 0.0
        worst_condition = ""
        worst_ci: tuple[float, float] = (0.0, 0.0)
        for (condition_family, condition), items in by_family_condition.items():
            if condition_family != family or condition == "baseline":
                continue
            condition_idx = _index_by_pair(items)
            common = sorted(set(baseline_idx) & set(condition_idx))
            if not common:
                continue
            deltas = [
                (1.0 if baseline_idx[key].get("is_correct") else 0.0)
                - (1.0 if condition_idx[key].get("is_correct") else 0.0)
                for key in common
            ]
            tax = sum(deltas) / len(deltas)
            ci_low, ci_high = bootstrap_ci(deltas, iters=bootstrap_iters, seed=seed)
            family_rows.append(
                {
                    "family": family,
                    "condition": condition,
                    "n_common": len(common),
                    "baseline_accuracy": round(base_acc, 6),
                    "condition_accuracy": round(accuracy([condition_idx[key] for key in common]), 6),
                    "interface_tax": round(tax, 6),
                    "ci_low": round(ci_low, 6),
                    "ci_high": round(ci_high, 6),
                    "format_violation_rate": round(
                        sum(1 for key in common if not condition_idx[key].get("format_ok")) / len(common),
                        6,
                    ),
                }
            )
            if tax > worst_tax:
                worst_tax = tax
                worst_condition = condition
                worst_ci = (ci_low, ci_high)
        if worst_tax >= 0.03 and worst_ci[0] > 0.0:
            family_passes.add(family)
        if worst_condition:
            family_rows.append(
                {
                    "family": family,
                    "condition": "__worst__",
                    "n_common": 0,
                    "baseline_accuracy": round(base_acc, 6),
                    "condition_accuracy": 0.0,
                    "interface_tax": round(worst_tax, 6),
                    "ci_low": round(worst_ci[0], 6),
                    "ci_high": round(worst_ci[1], 6),
                    "format_violation_rate": 0.0,
                    "worst_condition": worst_condition,
                }
            )

    claim_pass = len(family_passes) >= 2
    return {
        "experiment": "interface",
        "family_pass_count": len(family_passes),
        "family_passes": sorted(family_passes),
        "interface_rows": family_rows,
        "claim_check": {
            "primary_claim": "Interface interventions impose a paired reliability tax with confidence intervals excluding zero in at least two model families.",
            "pass_boolean": claim_pass,
            "failure_reason": "" if claim_pass else "fewer than two families passed the preregistered interface-tax gate",
        },
    }


def bootstrap_ci(values: list[float], *, iters: int = 1000, seed: int = 17, alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(max(1, iters)):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(sum(sample) / len(sample))
    estimates.sort()
    low_idx = max(0, int((alpha / 2) * len(estimates)) - 1)
    high_idx = min(len(estimates) - 1, int((1 - alpha / 2) * len(estimates)))
    return (estimates[low_idx], estimates[high_idx])


def mcnemar_exact(b01: int, b10: int) -> float:
    n = b01 + b10
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b01, b10) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0 for _ in p_values]
    running = 0.0
    m = len(p_values)
    for rank, (idx, value) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * value))
        adjusted[idx] = running
    return adjusted


def _index_by_item(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["item_id"]): row for row in rows}


def _index_by_pair(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["model_label"]), str(row["item_id"])): row for row in rows}


def _generation_value(value: str) -> float:
    cleaned = "".join(ch if ch.isdigit() or ch == "." else " " for ch in value)
    parts = [float(part) for part in cleaned.split() if part]
    return parts[-1] if parts else 0.0

