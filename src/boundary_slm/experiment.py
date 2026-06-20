from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from boundary_slm.interventions import apply_intervention, default_interventions, model_supports_intervention
from boundary_slm.io import append_jsonl, read_jsonl, write_csv, write_json
from boundary_slm.metrics import error_ecology_stats, interface_stats, summarize_records
from boundary_slm.models import ModelSpec, load_model_registry
from boundary_slm.runtime import BaseBackend, build_backend, probe_environment
from boundary_slm.scoring import score_response
from boundary_slm.tasks import EvalItem, build_task_items, load_task_config


def run_error_ecology(
    *,
    output_root: Path,
    backend_name: str = "mock",
    smoke: bool = False,
    include_appendix: bool = False,
    seed: int = 17,
) -> dict[str, Any]:
    models = load_model_registry(include_appendix=include_appendix, smoke=smoke)
    items = build_task_items(smoke=smoke, seed=seed)
    backend = build_backend(backend_name, cache_dir=output_root / "model_cache")
    try:
        return _run(
            experiment="error_ecology",
            output_root=output_root,
            backend=backend,
            models=models,
            items=items,
            conditions=["baseline"],
            seed=seed,
        )
    finally:
        backend.close()


def run_interface(
    *,
    output_root: Path,
    backend_name: str = "mock",
    smoke: bool = False,
    include_appendix: bool = False,
    seed: int = 17,
) -> dict[str, Any]:
    models = load_model_registry(include_appendix=include_appendix, smoke=smoke)
    items = build_task_items(smoke=smoke, seed=seed)
    backend = build_backend(backend_name, cache_dir=output_root / "model_cache")
    try:
        interventions = default_interventions()
        return _run(
            experiment="interface",
            output_root=output_root,
            backend=backend,
            models=models,
            items=items,
            conditions=[intervention.name for intervention in interventions],
            seed=seed,
        )
    finally:
        backend.close()


def _run(
    *,
    experiment: str,
    output_root: Path,
    backend: BaseBackend,
    models: list[ModelSpec],
    items: list[EvalItem],
    conditions: list[str],
    seed: int,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    run_manifest_path = output_root / "run_manifest.json"
    records_path = output_root / "records.jsonl"
    exclusions_path = output_root / "availability_exclusions.json"

    existing_records = read_jsonl(records_path)
    completed = {
        (row.get("experiment"), row.get("model_label"), row.get("item_id"), row.get("condition"))
        for row in existing_records
    }
    exclusions: list[dict[str, Any]] = []
    new_records: list[dict[str, Any]] = []
    interventions = {item.name: item for item in default_interventions()}

    write_json(
        run_manifest_path,
        {
            "experiment": experiment,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "backend": backend.name,
            "seed": seed,
            "model_count": len(models),
            "item_count": len(items),
            "conditions": conditions,
            "runtime_notes": [
                "TPU runs should use BF16 and PJRT_DEVICE=TPU.",
                "CUDA-only quantization packages are intentionally not used.",
                "Models are loaded one at a time by the backend.",
            ],
        },
    )
    probe_environment(output_root)

    for model in models:
        for condition in conditions:
            intervention = interventions.get(condition)
            if intervention and not model_supports_intervention(model, intervention):
                exclusions.append(
                    {
                        "model_label": model.label,
                        "repo_id": model.repo_id,
                        "condition": condition,
                        "reason": "model does not support required interface",
                    }
                )
                continue
            for item in items:
                key = (experiment, model.label, item.id, condition)
                if key in completed:
                    continue
                prompt = item.prompt
                intervention_metadata: dict[str, str] = {}
                if intervention:
                    prompt, intervention_metadata = apply_intervention(item, intervention)
                result = backend.generate(model, item, prompt, condition=condition, seed=seed)
                scored = score_response(item, result.text)
                record = {
                    "experiment": experiment,
                    "model_label": model.label,
                    "repo_id": model.repo_id,
                    "family": model.family,
                    "generation": model.generation,
                    "parameter_b": model.parameter_b,
                    "boundary_role": model.boundary_role,
                    "item_id": item.id,
                    "task": item.task,
                    "task_family": item.task_family,
                    "answer_type": item.answer_type,
                    "condition": condition,
                    "prompt": prompt,
                    "response_text": result.text,
                    "elapsed_seconds": round(result.elapsed_seconds, 6),
                    "completion_tokens": result.completion_tokens,
                    "tokens_per_second": round(result.tokens_per_second, 6),
                    "backend_name": result.backend_name,
                    "error": result.error,
                    **scored,
                    **intervention_metadata,
                }
                new_records.append(record)
                if len(new_records) >= 100:
                    append_jsonl(records_path, new_records)
                    new_records.clear()
    if new_records:
        append_jsonl(records_path, new_records)

    if exclusions:
        write_json(exclusions_path, exclusions)

    all_records = read_jsonl(records_path)
    summary = _write_summaries(output_root, experiment, all_records)
    return summary


def _write_summaries(output_root: Path, experiment: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    if experiment == "error_ecology":
        summary_rows = summarize_records(records, ["family", "generation", "model_label", "task_family", "condition"])
        stats = error_ecology_stats(records)
    elif experiment == "interface":
        summary_rows = summarize_records(records, ["family", "model_label", "task_family", "condition"])
        config = load_task_config()
        iters = int(config["claim_thresholds"]["interface"].get("bootstrap_ci_iterations", 1000))
        stats = interface_stats(records, bootstrap_iters=iters)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    write_csv(output_root / "summary.csv", summary_rows)
    write_json(output_root / "stats.json", stats)
    write_json(output_root / "claim_check.json", stats["claim_check"])
    return {"summary_rows": len(summary_rows), "records": len(records), "claim_check": stats["claim_check"]}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Boundary-SLM experiment runner.")
    parser.add_argument("--experiment", choices=["error_ecology", "interface"], required=True)
    parser.add_argument("--backend", default=None, help="mock or transformers. Defaults to BOUNDARY_SLM_BACKEND or mock.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--include-appendix", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    backend_name = args.backend or "mock"
    if args.experiment == "error_ecology":
        payload = run_error_ecology(
            output_root=args.output_root,
            backend_name=backend_name,
            smoke=args.smoke,
            include_appendix=args.include_appendix,
            seed=args.seed,
        )
    else:
        payload = run_interface(
            output_root=args.output_root,
            backend_name=backend_name,
            smoke=args.smoke,
            include_appendix=args.include_appendix,
            seed=args.seed,
        )
    print(payload)


if __name__ == "__main__":
    main()

