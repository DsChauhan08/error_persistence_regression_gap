from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from boundary_slm.io import load_json_or_yaml


DEFAULT_MODEL_REGISTRY = Path(__file__).resolve().parents[2] / "configs" / "model_registry.yaml"


@dataclass(frozen=True)
class ModelSpec:
    label: str
    repo_id: str
    family: str
    generation: str
    parameter_b: float
    boundary_role: str
    modalities: tuple[str, ...]
    supports_system_role: bool
    supports_thinking_toggle: bool
    primary: bool

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "ModelSpec":
        return cls(
            label=str(record["label"]),
            repo_id=str(record["repo_id"]),
            family=str(record["family"]),
            generation=str(record["generation"]),
            parameter_b=float(record["parameter_b"]),
            boundary_role=str(record["boundary_role"]),
            modalities=tuple(str(item) for item in record.get("modalities", ["text"])),
            supports_system_role=bool(record.get("supports_system_role", False)),
            supports_thinking_toggle=bool(record.get("supports_thinking_toggle", False)),
            primary=bool(record.get("primary", True)),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "repo_id": self.repo_id,
            "family": self.family,
            "generation": self.generation,
            "parameter_b": self.parameter_b,
            "boundary_role": self.boundary_role,
            "modalities": list(self.modalities),
            "supports_system_role": self.supports_system_role,
            "supports_thinking_toggle": self.supports_thinking_toggle,
            "primary": self.primary,
        }


def load_model_registry(
    path: Path | None = None,
    *,
    include_appendix: bool = False,
    smoke: bool = False,
) -> list[ModelSpec]:
    payload = load_json_or_yaml(path or DEFAULT_MODEL_REGISTRY)
    models = [ModelSpec.from_record(item) for item in payload["models"]]
    if not include_appendix:
        models = [model for model in models if model.primary]
    if smoke:
        preferred = {
            "Qwen2.5-0.5B-Instruct",
            "Qwen3.5-0.8B",
            "Gemma-3-1B-It",
            "Llama-3.2-1B-Instruct",
        }
        models = [model for model in models if model.label in preferred]
    return models


def generation_order_key(model: ModelSpec) -> tuple[str, float, float, str]:
    numeric = "".join(ch if ch.isdigit() or ch == "." else " " for ch in model.generation)
    parts = [float(part) for part in numeric.split() if part]
    generation_value = parts[-1] if parts else 0.0
    return (model.family, model.parameter_b, generation_value, model.label)

