from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any

from boundary_slm.io import load_json_or_yaml


DEFAULT_TASK_REGISTRY = Path(__file__).resolve().parents[2] / "configs" / "task_registry.yaml"


@dataclass(frozen=True)
class EvalItem:
    id: str
    task: str
    task_family: str
    prompt: str
    answer: str
    answer_type: str
    choices: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "task_family": self.task_family,
            "prompt": self.prompt,
            "answer": self.answer,
            "answer_type": self.answer_type,
            "choices": list(self.choices),
            "metadata": self.metadata,
        }


def _math_items(rng: random.Random, count: int) -> list[EvalItem]:
    rows: list[EvalItem] = []
    for idx in range(count):
        a = rng.randint(11, 98)
        b = rng.randint(7, 64)
        c = rng.randint(2, 9)
        answer = (a + b) * c
        rows.append(
            EvalItem(
                id=f"proc_math_{idx:04d}",
                task="procedural_arithmetic",
                task_family="math",
                prompt=(
                    "Solve the arithmetic problem. Return only the final number.\n"
                    f"Problem: ({a} + {b}) * {c} = ?"
                ),
                answer=str(answer),
                answer_type="number",
                metadata={"a": a, "b": b, "c": c},
            )
        )
    return rows


def _mcq_items(rng: random.Random, count: int) -> list[EvalItem]:
    rows: list[EvalItem] = []
    for idx in range(count):
        values = [rng.randint(10, 99) for _ in range(4)]
        target_index = max(range(4), key=lambda pos: (values[pos] % 10, values[pos]))
        letters = "ABCD"
        choices = tuple(f"{letters[pos]}. {values[pos]}" for pos in range(4))
        prompt = (
            "Choose the option whose number has the largest ones digit. "
            "If tied, choose the larger number.\n"
            + "\n".join(choices)
            + "\nReturn only A, B, C, or D."
        )
        rows.append(
            EvalItem(
                id=f"proc_mcq_{idx:04d}",
                task="procedural_multiple_choice",
                task_family="multiple_choice",
                prompt=prompt,
                answer=letters[target_index],
                answer_type="multiple_choice",
                choices=choices,
                metadata={"values": values},
            )
        )
    return rows


def _instruction_items(rng: random.Random, count: int) -> list[EvalItem]:
    rows: list[EvalItem] = []
    words = [
        "amber",
        "cobalt",
        "delta",
        "ember",
        "fable",
        "graph",
        "harbor",
        "ion",
        "jade",
        "kelp",
    ]
    for idx in range(count):
        selected = rng.sample(words, 3)
        required = selected[1].upper()
        prompt = (
            "Follow the instruction exactly. Output a JSON object with one key named "
            f"selected and the value {required!r}. Do not include any other keys.\n"
            f"Words: {', '.join(selected)}"
        )
        rows.append(
            EvalItem(
                id=f"proc_instruction_{idx:04d}",
                task="procedural_instruction_following",
                task_family="instruction",
                prompt=prompt,
                answer=required,
                answer_type="json_value",
                metadata={"json_key": "selected", "allowed_keys": ["selected"]},
            )
        )
    return rows


def _code_items(rng: random.Random, count: int) -> list[EvalItem]:
    rows: list[EvalItem] = []
    for idx in range(count):
        start = rng.randint(2, 12)
        step = rng.randint(2, 5)
        loops = rng.randint(2, 5)
        value = start
        for _ in range(loops):
            value += step
        prompt = (
            "What does this Python snippet print? Return only the printed value.\n"
            f"x = {start}\n"
            f"for _ in range({loops}):\n"
            f"    x += {step}\n"
            "print(x)"
        )
        rows.append(
            EvalItem(
                id=f"proc_code_{idx:04d}",
                task="procedural_code_execution",
                task_family="code",
                prompt=prompt,
                answer=str(value),
                answer_type="number",
                metadata={"start": start, "step": step, "loops": loops},
            )
        )
    return rows


def build_task_items(
    path: Path | None = None,
    *,
    smoke: bool = False,
    seed: int | None = None,
) -> list[EvalItem]:
    payload = load_json_or_yaml(path or DEFAULT_TASK_REGISTRY)
    profile = "smoke" if smoke else "full"
    counts = payload["procedural_counts"][profile]
    rng = random.Random(seed if seed is not None else int(payload.get("default_seed", 17)))
    items: list[EvalItem] = []
    items.extend(_math_items(rng, int(counts.get("math", 0))))
    items.extend(_mcq_items(rng, int(counts.get("multiple_choice", 0))))
    items.extend(_instruction_items(rng, int(counts.get("instruction", 0))))
    items.extend(_code_items(rng, int(counts.get("code", 0))))
    return items


def load_task_config(path: Path | None = None) -> dict[str, Any]:
    return dict(load_json_or_yaml(path or DEFAULT_TASK_REGISTRY))

