from __future__ import annotations

import json
import re
from typing import Any

from boundary_slm.tasks import EvalItem


CHOICE_PATTERN = re.compile(r"\b([A-E])\b", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")


def normalize_text(value: str) -> str:
    return " ".join(value.strip().split()).lower()


def extract_choice(text: str) -> str | None:
    matches = CHOICE_PATTERN.findall(text.upper())
    return matches[-1].upper() if matches else None


def extract_last_number(text: str) -> str | None:
    matches = NUMBER_PATTERN.findall(text.replace("####", " "))
    if not matches:
        return None
    return matches[-1].replace(",", "").rstrip(".")


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    candidates = [stripped]
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        candidates.append(stripped[first : last + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def score_response(item: EvalItem, response_text: str) -> dict[str, Any]:
    answer_type = item.answer_type
    answered = bool(response_text.strip())
    prediction: str | None
    format_ok = True

    if answer_type == "multiple_choice":
        prediction = extract_choice(response_text)
        format_ok = prediction is not None and normalize_text(response_text) in {
            "a",
            "b",
            "c",
            "d",
            "e",
        }
        correct = prediction == item.answer.upper()
    elif answer_type == "number":
        prediction = extract_last_number(response_text)
        correct = prediction == item.answer
    elif answer_type == "json_value":
        payload = _extract_json_object(response_text)
        key = str(item.metadata.get("json_key", "selected"))
        allowed_keys = set(str(key) for key in item.metadata.get("allowed_keys", [key]))
        if payload is None:
            prediction = None
            format_ok = False
            correct = False
        else:
            prediction = str(payload.get(key, ""))
            format_ok = set(payload.keys()) == allowed_keys
            correct = prediction == item.answer and format_ok
    else:
        prediction = normalize_text(response_text)
        correct = prediction == normalize_text(item.answer)

    return {
        "expected": item.answer,
        "prediction": prediction,
        "is_correct": bool(correct),
        "answered": bool(answered and prediction is not None),
        "format_ok": bool(format_ok),
    }

