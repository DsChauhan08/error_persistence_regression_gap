from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any

from boundary_slm.io import write_json
from boundary_slm.models import ModelSpec
from boundary_slm.tasks import EvalItem


@dataclass(frozen=True)
class GenerationResult:
    text: str
    elapsed_seconds: float
    completion_tokens: int
    backend_name: str
    error: str = ""

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.completion_tokens / self.elapsed_seconds


class BaseBackend:
    name = "base"

    def generate(self, model: ModelSpec, item: EvalItem, prompt: str, *, condition: str, seed: int) -> GenerationResult:
        raise NotImplementedError

    def close(self) -> None:
        return None


class MockBackend(BaseBackend):
    name = "mock"

    def generate(self, model: ModelSpec, item: EvalItem, prompt: str, *, condition: str, seed: int) -> GenerationResult:
        start = time.perf_counter()
        digest = hashlib.sha256(f"{model.label}|{item.id}|{condition}|{seed}".encode("utf-8")).hexdigest()
        score = int(digest[:8], 16) / 0xFFFFFFFF
        base_quality = min(0.82, 0.34 + 0.08 * model.parameter_b)
        if "3.5" in model.generation or model.generation == "gemma4":
            base_quality += 0.08
        if condition != "baseline":
            base_quality -= _mock_intervention_penalty(condition, model)
        correct = score < max(0.05, min(0.95, base_quality))
        text = _mock_answer(item, correct, condition)
        elapsed = max(0.0001, time.perf_counter() - start)
        return GenerationResult(
            text=text,
            elapsed_seconds=elapsed,
            completion_tokens=max(1, len(text.split())),
            backend_name=self.name,
        )


class TransformersBackend(BaseBackend):
    name = "transformers"

    def __init__(self, *, cache_dir: Path | None = None, max_new_tokens: int = 128) -> None:
        self.cache_dir = cache_dir
        self.max_new_tokens = max_new_tokens
        self._loaded_repo_id: str | None = None
        self._tokenizer: Any = None
        self._processor: Any = None
        self._model: Any = None
        self._device: Any = None

    def generate(self, model: ModelSpec, item: EvalItem, prompt: str, *, condition: str, seed: int) -> GenerationResult:
        start = time.perf_counter()
        try:
            self._ensure_loaded(model)
            import torch

            torch.manual_seed(seed)
            tokenizer = self._processor or self._tokenizer
            if self._processor is not None:
                messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            else:
                messages = [{"role": "user", "content": prompt}]
            if hasattr(tokenizer, "apply_chat_template"):
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            else:
                inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            input_len = int(inputs["input_ids"].shape[-1])
            with torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=getattr(self._tokenizer, "eos_token_id", None),
                )
            generated = outputs[0][input_len:]
            decoder = self._tokenizer or self._processor
            text = decoder.decode(generated, skip_special_tokens=True)
            elapsed = time.perf_counter() - start
            return GenerationResult(
                text=text,
                elapsed_seconds=elapsed,
                completion_tokens=int(generated.shape[-1]),
                backend_name=self.name,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return GenerationResult(
                text="",
                elapsed_seconds=elapsed,
                completion_tokens=0,
                backend_name=self.name,
                error=repr(exc),
            )

    def _ensure_loaded(self, model: ModelSpec) -> None:
        if self._loaded_repo_id == model.repo_id:
            return
        self.close()
        import torch
        import transformers

        self._device = _torch_device()
        dtype = torch.bfloat16 if str(self._device).startswith("xla") else "auto"
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        common_kwargs: dict[str, Any] = {
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "token": token,
            "trust_remote_code": True,
        }
        common_kwargs = {key: value for key, value in common_kwargs.items() if value is not None}
        try:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(model.repo_id, **common_kwargs)
        except Exception:
            self._tokenizer = None
        try:
            self._processor = transformers.AutoProcessor.from_pretrained(model.repo_id, **common_kwargs)
        except Exception:
            self._processor = None
        if self._tokenizer is None and self._processor is None:
            raise RuntimeError(f"Could not load tokenizer or processor for {model.repo_id}")

        load_errors: list[str] = []
        for class_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForMultimodalLM"):
            model_cls = getattr(transformers, class_name, None)
            if model_cls is None:
                continue
            try:
                self._model = model_cls.from_pretrained(
                    model.repo_id,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    **common_kwargs,
                )
                break
            except Exception as exc:
                load_errors.append(f"{class_name}: {exc!r}")
        if self._model is None:
            raise RuntimeError(f"Could not load model class for {model.repo_id}: {'; '.join(load_errors)}")
        self._model.to(self._device)
        self._model.eval()
        self._loaded_repo_id = model.repo_id

    def close(self) -> None:
        self._loaded_repo_id = None
        self._tokenizer = None
        self._processor = None
        self._model = None
        self._device = None
        try:
            import gc

            gc.collect()
        except Exception:
            pass


def build_backend(name: str | None = None, *, cache_dir: Path | None = None, max_new_tokens: int = 128) -> BaseBackend:
    backend = (name or os.environ.get("BOUNDARY_SLM_BACKEND") or "mock").strip().lower()
    if backend == "mock":
        return MockBackend()
    if backend == "transformers":
        return TransformersBackend(cache_dir=cache_dir, max_new_tokens=max_new_tokens)
    raise ValueError(f"Unsupported backend: {backend}")


def probe_environment(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cwd": str(Path.cwd()),
        "running_on_colab": "COLAB_RELEASE_TAG" in os.environ,
        "running_on_kaggle": bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/working").exists(),
        "pjrt_device": os.environ.get("PJRT_DEVICE", ""),
        "hf_token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
    }
    usage = shutil.disk_usage(Path.cwd())
    payload["disk_free_gb"] = round(usage.free / (1024**3), 2)
    try:
        import jax

        payload["jax_devices"] = [str(device) for device in jax.devices()]
        payload["jax_device_count"] = len(payload["jax_devices"])
    except Exception as exc:
        payload["jax_error"] = repr(exc)
    try:
        import torch_xla.core.xla_model as xm

        payload["torch_xla_device"] = str(xm.xla_device())
    except Exception as exc:
        payload["torch_xla_error"] = repr(exc)
    write_json(output_root / "environment.json", payload)
    return payload


def _torch_device() -> Any:
    try:
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
    except Exception:
        import torch

        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


def _mock_intervention_penalty(condition: str, model: ModelSpec) -> float:
    if condition.startswith("context_long"):
        return 0.16
    if condition.startswith("context_medium"):
        return 0.09
    if condition.startswith("context_short"):
        return 0.04
    if condition.startswith("multimodal"):
        return 0.11 if model.family != "gemma" else 0.07
    if condition.startswith("strict_format"):
        return 0.08
    if condition == "system_conflict":
        return 0.07
    if condition.startswith("thinking"):
        return 0.03
    return 0.0


def _mock_answer(item: EvalItem, correct: bool, condition: str) -> str:
    if correct:
        if item.answer_type == "json_value":
            return '{"selected":"' + item.answer + '"}'
        return item.answer
    if item.answer_type == "multiple_choice":
        letters = ["A", "B", "C", "D"]
        return next(letter for letter in letters if letter != item.answer)
    if item.answer_type == "json_value":
        if condition.startswith("strict_format"):
            return '{"selected":"WRONG"}'
        return "selected: WRONG"
    if item.answer_type == "number":
        try:
            return str(int(float(item.answer)) + 1)
        except ValueError:
            return "0"
    return "incorrect"
