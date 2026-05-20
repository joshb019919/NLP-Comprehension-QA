from __future__ import annotations

import gc
import json
import os
import resource
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import torch


HF_ROOT = Path("/data/data/huggingface")
DEFAULT_DIRS = {
    "root": HF_ROOT,
    "datasets": HF_ROOT / "datasets",
    "models": HF_ROOT / "models",
    "tokenized": HF_ROOT / "tokenized",
    "runs": HF_ROOT / "runs",
    "bench": HF_ROOT / "bench",
    "tmp": HF_ROOT / "tmp",
}

def ensure_dirs() -> dict[str, Path]:
    for path in DEFAULT_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)
    return DEFAULT_DIRS


def configure_runtime(tokenizer_parallelism: bool = False) -> None:
    ensure_dirs()
    os.environ.setdefault("HF_HOME", str(DEFAULT_DIRS["root"]))
    os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_DIRS["datasets"]))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(DEFAULT_DIRS["models"]))
    os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_DIRS["models"] / "hub"))
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if tokenizer_parallelism else "false"

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


def maybe_set_process_memory_limit(limit_mb: int | None) -> None:
    if limit_mb is None:
        return
    limit_bytes = int(limit_mb) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))


def atomic_save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp_path.replace(path)


def remove_dir_if_exists(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def utc_ts() -> float:
    return time.time()


class MultiprocessWorkerError(RuntimeError):
    """Raised in a worker process with enough detail for the parent to inspect."""


def _summarize_worker_arg(value: Any) -> str:
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            try:
                size = len(item)
            except TypeError:
                size = "?"
            parts.append(f"{key}={type(item).__name__}[len={size}]")
        return "{" + ", ".join(parts) + "}"

    try:
        size = len(value)
    except TypeError:
        size = "?"
    return f"{type(value).__name__}[len={size}]"


class WorkerExceptionWrapper:
    """
    Wraps Dataset.map worker functions so subprocess tracebacks survive num_proc mode.
    """

    def __init__(self, fn: Any, operation: str):
        self.fn = fn
        self.operation = operation

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self.fn(*args, **kwargs)
        except Exception as exc:
            arg_summary = ", ".join(_summarize_worker_arg(arg) for arg in args) or "none"
            kwargs_summary = (
                ", ".join(f"{key}={_summarize_worker_arg(value)}" for key, value in kwargs.items())
                or "none"
            )
            tb = traceback.format_exc()
            message = (
                f"Multiprocessing worker failed during {self.operation}.\n"
                f"Worker PID: {os.getpid()}\n"
                f"Function: {getattr(self.fn, '__name__', type(self.fn).__name__)}\n"
                f"Exception: {type(exc).__name__}: {exc}\n"
                f"Args: {arg_summary}\n"
                f"Kwargs: {kwargs_summary}\n"
                "Worker traceback:\n"
                f"{tb}"
            )
            raise MultiprocessWorkerError(message) from None
