from __future__ import annotations

import gc
import json
import os
import resource
import shutil
import time
import traceback
import ctypes
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

try:
    _LIBC = ctypes.CDLL("libc.so.6")
except OSError:
    _LIBC = None


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
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass
        try:
            torch.cuda.reset_peak_memory_stats()
        except RuntimeError:
            pass
    if _LIBC is not None and hasattr(_LIBC, "malloc_trim"):
        try:
            _LIBC.malloc_trim(0)
        except (AttributeError, OSError):
            pass


def _shutdown_dataloader_workers(dataloader: Any) -> None:
    if dataloader is None:
        return

    iterator = getattr(dataloader, "_iterator", None)
    if iterator is not None:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass
        dataloader._iterator = None


def shutdown_trainer_dataloader(trainer: Any, attr_name: str) -> None:
    if trainer is None or not hasattr(trainer, attr_name):
        return

    dataloader = getattr(trainer, attr_name, None)
    _shutdown_dataloader_workers(dataloader)
    setattr(trainer, attr_name, None)


def teardown_trainer(trainer: Any) -> None:
    if trainer is None:
        return

    for attr_name in (
        "_train_dataloader",
        "_eval_dataloader",
        "_test_dataloader",
    ):
        dataloader = getattr(trainer, attr_name, None)
        _shutdown_dataloader_workers(dataloader)
        setattr(trainer, attr_name, None)

    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None:
        free_memory = getattr(accelerator, "free_memory", None)
        if callable(free_memory):
            try:
                free_memory()
            except Exception:
                pass

    model = getattr(trainer, "model", None)
    if model is not None:
        try:
            model.to("cpu")
        except Exception:
            pass

    for attr_name in (
        "model",
        "model_wrapped",
        "optimizer",
        "lr_scheduler",
        "train_dataset",
        "eval_dataset",
        "_signature_columns",
        "_train_batch_size",
        "_raw_eval_examples",
        "_postprocess_eval_examples",
        "_postprocess_eval_features",
    ):
        if hasattr(trainer, attr_name):
            setattr(trainer, attr_name, None)

    callback_handler = getattr(trainer, "callback_handler", None)
    if callback_handler is not None and hasattr(callback_handler, "callbacks"):
        callback_handler.callbacks = []


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
