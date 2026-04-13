from __future__ import annotations

import json
import os
import resource
import shutil
import time
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
    torch.set_float32_matmul_precision("medium")
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


def utc_ts() -> float:
    return time.time()
