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
LOCAL_HF_ROOT = Path(__file__).resolve().parents[3] / "huggingface"


def get_storage_root(use_data_root: bool, data_root: str, storage_subfolder: str | None = None) -> Path:
    if use_data_root:
        root = Path(data_root)
    else:
        root = LOCAL_HF_ROOT

    if storage_subfolder:
        cleaned = storage_subfolder.strip().strip("/")
        if cleaned:
            root = root / cleaned
    return root


def get_default_dirs(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "datasets": root / "datasets",
        "models": root / "models",
        "tokenized": root / "tokenized",
        "runs": root / "runs",
        "bench": root / "bench",
        "tmp": root / "tmp",
    }


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = get_default_dirs(root)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def configure_runtime(storage_root: Path, tokenizer_parallelism: bool = False) -> None:
    dirs = ensure_dirs(storage_root)
    os.environ.setdefault("HF_HOME", str(dirs["root"]))
    os.environ.setdefault("HF_DATASETS_CACHE", str(dirs["datasets"]))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(dirs["models"]))
    os.environ.setdefault("HF_HUB_CACHE", str(dirs["models"] / "hub"))
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
