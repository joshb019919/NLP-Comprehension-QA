from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    model_name_or_path: str = "bert-base-uncased"
    tokenizer_name_or_path: str | None = None
    revision: str | None = None
    use_fast_tokenizer: bool = True


@dataclass
class DatasetConfig:
    dataset_name: str = "trivia_qa"
    dataset_config_name: str | None = "rc"
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str | None = "test"
    version_2_with_negative: bool = False
    null_score_diff_threshold: float = 0.0
    expand_triviaqa_contexts: bool = True
    filter_no_answer_rows_for_training: bool = True
    keep_in_memory: bool = False


@dataclass
class RunConfig:
    output_name: str = "bert_trivia_qa_rc"

    # Runtime
    streaming: bool = False
    # torch_compile: bool = False  # Set for debugging
    torch_compile: bool = True   # Set for full runs
    tokenizer_parallelism: bool = False
    force_rebuild_cache: bool = False
    process_memory_limit_mb: int | None = None

    # Preprocessing and dataloading
    max_length: int = 384
    doc_stride: int = 64
    max_examples: int | None = None
    preprocessing_num_proc: int | None = 8    # Set for parallel, more RAM
    # preprocessing_num_proc: int | None = 0  # Set for debuggin, less RAM
    preprocess_num_chunks: int = 8
    map_batch_size: int = 128
    writer_batch_size: int = 128
    dataloader_num_workers: int = 8    # Set for more RAM use, parallel
    # dataloader_num_workers: int = 0  # Set for less RAM, debugging
    dataloader_pin_memory: bool = True
    dataloader_drop_last: bool = False
    dataloader_persistent_workers: bool = True if dataloader_num_workers > 0 else False
    dataloader_prefetch_factor: int | None = 8 if dataloader_num_workers > 0 else None

    # num_example limit for already-cached token files
    limit_after_tokenization = True
    after_tokenization_limit = 50000

    # Optimization
    learning_rate: float = 5e-5
    # batch_size: int = 8  # Set for easier VRAM
    batch_size: int = 72   # Set for more VRAM use
    epochs: int = 2
    lr_scheduler: str = "cosine"
    weight_decay: float = 1e-8
    # gradient_accumulation_steps: int = 8    # Set for low it, lower it/s
    gradient_accumulation_steps: int = 1      # Set for high it, higher it/s
    strategy: str = "epoch"
    optim: str = "adamw_torch"
    gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0
    load_best_model_at_end: bool = True
    remove_unused_columns: bool = False
    report_to: str = "none"
    save_total_limit: int = 5

    # Eval / save
    logging_steps: int = 20
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True

    # Paths
    data_root: str = "/data/data/huggingface"

    # Extra behavior
    retry_broken_downloads: bool = True
    seed: int = 42  # The answer


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base



def parse_cli_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none" or lowered == "null":
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw



def apply_overrides(config_dict: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for item in overrides:
        key, value = item.split("=", 1)
        parsed = parse_cli_value(value)
        parts = key.split(".")
        cursor = config_dict
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = parsed
    return config_dict



def load_app_config(run_config_path: str | Path, overrides: list[str] | None = None) -> AppConfig:
    run_cfg = load_json(run_config_path)
    model_cfg = load_json(run_cfg["model_config_path"])
    dataset_cfg = load_json(run_cfg["dataset_config_path"])

    merged = {
        "model": model_cfg,
        "dataset": dataset_cfg,
        "run": {k: v for k, v in run_cfg.items() if k not in {"model_config_path", "dataset_config_path"}},
    }
    if overrides:
        merged = apply_overrides(merged, overrides)

    return AppConfig(
        model=ModelConfig(**merged.get("model", {})),
        dataset=DatasetConfig(**merged.get("dataset", {})),
        run=RunConfig(**merged.get("run", {})),
    )
