from __future__ import annotations

import hashlib
from pathlib import Path

from common import DEFAULT_DIRS, ensure_dirs
from config import AppConfig


def _slug(text: str) -> str:
    return text.replace("/", "--").replace(" ", "_")



def run_signature(config: AppConfig) -> str:
    text = (
        f"{config.model.model_name_or_path}|{config.dataset.dataset_name}|"
        f"{config.dataset.dataset_config_name}|{config.run.max_length}|{config.run.doc_stride}|"
        f"{config.run.max_examples}|{config.run.max_train_examples}|"
        f"{config.run.max_validation_examples}|{config.run.max_test_examples}|"
        f"{config.run.streaming}|{config.run.preprocess_num_chunks}|{config.run.seed}|"
        f"{config.run.limit_after_tokenization}|{config.run.after_tokenization_limit}|"
        f"{config.run.after_tokenization_train_limit}|"
        f"{config.run.after_tokenization_validation_limit}|"
        f"{config.run.after_tokenization_test_limit}|"
        f"{config.dataset.validation_dataset_name}|{config.dataset.validation_dataset_config_name}|"
        f"{config.dataset.validation_version_2_with_negative}|"
        f"{config.dataset.test_dataset_name}|{config.dataset.test_dataset_config_name}|"
        f"{config.dataset.test_version_2_with_negative}|"
        f"{config.dataset.expand_triviaqa_contexts}|{config.dataset.filter_no_answer_rows_for_training}"
    )
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]



def get_run_paths(config: AppConfig) -> dict[str, Path]:
    ensure_dirs()
    sig = run_signature(config)
    base = DEFAULT_DIRS["runs"] / f"{config.run.output_name}_{sig}"
    model_slug = _slug(config.model.model_name_or_path)
    dataset_slug = _slug(config.dataset.dataset_name + ("__" + config.dataset.dataset_config_name if config.dataset.dataset_config_name else ""))
    tokenized = DEFAULT_DIRS["tokenized"] / dataset_slug / model_slug / sig
    bench_dir = DEFAULT_DIRS["bench"] / f"{config.run.output_name}_{sig}"
    return {
        "output_dir": base,
        "tokenized_dir": tokenized,
        "bench_dir": bench_dir,
        "bench_file": bench_dir / "bench.jsonl",
        "resolved_config": base / "resolved_config.json",
    }
