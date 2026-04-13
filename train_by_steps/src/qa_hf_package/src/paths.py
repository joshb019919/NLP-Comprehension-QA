from __future__ import annotations

import hashlib
from pathlib import Path

from .common import ensure_dirs, get_storage_root
from .config import AppConfig


def _slug(text: str) -> str:
    return text.replace("/", "--").replace(" ", "_")



def run_signature(config: AppConfig) -> str:
    text = (
        f"{config.model.model_name_or_path}|{config.dataset.dataset_name}|"
        f"{config.dataset.dataset_config_name}|{config.run.max_length}|{config.run.doc_stride}|"
        f"{config.run.max_examples}|{config.run.streaming}|{config.run.preprocess_num_chunks}|"
        f"{config.dataset.expand_triviaqa_contexts}|{config.dataset.filter_no_answer_rows_for_training}"
    )
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]



def get_run_paths(config: AppConfig) -> dict[str, Path]:
    storage_root = get_storage_root(
        use_data_root=config.run.use_data_root,
        data_root=config.run.data_root,
        storage_subfolder=config.run.storage_subfolder,
    )
    dirs = ensure_dirs(storage_root)
    sig = run_signature(config)
    base = dirs["runs"] / f"{config.run.output_name}_{sig}"
    model_slug = _slug(config.model.model_name_or_path)
    dataset_slug = _slug(config.dataset.dataset_name + ("__" + config.dataset.dataset_config_name if config.dataset.dataset_config_name else ""))
    tokenized = dirs["tokenized"] / dataset_slug / model_slug / sig
    bench_dir = dirs["bench"] / f"{config.run.output_name}_{sig}"
    return {
        "output_dir": base,
        "tokenized_dir": tokenized,
        "bench_dir": bench_dir,
        "bench_file": bench_dir / "bench.jsonl",
        "resolved_config": base / "resolved_config.json",
    }
