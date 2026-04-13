from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, IterableDatasetDict, load_dataset

from .common import DEFAULT_DIRS, remove_dir_if_exists
from .config import AppConfig
from .postprocess import build_alias_list, select_best_alias_match


COMPLETE_MARKER = "_COMPLETE"


class LoadRetryError(RuntimeError):
    pass



def _safe_load_dataset(config: AppConfig, split: str | None = None):
    kwargs = {
        "path": config.dataset.dataset_name,
        "name": config.dataset.dataset_config_name,
        "cache_dir": str(DEFAULT_DIRS["datasets"]),
        "streaming": config.run.streaming,
    }
    if split is not None:
        kwargs["split"] = split

    try:
        return load_dataset(**kwargs)
    except Exception:
        if not config.run.retry_broken_downloads:
            raise
        kwargs["download_mode"] = "force_redownload"
        return load_dataset(**kwargs)



def load_raw_splits(config: AppConfig):
    train_ds = _safe_load_dataset(config, config.dataset.train_split)
    valid_ds = _safe_load_dataset(config, config.dataset.validation_split)
    test_ds = _safe_load_dataset(config, config.dataset.test_split) if config.dataset.test_split else None
    return train_ds, valid_ds, test_ds



def _limit_examples(ds, max_examples: int | None):
    if max_examples is None:
        return ds
    if hasattr(ds, "take") and not isinstance(ds, Dataset):
        return ds.take(max_examples)
    n = min(max_examples, len(ds))
    return ds.select(range(n))



def maybe_limit_splits(train_ds, valid_ds, test_ds, max_examples: int | None):
    return (
        _limit_examples(train_ds, max_examples),
        _limit_examples(valid_ds, max_examples),
        _limit_examples(test_ds, max_examples) if test_ds is not None else None,
    )



def expand_triviaqa_rows(ds, filter_no_answer_rows_for_training: bool = False):
    def generator():

        for example in ds:
            qid = str(example.get("question_id") or example.get("id") or example.get("question"))
            question = example["question"]
            answer = example.get("answer", {}) or {}
            answer_value = answer.get("value")
            aliases = build_alias_list(answer_value, answer.get("aliases", []))

            entity_pages = example.get("entity_pages", {}) or {}
            wiki_contexts = entity_pages.get("wiki_context", []) or []
            search_results = example.get("search_results", {}) or {}
            search_contexts = search_results.get("search_context", []) or []

            merged_contexts: list[tuple[str, str]] = []
            merged_contexts.extend(("wiki", c) for c in wiki_contexts if c)
            merged_contexts.extend(("search", c) for c in search_contexts if c)

            for idx, (source, context) in enumerate(merged_contexts):
                match = select_best_alias_match(context, aliases)
                if match is None:
                    answer_start = []
                    answer_text = []
                    has_answer = False
                else:
                    answer_start = [match[0]]
                    answer_text = [match[1]]
                    has_answer = True

                if filter_no_answer_rows_for_training and not has_answer:
                    continue

                yield {
                    "id": f"{qid}_{source}_{idx}",
                    "parent_id": qid,
                    "question": question,
                    "context": context,
                    "answers": {"text": answer_text, "answer_start": answer_start},
                    "aliases": aliases,
                    "answer_value": answer_value or "",
                    "context_source": source,
                    "context_idx": idx,
                    "has_answer": has_answer,
                }

    return Dataset.from_generator(generator, keep_in_memory=False)


def normalize_standard_qa_columns(ds: Dataset, id_column: str = "id") -> Dataset:
    required = set(ds.column_names)
    if {"id", "question", "context", "answers"}.issubset(required):
        return ds

    def mapper(example, idx):
        answers = example.get("answers", {"text": [], "answer_start": []})
        return {
            "id": str(example.get(id_column, idx)),
            "question": example["question"],
            "context": example["context"],
            "answers": answers,
            "aliases": [],
            "has_answer": len(answers.get("text", [])) > 0,
        }

    return ds.map(mapper, with_indices=True)



def prepare_raw_qa_splits(config: AppConfig):
    train_ds, valid_ds, test_ds = load_raw_splits(config)
    train_ds, valid_ds, test_ds = maybe_limit_splits(train_ds, valid_ds, test_ds, config.run.max_examples)

    if config.dataset.dataset_name == "trivia_qa" and config.dataset.expand_triviaqa_contexts:
        train_ds = expand_triviaqa_rows(train_ds, filter_no_answer_rows_for_training=config.dataset.filter_no_answer_rows_for_training)
        valid_ds = expand_triviaqa_rows(valid_ds, filter_no_answer_rows_for_training=False)
        if test_ds is not None:
            test_ds = expand_triviaqa_rows(test_ds, filter_no_answer_rows_for_training=False)
    else:
        if isinstance(train_ds, Dataset):
            train_ds = normalize_standard_qa_columns(train_ds)
            valid_ds = normalize_standard_qa_columns(valid_ds)
            if test_ds is not None:
                test_ds = normalize_standard_qa_columns(test_ds)

    return train_ds, valid_ds, test_ds



def estimated_dataset_nbytes(ds: Dataset, sample_size: int = 128) -> int:
    if len(ds) == 0:
        return 1024 * 1024
    k = min(sample_size, len(ds))
    total = 0
    for i in range(k):
        row = ds[i]
        total += sum(len(str(v)) for v in row.values())
    avg = max(1, total // k)
    return avg * len(ds)



def dynamic_shard_size_str(ds: Dataset, preprocess_num_chunks: int) -> str:
    estimated = estimated_dataset_nbytes(ds)
    chunk_bytes = max(1, estimated // max(1, preprocess_num_chunks))
    shard_bytes = max(16 * 1024 * 1024, chunk_bytes // 2)
    shard_bytes = min(shard_bytes, 2 * 1024 * 1024 * 1024)
    mb = max(16, math.ceil(shard_bytes / (1024 * 1024)))
    return f"{mb}MB"



def tokenized_cache_is_complete(path: Path) -> bool:
    return path.exists() and (path / COMPLETE_MARKER).exists()



def mark_tokenized_cache_complete(path: Path) -> None:
    (path / COMPLETE_MARKER).write_text("ok\n", encoding="utf-8")



def rebuild_incomplete_tokenized_dir(path: Path, force: bool) -> None:
    if force or (path.exists() and not tokenized_cache_is_complete(path)):
        remove_dir_if_exists(path)
