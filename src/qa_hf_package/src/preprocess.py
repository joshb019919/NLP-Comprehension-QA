from __future__ import annotations

import re
from typing import Any

from datasets import Dataset, load_from_disk
from transformers import PreTrainedTokenizerBase

from .config import AppConfig
from .datasets import dynamic_shard_size_str, mark_tokenized_cache_complete, rebuild_incomplete_tokenized_dir, tokenized_cache_is_complete


def _tokenize_with_safe_stride(
    tokenizer: PreTrainedTokenizerBase,
    questions: list[str],
    contexts: list[str],
    config: AppConfig,
):

    return tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=config.run.max_length,
        stride=config.run.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )



def prepare_train_features(examples: dict[str, list[Any]], tokenizer: PreTrainedTokenizerBase, config: AppConfig) -> dict[str, list[Any]]:
    questions = [q.lstrip() for q in examples["question"]]
    tokenized = _tokenize_with_safe_stride(tokenizer, questions, examples["context"], config)

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []
    example_ids = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        example_ids.append(examples["id"][sample_index])

        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if offsets[token_start_index][0] > start_char or offsets[token_end_index][1] < end_char:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            start_positions.append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            end_positions.append(token_end_index + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    tokenized["example_id"] = example_ids
    return tokenized



def prepare_validation_features(examples: dict[str, list[Any]], tokenizer: PreTrainedTokenizerBase, config: AppConfig) -> dict[str, list[Any]]:
    questions = [q.lstrip() for q in examples["question"]]

    tokenized = _tokenize_with_safe_stride(tokenizer, questions, examples["context"], config)
    
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")

    tokenized["example_id"] = []
    
    # Create new offset_mapping with proper ignore values
    new_offset_mapping = []
    
    for i in range(len(tokenized["input_ids"])):
        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        tokenized["example_id"].append(examples["id"][sample_index])
        
        # Create offset mapping for this example
        example_offset_mapping = []
        for k, offset in enumerate(tokenized["offset_mapping"][i]):
            if sequence_ids[k] == 1:  # Context part
                example_offset_mapping.append(offset)
            else:  # Question or special tokens - use ignore value
                example_offset_mapping.append((0, 0))  # Use (0,0) instead of None
        
        new_offset_mapping.append(example_offset_mapping)
    
    tokenized["offset_mapping"] = new_offset_mapping
    return tokenized


def tokenize_split(
    raw_ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    config: AppConfig,
    save_dir,
    split_name: str,
    training: bool,
):
    split_dir = save_dir / split_name
    rebuild_incomplete_tokenized_dir(split_dir, force=config.run.force_rebuild_cache)

    if (not config.run.streaming) and tokenized_cache_is_complete(split_dir):
        return load_from_disk(str(split_dir), keep_in_memory=False)

    fn = prepare_train_features if training else prepare_validation_features
    remove_columns = raw_ds.column_names
    tokenized = raw_ds.map(
        lambda batch: fn(batch, tokenizer, config),
        batched=True,
        batch_size=config.run.map_batch_size,
        remove_columns=remove_columns,
        num_proc=config.run.preprocessing_num_proc,
        writer_batch_size=config.run.writer_batch_size,
        load_from_cache_file=not config.run.force_rebuild_cache,
        keep_in_memory=config.dataset.keep_in_memory,
        desc=f"Tokenizing {split_name}",
    )

    if not config.run.streaming:
        split_dir.mkdir(parents=True, exist_ok=True)
        shard_size = dynamic_shard_size_str(tokenized, config.run.preprocess_num_chunks)
        tokenized.save_to_disk(str(split_dir), max_shard_size=shard_size)
        mark_tokenized_cache_complete(split_dir)
        tokenized = load_from_disk(str(split_dir), keep_in_memory=False)
    return tokenized
