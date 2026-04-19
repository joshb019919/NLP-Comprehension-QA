import hashlib
from pathlib import Path

from datasets import load_dataset, load_from_disk
from label import expand_and_label_batch
from tokenize_dataset import (
    prepare_test_features_squad,
    prepare_test_features_triviaqa,
    prepare_train_features_squad,
    prepare_train_features_triviaqa,
    prepare_validation_features_squad,
    prepare_validation_features_triviaqa,
)
from transformers import AutoTokenizer, DataCollatorWithPadding

from common import DEFAULT_DIRS, WorkerExceptionWrapper


def _slug(text: str) -> str:
    return text.replace("/", "--").replace(" ", "_")


def _tokenized_cache_signature(
    dataset_name: str,
    dataset_config_name: str | None,
    tokenizer_name: str,
    max_length: int,
    doc_stride: int,
    revision: str | None,
    split: str,
    mode: str,
) -> str:
    text = (
        f"{dataset_name}|{dataset_config_name}|{tokenizer_name}|"
        f"{max_length}|{doc_stride}|{revision}|{split}|{mode}"
    )
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _tokenized_split_path(
    dataset_name: str,
    dataset_config_name: str | None,
    tokenizer_name: str,
    max_length: int,
    doc_stride: int,
    revision: str | None,
    split: str,
    mode: str,
) -> Path:
    dataset_slug = _slug(
        dataset_name.replace("mandarjoshi/", "") + ("__" + dataset_config_name if dataset_config_name else "")
    )
    tokenizer_slug = _slug(tokenizer_name)
    sig = _tokenized_cache_signature(
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        doc_stride=doc_stride,
        revision=revision,
        split=split,
        mode=mode,
    )
    return DEFAULT_DIRS["tokenized"] / dataset_slug / tokenizer_slug / sig / split


def _find_legacy_triviaqa_full_tokenized_split_path(
    dataset_config_name: str | None,
    tokenizer_name: str,
    split: str,
) -> Path | None:
    dataset_slug = _slug("trivia_qa" + ("__" + dataset_config_name if dataset_config_name else ""))
    tokenizer_slug = _slug(tokenizer_name)
    base_dir = DEFAULT_DIRS["tokenized"] / dataset_slug / tokenizer_slug
    if not base_dir.exists():
        return None

    candidates = []
    for child in sorted(base_dir.iterdir()):
        split_dir = child / split
        if split_dir.is_dir() and (split_dir / "_COMPLETE").exists():
            candidates.append(split_dir)

    if len(candidates) == 1:
        return candidates[0]
    return None


def _filter_tokenized_triviaqa_features(tokenized_ds, selected_example_ids: set[str]):
    if "example_id" not in tokenized_ds.column_names:
        return tokenized_ds
    return tokenized_ds.filter(
        lambda example_id: example_id in selected_example_ids,
        input_columns=["example_id"],
        desc="Filter TriviaQA tokenized features",
    )


def _get_triviaqa_selected_example_ids(expanded_or_raw_ds) -> set[str]:
    if "id" not in expanded_or_raw_ds.column_names:
        return set()
    return set(expanded_or_raw_ds["id"])


def _wrap_map_fn(fn, operation: str):
    return WorkerExceptionWrapper(fn=fn, operation=operation)


def _load_or_build_tokenized_split(
    expanded_or_raw_ds,
    dataset_name: str,
    dataset_config_name: str | None,
    tokenizer_name: str,
    split: str,
    mode: str,
    tokenize_fn,
    tokenize_fn_kwargs: dict,
    tokenize_batch_size: int,
    writer_batch_size: int | None,
    preprocess_num_proc: int,
    max_length: int,
    doc_stride: int,
    revision: str | None,
    keep_in_memory: bool,
):
    preferred_split_path = _tokenized_split_path(
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        doc_stride=doc_stride,
        revision=revision,
        split=split,
        mode=mode,
    )

    cached_split_path = preferred_split_path
    if dataset_name == "mandarjoshi/trivia_qa" and not cached_split_path.exists():
        legacy_split_path = _find_legacy_triviaqa_full_tokenized_split_path(
            dataset_config_name=dataset_config_name,
            tokenizer_name=tokenizer_name,
            split=split,
        )
        if legacy_split_path is not None:
            cached_split_path = legacy_split_path

    if cached_split_path.exists():
        return load_from_disk(str(cached_split_path))

    tokenized_ds = expanded_or_raw_ds.map(
        _wrap_map_fn(tokenize_fn, f"tokenizing {split} split for {mode}"),
        batched=True,
        batch_size=tokenize_batch_size,
        writer_batch_size=writer_batch_size,
        num_proc=preprocess_num_proc,
        fn_kwargs=tokenize_fn_kwargs,
        keep_in_memory=keep_in_memory,
        remove_columns=expanded_or_raw_ds.column_names,
        desc=f"Tokenize {dataset_name} {split} for {mode}",
    )
    preferred_split_path.parent.mkdir(parents=True, exist_ok=True)
    tokenized_ds.save_to_disk(str(preferred_split_path))
    return tokenized_ds


def load_qa_raw_split(
    dataset_name: str,
    dataset_config_name: str | None,
    split: str,
    max_examples: int | None = None,
    seed: int = 42,
    keep_in_memory: bool = False,
):
    if dataset_name == "mandarjoshi/trivia_qa":
        raw_ds = load_dataset(
            dataset_name,
            dataset_config_name or "rc",
            split=split,
            keep_in_memory=keep_in_memory,
        )
    elif dataset_name == "rajpurkar/squad":
        raw_ds = load_dataset("rajpurkar/squad", split=split, keep_in_memory=keep_in_memory)
    elif dataset_name == "rajpurkar/squad_v2":
        raw_ds = load_dataset("rajpurkar/squad_v2", split=split, keep_in_memory=keep_in_memory)
    else:
        raise ValueError(
            f"Unsupported dataset_name={dataset_name!r}. "
            "Expected one of: 'mandarjoshi/trivia_qa', 'rajpurkar/squad', 'rajpurkar/squad_v2'."
        )

    if max_examples is not None:
        subset_size = min(max_examples, len(raw_ds))
        raw_ds = raw_ds.shuffle(seed=seed).select(range(subset_size))

    return raw_ds


def build_expanded_or_raw_split(
    raw_ds,
    dataset_name: str,
    preprocess_num_proc: int,
    flatten_batch_size: int,
    writer_batch_size: int | None,
    keep_in_memory: bool,
):
    if dataset_name == "mandarjoshi/trivia_qa":
        return raw_ds.map(
            _wrap_map_fn(expand_and_label_batch, "expanding and labeling TriviaQA examples"),
            batched=True,
            batch_size=flatten_batch_size,
            writer_batch_size=writer_batch_size,
            num_proc=preprocess_num_proc,
            keep_in_memory=keep_in_memory,
            remove_columns=raw_ds.column_names,
            desc="Expand and label TriviaQA",
        )

    if dataset_name in {"rajpurkar/squad", "rajpurkar/squad_v2"}:
        return raw_ds

    raise ValueError(f"Unsupported dataset_name={dataset_name!r}")


def build_qa_split(
    dataset_name: str,
    dataset_config_name: str | None,
    model_name: str,
    tokenizer_name: str | None,
    split: str,
    mode: str,
    preprocess_num_proc: int = 8,
    flatten_batch_size: int = 64,
    tokenize_batch_size: int = 256,
    writer_batch_size: int | None = None,
    max_length: int = 384,
    doc_stride: int = 128,
    max_examples: int | None = None,
    seed: int = 42,
    limit_after_tokenization: bool = False,
    after_tokenization_limit: int | None = None,
    version_2_with_negative: bool = False,
    cache_dir: str | None = None,
    revision: str | None = None,
    prefer_full_triviaqa_tokenized_cache: bool = True,
    tokenizer=None,
    data_collator=None,
    keep_raw_dataset: bool = True,
    keep_examples_dataset: bool = True,
    keep_in_memory: bool = False,
):
    raw_ds = load_qa_raw_split(
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        split=split,
        max_examples=max_examples,
        seed=seed,
        keep_in_memory=keep_in_memory,
    )

    expanded_or_raw_ds = build_expanded_or_raw_split(
        raw_ds=raw_ds,
        dataset_name=dataset_name,
        preprocess_num_proc=preprocess_num_proc,
        flatten_batch_size=flatten_batch_size,
        writer_batch_size=writer_batch_size,
        keep_in_memory=keep_in_memory,
    )

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or model_name,
            use_fast=True,
            cache_dir=cache_dir,
            revision=revision,
        )

    tokenized_ds = None
    tokenize_fn = None
    tokenize_fn_kwargs = {
        "tokenizer": tokenizer,
        "max_length": max_length,
        "doc_stride": doc_stride,
    }
    tokenization_purpose = {
        "train": "training",
        "validation": "validation",
        "test": "test",
    }.get(mode, mode)

    if dataset_name == "mandarjoshi/trivia_qa":
        if mode == "train":
            tokenize_fn = prepare_train_features_triviaqa
        elif mode == "validation":
            tokenize_fn = prepare_validation_features_triviaqa
        elif mode == "test":
            tokenize_fn = prepare_test_features_triviaqa
    elif dataset_name in {"rajpurkar/squad", "rajpurkar/squad_v2"}:
        tokenize_fn_kwargs["version_2_with_negative"] = version_2_with_negative
        if mode == "train":
            tokenize_fn = prepare_train_features_squad
        elif mode == "validation":
            tokenize_fn = prepare_validation_features_squad
        elif mode == "test":
            tokenize_fn = prepare_test_features_squad
    else:
        raise ValueError(f"Unsupported dataset_name={dataset_name!r}")

    if tokenize_fn is not None:
        if dataset_name == "mandarjoshi/trivia_qa" and prefer_full_triviaqa_tokenized_cache:
            tokenized_ds = _load_or_build_tokenized_split(
                expanded_or_raw_ds=expanded_or_raw_ds,
                dataset_name=dataset_name,
                dataset_config_name=dataset_config_name,
                tokenizer_name=tokenizer_name or model_name,
                split=split,
                mode=mode,
                tokenize_fn=tokenize_fn,
                tokenize_fn_kwargs=tokenize_fn_kwargs,
                tokenize_batch_size=tokenize_batch_size,
                writer_batch_size=writer_batch_size,
                preprocess_num_proc=preprocess_num_proc,
                max_length=max_length,
                doc_stride=doc_stride,
                revision=revision,
                keep_in_memory=keep_in_memory,
            )
            selected_example_ids = _get_triviaqa_selected_example_ids(expanded_or_raw_ds)
            tokenized_ds = _filter_tokenized_triviaqa_features(tokenized_ds, selected_example_ids)
        elif dataset_name in {"rajpurkar/squad", "rajpurkar/squad_v2"}:
            tokenized_ds = _load_or_build_tokenized_split(
                expanded_or_raw_ds=expanded_or_raw_ds,
                dataset_name=dataset_name,
                dataset_config_name=dataset_config_name,
                tokenizer_name=tokenizer_name or model_name,
                split=split,
                mode=mode,
                tokenize_fn=tokenize_fn,
                tokenize_fn_kwargs=tokenize_fn_kwargs,
                tokenize_batch_size=tokenize_batch_size,
                writer_batch_size=writer_batch_size,
                preprocess_num_proc=preprocess_num_proc,
                max_length=max_length,
                doc_stride=doc_stride,
                revision=revision,
                keep_in_memory=keep_in_memory,
            )
        else:
            tokenized_ds = expanded_or_raw_ds.map(
                _wrap_map_fn(tokenize_fn, f"tokenizing {split} split for {tokenization_purpose}"),
                batched=True,
                batch_size=tokenize_batch_size,
                writer_batch_size=writer_batch_size,
                num_proc=preprocess_num_proc,
                fn_kwargs=tokenize_fn_kwargs,
                keep_in_memory=keep_in_memory,
                remove_columns=expanded_or_raw_ds.column_names,
                desc=f"Tokenize {dataset_name} {split} for {tokenization_purpose}",
            )
        if limit_after_tokenization and after_tokenization_limit is not None:
            sample_size = min(after_tokenization_limit, len(tokenized_ds))
            tokenized_ds = tokenized_ds.shuffle(seed=seed).select(range(sample_size))
        if dataset_name == "mandarjoshi/trivia_qa" and mode == "train" and "example_id" in tokenized_ds.column_names:
            tokenized_ds = tokenized_ds.remove_columns(["example_id"])

    if data_collator is None:
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
        )

    returned_raw_ds = raw_ds if keep_raw_dataset else None
    returned_examples_ds = expanded_or_raw_ds if keep_examples_dataset else None

    return returned_raw_ds, returned_examples_ds, tokenized_ds, tokenizer, data_collator
