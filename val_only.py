#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common import atomic_save_json, configure_runtime, maybe_set_process_memory_limit, release_memory, teardown_trainer
from config import load_app_config, load_json
from evaluation import run_postprocessed_eval
from paths import get_run_paths
from qa_trainer import QATrainer
from train import build_qa_split
from transformers import AutoModelForQuestionAnswering, TrainingArguments, set_seed


PROCESS_START_SECONDS = perf_counter()


def infer_version_2_with_negative(dataset_name: str) -> bool:
    return dataset_name == "rajpurkar/squad_v2"


def resolve_phase_dataset(config, phase: str) -> tuple[str, str | None, bool]:
    dataset_name = config.dataset.dataset_name
    dataset_config_name = config.dataset.dataset_config_name
    version_2_with_negative = config.dataset.version_2_with_negative

    if phase == "validation":
        override_dataset_name = config.dataset.validation_dataset_name
        dataset_name = override_dataset_name or dataset_name
        dataset_config_name = (
            config.dataset.validation_dataset_config_name
            if override_dataset_name is not None
            else dataset_config_name
        )
        if config.dataset.validation_version_2_with_negative is not None:
            version_2_with_negative = config.dataset.validation_version_2_with_negative
        elif override_dataset_name is not None:
            version_2_with_negative = infer_version_2_with_negative(dataset_name)
    elif phase == "test":
        override_dataset_name = config.dataset.test_dataset_name
        dataset_name = override_dataset_name or dataset_name
        dataset_config_name = (
            config.dataset.test_dataset_config_name
            if override_dataset_name is not None
            else dataset_config_name
        )
        if config.dataset.test_version_2_with_negative is not None:
            version_2_with_negative = config.dataset.test_version_2_with_negative
        elif override_dataset_name is not None:
            version_2_with_negative = infer_version_2_with_negative(dataset_name)

    return dataset_name, dataset_config_name, version_2_with_negative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run validation only and overwrite final eval outputs.")
    parser.add_argument("--run-config", required=True, help="Path to run config JSON.")
    parser.add_argument("--set", action="append", default=[], help="Override config values.")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional path to a saved checkpoint directory. Defaults to the resolved run output_dir.",
    )
    return parser.parse_args()


def load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    return load_json(path)


def overwrite_eval_outputs(output_dir: Path, eval_metrics: dict[str, float]) -> None:
    eval_results_path = output_dir / "eval_results.json"
    all_results_path = output_dir / "all_results.json"
    summary_metrics_path = output_dir / "summary_metrics.json"

    atomic_save_json(eval_metrics, eval_results_path)

    all_results = load_json_if_exists(all_results_path)
    all_results = {key: value for key, value in all_results.items() if not key.startswith("eval_")}
    all_results.update(eval_metrics)
    atomic_save_json(all_results, all_results_path)

    summary_metrics = load_json_if_exists(summary_metrics_path)
    summary_metrics["eval_metrics"] = eval_metrics
    summary_metrics.setdefault("total_wall_time_seconds", float(perf_counter() - PROCESS_START_SECONDS))
    atomic_save_json(summary_metrics, summary_metrics_path)


def main() -> None:
    cli_args = parse_args()

    overrides = []
    for item in cli_args.set:
        overrides.append(item if "." in item else f"run.{item}")

    config = load_app_config(cli_args.run_config, overrides=overrides)

    trainer = None
    model = None
    tokenizer = None
    data_collator = None
    valid_features_for_predict = None
    raw_valid = None
    valid_examples = None
    valid_features_for_postprocess = None

    try:
        configure_runtime(tokenizer_parallelism=config.run.tokenizer_parallelism)
        maybe_set_process_memory_limit(config.run.process_memory_limit_mb)
        set_seed(config.run.seed)

        paths = get_run_paths(config)
        output_dir = paths["output_dir"]
        model_path = Path(cli_args.model_path) if cli_args.model_path else output_dir

        num_proc = config.run.preprocessing_num_proc or 0
        valid_dataset_name, valid_dataset_config_name, valid_version_2_with_negative = resolve_phase_dataset(config, "validation")
        max_validation_examples = config.run.max_validation_examples
        if max_validation_examples is None:
            max_validation_examples = config.run.max_examples
        after_tokenization_validation_limit = config.run.after_tokenization_validation_limit
        if after_tokenization_validation_limit is None:
            after_tokenization_validation_limit = config.run.after_tokenization_limit

        raw_valid, valid_examples, valid_features_for_postprocess, tokenizer, data_collator = build_qa_split(
            dataset_name=valid_dataset_name,
            dataset_config_name=valid_dataset_config_name,
            model_name=config.model.model_name_or_path,
            tokenizer_name=config.model.tokenizer_name_or_path,
            split=config.dataset.validation_split,
            mode="validation",
            preprocess_num_proc=num_proc,
            flatten_batch_size=config.run.map_batch_size,
            tokenize_batch_size=config.run.map_batch_size,
            writer_batch_size=config.run.writer_batch_size,
            max_length=config.run.max_length,
            doc_stride=config.run.doc_stride,
            max_examples=max_validation_examples,
            seed=config.run.seed,
            limit_after_tokenization=config.run.limit_after_tokenization,
            after_tokenization_limit=after_tokenization_validation_limit,
            version_2_with_negative=valid_version_2_with_negative,
            cache_dir=config.run.data_root + "/models",
            revision=config.model.revision,
            prefer_full_triviaqa_tokenized_cache=config.run.prefer_full_triviaqa_tokenized_cache,
            keep_in_memory=config.dataset.keep_in_memory,
        )
        release_memory()

        if valid_dataset_name == "mandarjoshi/trivia_qa":
            removable = [c for c in ["example_id", "offset_mapping", "question_id"] if c in valid_features_for_postprocess.column_names]
            valid_features_for_predict = valid_features_for_postprocess.remove_columns(removable)
        elif valid_dataset_name in {"rajpurkar/squad", "rajpurkar/squad_v2"}:
            valid_features_for_predict = valid_features_for_postprocess.remove_columns(["example_id", "offset_mapping"])
        else:
            raise ValueError(f"Unsupported dataset_name={valid_dataset_name!r}")

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=config.run.batch_size,
            per_device_eval_batch_size=max(1, config.run.batch_size),
            dataloader_drop_last=False,
            dataloader_pin_memory=config.run.dataloader_pin_memory,
            dataloader_num_workers=config.run.dataloader_num_workers,
            dataloader_persistent_workers=config.run.dataloader_persistent_workers,
            dataloader_prefetch_factor=config.run.dataloader_prefetch_factor,
            remove_unused_columns=config.run.remove_unused_columns,
            report_to="none",
            bf16=True,
            tf32=True,
            prediction_loss_only=False,
            label_names=["start_positions", "end_positions"],
        )

        model = AutoModelForQuestionAnswering.from_pretrained(
            str(model_path),
            cache_dir=config.run.data_root + "/models",
            revision=config.model.revision,
            attn_implementation="sdpa",
        )

        trainer = QATrainer(
            model=model,
            args=training_args,
            train_dataset=None,
            eval_dataset=valid_features_for_predict,
            data_collator=data_collator,
            dataset_name=valid_dataset_name,
            version_2_with_negative=valid_version_2_with_negative,
            sgd_momentum=config.run.sgd_momentum,
            raw_eval_examples=raw_valid,
            postprocess_eval_examples=valid_examples,
            postprocess_eval_features=valid_features_for_postprocess,
            callbacks=[],
        )

        eval_metrics = run_postprocessed_eval(
            trainer=trainer,
            dataset_name=valid_dataset_name,
            version_2_with_negative=valid_version_2_with_negative,
            raw_examples=raw_valid,
            eval_examples=valid_examples,
            eval_features_for_postprocess=valid_features_for_postprocess,
            prefix="eval",
        )
        overwrite_eval_outputs(output_dir, eval_metrics)

        print(f"Validation-only metrics written to {output_dir}")
        print(eval_metrics)
    finally:
        teardown_trainer(trainer)
        trainer = None
        model = None
        tokenizer = None
        data_collator = None
        valid_features_for_predict = None
        raw_valid = None
        valid_examples = None
        valid_features_for_postprocess = None
        release_memory()
        release_memory()


if __name__ == "__main__":
    main()
