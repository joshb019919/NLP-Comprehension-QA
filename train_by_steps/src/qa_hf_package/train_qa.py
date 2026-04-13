from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, disable_caching
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from src.bench import BenchCallback, BenchLogger
from src.common import atomic_save_json, configure_runtime, get_storage_root, maybe_set_process_memory_limit
from src.config import AppConfig, load_app_config
from src.datasets import prepare_raw_qa_splits
from src.metrics import QAMetricBundle
from src.paths import get_run_paths
from src.postprocess import postprocess_qa_predictions
from src.preprocess import tokenize_split


PROCESS_START_SECONDS = time.perf_counter()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune HF extractive QA models conservatively.")
    parser.add_argument("--run-config", required=True, help="Path to run config JSON.")
    parser.add_argument("--set", action="append", default=[], help="Override config values, e.g. run.batch_size=8 or batch_size=8.")
    return parser.parse_args()



def build_training_args(config: AppConfig, paths: dict[str, Path], has_eval_labels: bool) -> TrainingArguments:
    use_best_model = config.run.load_best_model_at_end
    metric_for_best_model: str | None = config.run.metric_for_best_model
    if use_best_model and metric_for_best_model == "eval_loss" and not has_eval_labels:
        metric_for_best_model = "eval_f1"

    greater_is_better = config.run.greater_is_better
    if use_best_model and metric_for_best_model in {"eval_f1", "eval_exact"}:
        greater_is_better = True

    if not use_best_model:
        metric_for_best_model = None

    return TrainingArguments(
        output_dir=str(paths["output_dir"]),
        # output_dir=False,
        learning_rate=config.run.learning_rate,
        per_device_train_batch_size=config.run.batch_size,
        per_device_eval_batch_size=max(1, min(config.run.batch_size, 16)),
        num_train_epochs=config.run.epochs,
        lr_scheduler_type=config.run.lr_scheduler,
        weight_decay=config.run.weight_decay,
        gradient_accumulation_steps=config.run.gradient_accumulation_steps,
        eval_strategy=config.run.evaluation_strategy,
        eval_steps=config.run.eval_steps,
        save_strategy=config.run.save_strategy,
        save_steps=config.run.save_steps,
        optim=config.run.optim,
        gradient_checkpointing=config.run.gradient_checkpointing,
        max_grad_norm=config.run.max_grad_norm,
        remove_unused_columns=config.run.remove_unused_columns,
        report_to=config.run.report_to,
        save_total_limit=config.run.save_total_limit,
        dataloader_drop_last=config.run.dataloader_drop_last,
        dataloader_pin_memory=config.run.dataloader_pin_memory,
        dataloader_num_workers=config.run.dataloader_num_workers,
        dataloader_persistent_workers=config.run.dataloader_persistent_workers,
        dataloader_prefetch_factor=config.run.dataloader_prefetch_factor,
        logging_steps=config.run.logging_steps,
        greater_is_better=greater_is_better,
        seed=config.run.seed,
        torch_compile=config.run.torch_compile,
        logging_first_step=True,
        tf32=True,
        bf16=True,
        load_best_model_at_end=use_best_model,
        metric_for_best_model=metric_for_best_model,
        label_names=["start_positions", "end_positions"]
    )


class QATrainer(Trainer):
    def __init__(
        self,
        *args,
        postprocess_eval_examples: Dataset | None = None,
        postprocess_eval_features: Dataset | None = None,
        app_config: AppConfig | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._postprocess_eval_examples = postprocess_eval_examples
        self._postprocess_eval_features = postprocess_eval_features
        self._app_config = app_config

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        if (
            metric_key_prefix == "eval"
            and self._postprocess_eval_examples is not None
            and self._postprocess_eval_features is not None
            and self._app_config is not None
        ):
            post_metrics = run_postprocessed_eval(
                trainer=self,
                raw_examples=self._postprocess_eval_examples,
                eval_features=self._postprocess_eval_features,
                config=self._app_config,
                prefix=metric_key_prefix,
            )
            metrics.update(post_metrics)
        return metrics



def safe_load_tokenizer_and_model(config: AppConfig, storage_root: Path):
    model_name = config.model.model_name_or_path
    tokenizer_name = config.model.tokenizer_name_or_path or model_name
    common_kwargs = {
        "cache_dir": str(storage_root / "models"),
        "revision": config.model.revision
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=config.model.use_fast_tokenizer,
            **common_kwargs,
        )
        model = AutoModelForQuestionAnswering.from_pretrained(model_name, **common_kwargs)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=config.model.use_fast_tokenizer,
            force_download=True,
            attn_imiplementation="sdpa",
            **common_kwargs
        )
        model = AutoModelForQuestionAnswering.from_pretrained(model_name, force_download=True, **common_kwargs)
    return tokenizer, model



def has_reference_answers(ds: Dataset) -> bool:
    if len(ds) == 0:
        return False
    answers = ds[0].get("answers")
    return isinstance(answers, dict) and "text" in answers



def run_postprocessed_eval(
    trainer: Trainer,
    raw_examples: Dataset,
    eval_features: Dataset,
    config: AppConfig,
    prefix: str,
) -> dict[str, float]:
    # Keep only model inputs/labels for the dataloader; retain full features for postprocessing.
    eval_features_for_predict = cast(Dataset, eval_features.remove_columns(["example_id", "offset_mapping"]))

    # Ensure eval/predict does not drop the last partial batch.
    original_drop_last = trainer.args.dataloader_drop_last
    trainer.args.dataloader_drop_last = False
    try:
        pred_output = trainer.predict(cast(Any, eval_features_for_predict), metric_key_prefix=prefix)
    finally:
        trainer.args.dataloader_drop_last = original_drop_last

    if not isinstance(pred_output.predictions, tuple) or len(pred_output.predictions) != 2:
        raise ValueError("Expected QA predictions as (start_logits, end_logits).")

    predictions_tuple = cast(tuple[Any, Any], pred_output.predictions)
    all_start_logits, all_end_logits = predictions_tuple
    aligned_count = min(len(eval_features), len(all_start_logits), len(all_end_logits))
    if aligned_count != len(eval_features):
        eval_features = eval_features.select(range(aligned_count))
    raw_predictions = (all_start_logits[:aligned_count], all_end_logits[:aligned_count])

    predictions, references = postprocess_qa_predictions(
        examples=raw_examples,
        features=eval_features,
        raw_predictions=raw_predictions,
        version_2_with_negative=config.dataset.version_2_with_negative,
        null_score_diff_threshold=config.dataset.null_score_diff_threshold,
    )

    metric_bundle = QAMetricBundle(version_2_with_negative=config.dataset.version_2_with_negative)
    qa_scores = metric_bundle.compute(predictions=predictions, references=references)

    combined = {f"{prefix}_{k}": v for k, v in qa_scores.items()}
    for key, value in (pred_output.metrics or {}).items():
        if isinstance(value, (int, float)):
            combined[key] = float(value)
    return combined


def main() -> None:
    args = parse_args()
    normalized = []
    for item in args.set:
        if "." not in item:
            normalized.append(f"run.{item}")
        else:
            normalized.append(item)
    config = load_app_config(args.run_config, overrides=normalized)

    storage_root = get_storage_root(
        use_data_root=config.run.use_data_root,
        data_root=config.run.data_root,
        storage_subfolder=config.run.storage_subfolder,
    )
    configure_runtime(storage_root=storage_root, tokenizer_parallelism=config.run.tokenizer_parallelism)
    maybe_set_process_memory_limit(config.run.process_memory_limit_mb)
    set_seed(config.run.seed)

    paths = get_run_paths(config)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    paths["bench_dir"].mkdir(parents=True, exist_ok=True)
    atomic_save_json(config.to_dict(), paths["resolved_config"])

    bench_logger = BenchLogger(paths["bench_file"])
    bench_callback = BenchCallback(bench_logger)
    run_status = "success"
    run_error: dict[str, str] | None = None

    try:
        tokenizer, model = safe_load_tokenizer_and_model(config, storage_root)

        train_raw, valid_raw, test_raw = prepare_raw_qa_splits(config)

        if config.run.streaming:
            raise NotImplementedError(
                "streaming=True is exposed as a config knob, but this package's default Trainer path is implemented for disk-backed datasets with streaming=False."
            )

        train_features = cast(Dataset, tokenize_split(train_raw, tokenizer, config, paths["tokenized_dir"], "train", training=True))
        valid_features = cast(Dataset, tokenize_split(valid_raw, tokenizer, config, paths["tokenized_dir"], "validation", training=False))
        test_features = None
        if test_raw is not None:
            test_features = cast(Dataset, tokenize_split(test_raw, tokenizer, config, paths["tokenized_dir"], "test", training=False))

        data_collator = default_data_collator
        train_features_for_trainer = train_features.remove_columns(
            "example_id"
        )
        print("eval cols: ", valid_features.column_names)
        valid_features_for_trainer = valid_features.remove_columns(
            ["example_id", "offset_mapping"]
        )
        if not config.run.remove_unused_columns:
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=None)

        has_eval_labels = {"start_positions", "end_positions"}.issubset(set(valid_features_for_trainer.column_names))
        if config.run.load_best_model_at_end and config.run.metric_for_best_model == "eval_loss" and not has_eval_labels:
            print("WARNING: eval_loss unavailable (missing eval labels); using eval_f1 from postprocessed metrics for best-model selection.")

        training_args = build_training_args(config, paths, has_eval_labels=has_eval_labels)

        # Make sure your validation dataset has labels
        print("Validation columns:", valid_features_for_trainer.column_names)
        if "start_positions" not in valid_features_for_trainer.column_names:
            print("ERROR: Validation dataset missing labels!")

        trainer = QATrainer(
            model=model,
            args=training_args,
            train_dataset=train_features_for_trainer,
            eval_dataset=valid_features_for_trainer,
            data_collator=data_collator,
            postprocess_eval_examples=valid_raw,
            postprocess_eval_features=valid_features,
            app_config=config,
            callbacks=[bench_callback]
        )

        print("trainer label names:", trainer.label_names)

        resume_from_checkpoint: str | None = None
        if config.run.auto_resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(str(paths["output_dir"]))
            if resume_from_checkpoint is not None:
                print(f"Resuming from checkpoint: {resume_from_checkpoint}")

        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model()
        trainer.save_state()

        train_metrics = {k: float(v) for k, v in train_result.metrics.items() if isinstance(v, (int, float))}
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)

        eval_metrics = run_postprocessed_eval(trainer, valid_raw, valid_features, config, prefix="eval")
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        bench_logger.write({"event": "eval_postprocessed", **eval_metrics})

        if test_raw is not None and test_features is not None:
            test_metrics = run_postprocessed_eval(trainer, test_raw, test_features, config, prefix="test")
            trainer.log_metrics("test", test_metrics)
            trainer.save_metrics("test", test_metrics)
            bench_logger.write({"event": "test_postprocessed", **test_metrics})

        total_wall_time_seconds = float(time.perf_counter() - PROCESS_START_SECONDS)

        atomic_save_json(
            {
                "train_metrics": train_metrics,
                "eval_metrics": eval_metrics,
                "total_wall_time_seconds": total_wall_time_seconds,
            },
            paths["output_dir"] / "summary_metrics.json",
        )
    except Exception as exc:
        run_status = "failed"
        run_error = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        raise
    finally:
        payload: dict[str, Any] = {
            "event": "run_complete",
            "status": run_status,
            "total_wall_time_seconds": float(time.perf_counter() - PROCESS_START_SECONDS),
        }
        if run_error is not None:
            payload.update(run_error)
        bench_logger.write(payload)


if __name__ == "__main__":
    main()
