import argparse
import re
from pathlib import Path
from time import perf_counter

from bench import BenchCallback, BenchLogger
from common import (
    atomic_save_json,
    configure_runtime,
    maybe_set_process_memory_limit,
)
from config import load_app_config
from evaluation import run_postprocessed_eval
from paths import get_run_paths
from qa_trainer import GradientValueClippingCallback, QATrainer
from train import build_qa_split
from transformers import AutoModelForQuestionAnswering, TrainingArguments, set_seed


PROCESS_START_SECONDS = perf_counter()
CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


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


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_RE.match(path.name)
    if match is None:
        return -1
    return int(match.group(1))


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = [
        path
        for path in output_dir.iterdir()
        if path.is_dir() and checkpoint_step(path) >= 0
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=checkpoint_step)


def checkpoint_payload(path: Path | None) -> dict[str, str | int | None]:
    if path is None:
        return {"path": None, "name": None, "step": None}
    return {
        "path": str(path),
        "name": path.name,
        "step": checkpoint_step(path),
    }


def write_variant_config(config, variant_dir: Path, variant_payload: dict) -> None:
    atomic_save_json(config.to_dict(), variant_dir / "resolved_config.json")
    atomic_save_json(variant_payload, variant_dir / "variant.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune HF extractive QA models.")
    parser.add_argument("--run-config", required=True, help="Path to run config JSON.")
    parser.add_argument("--set", action="append", default=[], help="Override config values.")
    return parser.parse_args()


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
    train_features = None
    valid_features_for_predict = None
    raw_valid = None
    valid_examples = None
    valid_features_for_postprocess = None
    raw_test = None
    test_examples = None
    test_features_for_postprocess = None
    bench_logger = None
    train_results = None
    train_metrics = None
    variant_metrics = {}

    try:
        configure_runtime(tokenizer_parallelism=config.run.tokenizer_parallelism)
        maybe_set_process_memory_limit(config.run.process_memory_limit_mb)
        set_seed(config.run.seed)

        if config.run.load_best_model_at_end:
            config.run.save_total_limit = 2

        paths = get_run_paths(config)
        paths["output_dir"].mkdir(parents=True, exist_ok=True)
        paths["bench_dir"].mkdir(parents=True, exist_ok=True)
        atomic_save_json(config.to_dict(), paths["resolved_config"])

        num_proc = config.run.preprocessing_num_proc or 0
        logging_strategy = config.run.logging_strategy or config.run.strategy
        evaluation_strategy = config.run.evaluation_strategy or config.run.strategy
        save_strategy = config.run.save_strategy
        if save_strategy is None:
            save_strategy = evaluation_strategy if config.run.load_best_model_at_end else "no"
        step_interval = config.run.steps
        dataloader_num_workers = config.run.dataloader_num_workers
        dataloader_persistent_workers = config.run.dataloader_persistent_workers

        training_args = TrainingArguments(
            output_dir=str(paths["output_dir"]),
            learning_rate=config.run.learning_rate,
            per_device_train_batch_size=config.run.batch_size,
            per_device_eval_batch_size=max(1, config.run.batch_size),
            num_train_epochs=config.run.epochs,
            lr_scheduler_type=config.run.lr_scheduler,
            weight_decay=config.run.weight_decay,
            gradient_accumulation_steps=config.run.gradient_accumulation_steps,
            logging_strategy=logging_strategy,
            eval_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            optim=config.run.optim,
            gradient_checkpointing=config.run.gradient_checkpointing,
            max_grad_norm=config.run.max_grad_norm,
            remove_unused_columns=config.run.remove_unused_columns,
            report_to=config.run.report_to,
            save_total_limit=config.run.save_total_limit,
            dataloader_drop_last=config.run.dataloader_drop_last,
            dataloader_pin_memory=config.run.dataloader_pin_memory,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_persistent_workers=dataloader_persistent_workers,
            dataloader_prefetch_factor=config.run.dataloader_prefetch_factor,
            logging_steps=config.run.logging_steps,
            eval_steps=step_interval if evaluation_strategy == "steps" else None,
            save_steps=step_interval if save_strategy == "steps" else None,
            greater_is_better=config.run.greater_is_better,
            seed=config.run.seed,
            torch_compile=config.run.torch_compile,
            logging_first_step=True,
            tf32=True,
            bf16=True,
            load_best_model_at_end=config.run.load_best_model_at_end,
            metric_for_best_model=config.run.metric_for_best_model,
            label_names=["start_positions", "end_positions"],
        )

        bench_logger = BenchLogger(paths["bench_file"])
        bench_callback = BenchCallback(bench_logger)
        gradient_clip_callback = GradientValueClippingCallback(config.run.max_grad_norm)

        dataset_name, dataset_config_name, version_2_with_negative = resolve_phase_dataset(config, "train")
        valid_dataset_name, valid_dataset_config_name, valid_version_2_with_negative = resolve_phase_dataset(config, "validation")
        test_dataset_name, test_dataset_config_name, test_version_2_with_negative = resolve_phase_dataset(config, "test")
        max_train_examples = config.run.max_train_examples
        if max_train_examples is None:
            max_train_examples = config.run.max_examples
        max_validation_examples = config.run.max_validation_examples
        if max_validation_examples is None:
            max_validation_examples = config.run.max_examples
        max_test_examples = config.run.max_test_examples
        if max_test_examples is None:
            max_test_examples = config.run.max_examples
        after_tokenization_train_limit = config.run.after_tokenization_train_limit
        if after_tokenization_train_limit is None:
            after_tokenization_train_limit = config.run.after_tokenization_limit
        after_tokenization_validation_limit = config.run.after_tokenization_validation_limit
        if after_tokenization_validation_limit is None:
            after_tokenization_validation_limit = config.run.after_tokenization_limit
        after_tokenization_test_limit = config.run.after_tokenization_test_limit
        if after_tokenization_test_limit is None:
            after_tokenization_test_limit = config.run.after_tokenization_limit

        _, _, train_features, tokenizer, data_collator = build_qa_split(
            dataset_name=dataset_name,
            dataset_config_name=dataset_config_name,
            model_name=config.model.model_name_or_path,
            tokenizer_name=config.model.tokenizer_name_or_path,
            split=config.dataset.train_split,
            mode="train",
            preprocess_num_proc=num_proc,
            flatten_batch_size=config.run.map_batch_size,
            tokenize_batch_size=config.run.map_batch_size,
            writer_batch_size=config.run.writer_batch_size,
            max_length=config.run.max_length,
            doc_stride=config.run.doc_stride,
            max_examples=max_train_examples,
            seed=config.run.seed,
            limit_after_tokenization=config.run.limit_after_tokenization,
            after_tokenization_limit=after_tokenization_train_limit,
            version_2_with_negative=version_2_with_negative,
            cache_dir=config.run.data_root + "/models",
            revision=config.model.revision,
            prefer_full_triviaqa_tokenized_cache=config.run.prefer_full_triviaqa_tokenized_cache,
            keep_raw_dataset=False,
            keep_examples_dataset=False,
            keep_in_memory=config.dataset.keep_in_memory,
        )
        if (
            train_features is not None
            and training_args.dataloader_drop_last
            and len(train_features) < training_args.per_device_train_batch_size
        ):
            training_args.dataloader_drop_last = False
            bench_logger.write(
                {
                    "event": "drop_last_disabled_for_small_train_set",
                    "train_examples": len(train_features),
                    "per_device_train_batch_size": training_args.per_device_train_batch_size,
                }
            )
        raw_valid, valid_examples, valid_features_for_postprocess, _, _ = build_qa_split(
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
            tokenizer=tokenizer,
            data_collator=data_collator,
            keep_in_memory=config.dataset.keep_in_memory,
        )
        if valid_dataset_name == "mandarjoshi/trivia_qa":
            removable = [c for c in ["example_id", "offset_mapping", "question_id"] if c in valid_features_for_postprocess.column_names]
            valid_features_for_predict = valid_features_for_postprocess.remove_columns(removable)
        elif valid_dataset_name in {"rajpurkar/squad", "rajpurkar/squad_v2"}:
            valid_features_for_predict = valid_features_for_postprocess.remove_columns(["example_id", "offset_mapping"])
        else:
            raise ValueError(f"Unsupported dataset_name={valid_dataset_name!r}")

        model = AutoModelForQuestionAnswering.from_pretrained(
            config.model.model_name_or_path,
            cache_dir=config.run.data_root + "/models",
            revision=config.model.revision,
            attn_implementation="sdpa"
        )

        trainer = QATrainer(
            model=model,
            args=training_args,
            train_dataset=train_features,
            eval_dataset=valid_features_for_predict,
            data_collator=data_collator,
            dataset_name=valid_dataset_name,
            version_2_with_negative=valid_version_2_with_negative,
            sgd_momentum=config.run.sgd_momentum,
            raw_eval_examples=raw_valid,
            postprocess_eval_examples=valid_examples,
            postprocess_eval_features=valid_features_for_postprocess,
            callbacks=[bench_callback, gradient_clip_callback],
        )

        train_results = trainer.train()

        train_metrics = {k: float(v) for k, v in train_results.metrics.items() if isinstance(v, (int, float))}
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)

        best_checkpoint = (
            Path(trainer.state.best_model_checkpoint)
            if trainer.state.best_model_checkpoint is not None
            else None
        )
        final_checkpoint = latest_checkpoint(paths["output_dir"])
        checkpoint_pointers = {
            "best": checkpoint_payload(best_checkpoint),
            "final": checkpoint_payload(final_checkpoint),
            "same_checkpoint": (
                best_checkpoint is not None
                and final_checkpoint is not None
                and best_checkpoint.resolve() == final_checkpoint.resolve()
            ),
            "hf_best_metric": float(trainer.state.best_metric)
            if trainer.state.best_metric is not None
            else None,
        }
        atomic_save_json(checkpoint_pointers, paths["output_dir"] / "checkpoint_pointers.json")

        if config.dataset.test_split is not None:
            raw_test, test_examples, test_features_for_postprocess, _, _ = build_qa_split(
                dataset_name=test_dataset_name,
                dataset_config_name=test_dataset_config_name,
                model_name=config.model.model_name_or_path,
                tokenizer_name=config.model.tokenizer_name_or_path,
                split=config.dataset.test_split,
                mode="test",
                preprocess_num_proc=num_proc,
                flatten_batch_size=config.run.map_batch_size,
                tokenize_batch_size=config.run.map_batch_size,
                writer_batch_size=config.run.writer_batch_size,
                max_length=config.run.max_length,
                doc_stride=config.run.doc_stride,
                max_examples=max_test_examples,
                seed=config.run.seed,
                limit_after_tokenization=config.run.limit_after_tokenization,
                after_tokenization_limit=after_tokenization_test_limit,
                version_2_with_negative=test_version_2_with_negative,
                cache_dir=config.run.data_root + "/models",
                revision=config.model.revision,
                prefer_full_triviaqa_tokenized_cache=config.run.prefer_full_triviaqa_tokenized_cache,
                tokenizer=tokenizer,
                data_collator=data_collator,
                keep_in_memory=config.dataset.keep_in_memory,
            )

        variants: list[tuple[str, Path | None]] = []
        if best_checkpoint is not None:
            variants.append(("best", best_checkpoint))
        variants.append(("final", final_checkpoint))

        seen_variants: set[str] = set()
        for variant_name, checkpoint in variants:
            if variant_name in seen_variants:
                continue
            seen_variants.add(variant_name)

            variant_dir = paths["output_dir"] / variant_name
            variant_bench_logger = BenchLogger(paths["bench_dir"] / f"{variant_name}.jsonl")
            bench_callback.logger = variant_bench_logger

            variant_payload = {
                "variant": variant_name,
                "checkpoint": checkpoint_payload(checkpoint),
                "checkpoint_pointers": checkpoint_pointers,
            }
            variant_bench_logger.write({"event": "variant_begin", **variant_payload})

            if checkpoint is not None:
                trainer._load_from_checkpoint(str(checkpoint))

            trainer.save_model(str(variant_dir))
            trainer.state.save_to_json(str(variant_dir / "trainer_state.json"))
            write_variant_config(config, variant_dir, variant_payload)

            eval_metrics = run_postprocessed_eval(
                trainer=trainer,
                dataset_name=valid_dataset_name,
                version_2_with_negative=valid_version_2_with_negative,
                raw_examples=raw_valid,
                eval_examples=valid_examples,
                eval_features_for_postprocess=valid_features_for_postprocess,
                prefix="eval",
                output_path=variant_dir,
            )
            variant_bench_logger.write({"event": "eval_postprocessed", "variant": variant_name, **eval_metrics})
            atomic_save_json(eval_metrics, variant_dir / "eval_metrics.json")

            test_metrics = None
            if raw_test is not None and test_examples is not None and test_features_for_postprocess is not None:
                test_metrics = run_postprocessed_eval(
                    trainer=trainer,
                    dataset_name=test_dataset_name,
                    version_2_with_negative=test_version_2_with_negative,
                    raw_examples=raw_test,
                    eval_examples=test_examples,
                    eval_features_for_postprocess=test_features_for_postprocess,
                    prefix="test",
                    output_path=variant_dir,
                )
                variant_bench_logger.write({"event": "test_postprocessed", "variant": variant_name, **test_metrics})
                atomic_save_json(test_metrics, variant_dir / "test_metrics.json")

            variant_metrics[variant_name] = {
                "eval_metrics": eval_metrics,
                "test_metrics": test_metrics,
                "checkpoint": checkpoint_payload(checkpoint),
            }
            atomic_save_json(
                {
                    "variant": variant_name,
                    "train_metrics": train_metrics,
                    "eval_metrics": eval_metrics,
                    "test_metrics": test_metrics,
                    "checkpoint": checkpoint_payload(checkpoint),
                },
                variant_dir / "summary_metrics.json",
            )
            variant_bench_logger.write(
                {
                    "event": "run_complete",
                    "variant": variant_name,
                    "status": "success",
                    "total_wall_time_seconds": float(perf_counter() - PROCESS_START_SECONDS),
                }
            )

        bench_callback.logger = bench_logger

        total_wall_time_seconds = float(perf_counter() - PROCESS_START_SECONDS)
        atomic_save_json(
            {
                "train_metrics": train_metrics,
                "variant_metrics": variant_metrics,
                "checkpoint_pointers": checkpoint_pointers,
                "total_wall_time_seconds": total_wall_time_seconds,
            },
            paths["output_dir"] / "summary_metrics.json",
        )

        bench_logger.write(
            {
                "event": "run_complete",
                "status": "success",
                "total_wall_time_seconds": total_wall_time_seconds,
            }
        )
    finally:
        pass


if __name__ == "__main__":
    main()
