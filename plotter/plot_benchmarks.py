#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


EXPERIMENT_RE = re.compile(r"exp0?(\d+)", re.IGNORECASE)

EVAL_SERIES_SPECS = [
    ("exact", ("eval_exact", "eval_exact_match"), "Exact"),
    ("f1", ("eval_f1",), "F1"),
]

OPTIONAL_ANS_SERIES_SPECS = [
    ("hasans_exact", ("eval_HasAns_exact",), "HasAns Exact"),
    ("hasans_f1", ("eval_HasAns_f1",), "HasAns F1"),
    ("noans_exact", ("eval_NoAns_exact",), "NoAns Exact"),
    ("noans_f1", ("eval_NoAns_f1",), "NoAns F1"),
]

FINAL_EVAL_METRIC_SPECS = [
    ("exact", ("eval_exact", "eval_exact_match"), "Exact"),
    ("f1", ("eval_f1",), "F1"),
    ("hasans_exact", ("eval_HasAns_exact",), "HasAns Exact"),
    ("hasans_f1", ("eval_HasAns_f1",), "HasAns F1"),
    ("noans_exact", ("eval_NoAns_exact",), "NoAns Exact"),
    ("noans_f1", ("eval_NoAns_f1",), "NoAns F1"),
]

FINAL_TEST_METRIC_SPECS = [
    ("exact", ("test_exact", "test_exact_match"), "Exact"),
    ("f1", ("test_f1",), "F1"),
    ("hasans_exact", ("test_HasAns_exact",), "HasAns Exact"),
    ("hasans_f1", ("test_HasAns_f1",), "HasAns F1"),
    ("noans_exact", ("test_NoAns_exact",), "NoAns Exact"),
    ("noans_f1", ("test_NoAns_f1",), "NoAns F1"),
]


@dataclass
class ExperimentMeta:
    number: int
    run_name: str
    run_config_path: Path | None
    overrides: list[str]
    model_name: str
    train_dataset: str
    eval_dataset: str
    test_dataset: str | None
    epochs: int | float | None
    learning_rate: float | None
    weight_decay: float | None
    max_grad_norm: float | None
    optimizer: str | None
    null_score_diff_threshold: float | None
    max_length: int | None
    doc_stride: int | None
    batch_size: int | None
    after_tokenization_limit: int | None
    after_tokenization_train_limit: int | None
    after_tokenization_validation_limit: int | None
    after_tokenization_test_limit: int | None
    seed: int | None

    @property
    def short_name(self) -> str:
        return f"exp{self.number:02d}"

    @property
    def eval_subtitle(self) -> str:
        return (
            f"Model: {humanize_name(self.model_name)} | "
            f"Train: {humanize_name(self.train_dataset)} | "
            f"Eval: {humanize_name(self.eval_dataset)} | "
            f"Ep: {format_value(self.epochs)} | "
            f"LR: {format_value(self.learning_rate)} | "
            f"WD: {format_value(self.weight_decay)}"
        )

    @property
    def test_subtitle(self) -> str:
        dataset = self.test_dataset or self.eval_dataset
        return (
            f"Model: {humanize_name(self.model_name)} | "
            f"Train: {humanize_name(self.train_dataset)} | "
            f"Test: {humanize_name(dataset)} | "
            f"Ep: {format_value(self.epochs)} | "
            f"LR: {format_value(self.learning_rate)} | "
            f"WD: {format_value(self.weight_decay)}"
        )

    def key_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": humanize_name(self.model_name),
            "train": humanize_name(self.train_dataset),
            "eval": humanize_name(self.eval_dataset),
            "test": humanize_name(self.test_dataset) if self.test_dataset else None,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "optimizer": self.optimizer,
            "max_length": self.max_length,
            "doc_stride": self.doc_stride,
            "batch_size": self.batch_size,
            "after_tokenization_limit": self.after_tokenization_limit,
            "after_tokenization_train_limit": self.after_tokenization_train_limit,
            "after_tokenization_validation_limit": self.after_tokenization_validation_limit,
            "after_tokenization_test_limit": self.after_tokenization_test_limit,
            "seed": self.seed,
        }
        if self.train_dataset == "rajpurkar/squad_v2" or self.eval_dataset == "rajpurkar/squad_v2":
            payload["null_score_diff_threshold"] = self.null_score_diff_threshold
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot benchmark JSONL files for all experiments.")
    parser.add_argument(
        "--bench-root",
        type=Path,
        default=None,
        help="Path to the bench directory. If omitted, common locations are auto-detected.",
    )
    parser.add_argument(
        "--run-script",
        type=Path,
        default=Path("run_experiments.sh"),
        help="Path to run_experiments.sh for experiment metadata.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results"),
        help="Directory where plot outputs should be written.",
    )
    return parser.parse_args()


def detect_bench_root(explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            Path("/data/data/huggingface/bench"),
            Path.home() / "data/huggingface/bench",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find a benchmark root. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def parse_cli_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def apply_override(mapping: dict[str, Any], dotted_key: str, raw_value: str) -> None:
    cursor = mapping
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = parse_cli_value(raw_value)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_config_path(raw_path: str, base_dir: Path | None = None) -> Path:
    candidates: list[Path] = []
    raw = Path(raw_path)
    if raw.is_absolute():
        candidates.append(raw)
    if base_dir is not None:
        candidates.append(base_dir / raw)
    candidates.append(raw)
    candidates.append(Path("src") / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve config path: {raw_path}")


def load_run_metadata(run_script_path: Path) -> dict[int, ExperimentMeta]:
    lines = run_script_path.read_text(encoding="utf-8").splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.lstrip()
        if "run_exp " not in stripped:
            i += 1
            continue
        block_lines = []
        while i < len(lines):
            candidate = lines[i].strip()
            if candidate.startswith("#"):
                candidate = candidate[1:].lstrip()
            block_lines.append(candidate.rstrip("\\").rstrip())
            if not lines[i].rstrip().endswith("\\"):
                break
            i += 1
        blocks.append(" ".join(part for part in block_lines if part))
        i += 1

    metadata: dict[int, ExperimentMeta] = {}
    for block in blocks:
        tokens = shlex.split(block)
        if len(tokens) < 3 or tokens[0] != "run_exp":
            continue
        run_name = tokens[1]
        run_config_path = resolve_config_path(tokens[2], run_script_path.parent)
        match = EXPERIMENT_RE.search(run_name)
        if not match:
            continue
        exp_number = int(match.group(1))
        overrides: list[str] = []
        idx = 3
        while idx < len(tokens):
            if tokens[idx] == "--set" and idx + 1 < len(tokens):
                overrides.append(tokens[idx + 1])
                idx += 2
            else:
                idx += 1

        run_cfg = read_json(run_config_path)
        model_cfg = read_json(resolve_config_path(run_cfg["model_config_path"], run_config_path.parent))
        dataset_cfg = read_json(resolve_config_path(run_cfg["dataset_config_path"], run_config_path.parent))
        merged = {
            "model": model_cfg,
            "dataset": dataset_cfg,
            "run": {k: v for k, v in run_cfg.items() if k not in {"model_config_path", "dataset_config_path"}},
        }
        for override in overrides:
            dotted_key, raw_value = override.split("=", 1)
            apply_override(merged, dotted_key, raw_value)

        model_name = merged["model"].get("model_name_or_path", "unknown")
        train_dataset = merged["dataset"].get("dataset_name", "unknown")
        eval_dataset = merged["dataset"].get("validation_dataset_name") or merged["dataset"].get("dataset_name", "unknown")
        test_dataset = merged["dataset"].get("test_dataset_name")
        metadata[exp_number] = ExperimentMeta(
            number=exp_number,
            run_name=run_name,
            run_config_path=run_config_path,
            overrides=overrides,
            model_name=model_name,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            test_dataset=test_dataset,
            epochs=merged["run"].get("epochs"),
            learning_rate=merged["run"].get("learning_rate"),
            weight_decay=merged["run"].get("weight_decay"),
            max_grad_norm=merged["run"].get("max_grad_norm"),
            optimizer=merged["run"].get("optim"),
            null_score_diff_threshold=merged["dataset"].get("null_score_diff_threshold"),
            max_length=merged["run"].get("max_length"),
            doc_stride=merged["run"].get("doc_stride"),
            batch_size=merged["run"].get("batch_size"),
            after_tokenization_limit=merged["run"].get("after_tokenization_limit"),
            after_tokenization_train_limit=merged["run"].get("after_tokenization_train_limit"),
            after_tokenization_validation_limit=merged["run"].get("after_tokenization_validation_limit"),
            after_tokenization_test_limit=merged["run"].get("after_tokenization_test_limit"),
            seed=merged["run"].get("seed"),
        )
    return metadata


def humanize_name(raw: str | None) -> str:
    if not raw:
        return "N/A"
    replacements = {
        "bert-base-uncased": "BERT-base",
        "distilbert-base-uncased": "DistilBERT",
        "rajpurkar/squad": "SQuAD 1.1",
        "rajpurkar/squad_v2": "SQuAD 2.0",
        "mandarjoshi/trivia_qa": "TriviaQA",
        "mandarjoshi/trivia_qa/rc": "TriviaQA RC",
        "rc": "RC",
    }
    if raw in replacements:
        return replacements[raw]
    if raw.endswith("/squad"):
        return "SQuAD 1.1"
    if raw.endswith("/squad_v2"):
        return "SQuAD 2.0"
    if raw.endswith("/trivia_qa"):
        return "TriviaQA"
    return raw


def format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def load_bench_events(bench_file: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in bench_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not payload:
            continue
        inner = next(iter(payload.values()))
        events.append(inner)
    return events


def choose_metric_value(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def dedupe_by_step(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[int, dict[str, Any]] = {}
    for row in rows:
        step = row.get("step")
        if not isinstance(step, int):
            continue
        current = deduped.get(step)
        if current is None or float(row.get("elapsed_seconds", -math.inf)) >= float(current.get("elapsed_seconds", -math.inf)):
            deduped[step] = row
    return [deduped[step] for step in sorted(deduped)]


def collect_training_loss(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [event for event in events if event.get("event") == "log" and isinstance(event.get("loss"), (int, float))]
    return dedupe_by_step(rows)


def collect_eval_metric_logs(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        if event.get("event") != "log":
            continue
        if any(key in event for key in ("eval_exact", "eval_exact_match", "eval_f1")):
            rows.append(event)
    return dedupe_by_step(rows)


def find_final_event(events: list[dict[str, Any]], event_name: str) -> dict[str, Any] | None:
    matches = [event for event in events if event.get("event") == event_name]
    if not matches:
        return None
    return max(matches, key=lambda event: float(event.get("elapsed_seconds", -math.inf)))


def find_best_model_restored(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    matches = [event for event in events if event.get("event") == "best_model_restored"]
    if not matches:
        return None
    return max(matches, key=lambda event: float(event.get("elapsed_seconds", -math.inf)))


def build_experiment_lookup(metadata: dict[int, ExperimentMeta], bench_dirs: list[Path]) -> dict[int, tuple[ExperimentMeta, Path]]:
    lookup: dict[int, tuple[ExperimentMeta, Path]] = {}
    for bench_dir in bench_dirs:
        match = EXPERIMENT_RE.search(bench_dir.name)
        if not match:
            continue
        exp_number = int(match.group(1))
        if exp_number in metadata:
            meta = metadata[exp_number]
        else:
            meta = ExperimentMeta(
                number=exp_number,
                run_name=bench_dir.name,
                run_config_path=None,
                overrides=[],
                model_name="unknown",
                train_dataset="unknown",
                eval_dataset="unknown",
                test_dataset=None,
                epochs=None,
                learning_rate=None,
                weight_decay=None,
                max_grad_norm=None,
                optimizer=None,
                null_score_diff_threshold=None,
                max_length=None,
                doc_stride=None,
                batch_size=None,
                after_tokenization_limit=None,
                after_tokenization_train_limit=None,
                after_tokenization_validation_limit=None,
                after_tokenization_test_limit=None,
                seed=None,
            )
        lookup[exp_number] = (meta, bench_dir)
    return dict(sorted(lookup.items()))


def set_common_axis_format(ax: plt.Axes, max_step: int) -> None:
    upper = int(math.ceil(max_step / 100.0) * 100) if max_step > 0 else 100
    ticks = np.arange(0, upper + 1, 100, dtype=int)
    if len(ticks) == 0:
        ticks = np.array([0, 100], dtype=int)
    ax.set_xlim(0, upper)
    ax.set_xticks(ticks)
    ax.grid(True, alpha=0.25)


def finalize_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    arr = np.array(values, dtype=float)
    std = float(arr.std())
    if std == 0.0:
        return [0.0 for _ in values]
    mean = float(arr.mean())
    return [float((value - mean) / std) for value in values]


def moving_average(values: list[float], window: int = 5) -> list[float]:
    if not values:
        return []
    if window <= 1 or len(values) == 1:
        return [float(v) for v in values]
    half = window // 2
    smoothed: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - half)
        end = min(len(values), idx + half + 1)
        smoothed.append(float(sum(values[start:end]) / (end - start)))
    return smoothed


def interpolate_series(x_source: list[int], y_source: list[float], x_target: list[int]) -> list[float]:
    if not x_source or not y_source or not x_target:
        return []
    xs = np.array(x_source, dtype=float)
    ys = np.array(y_source, dtype=float)
    xt = np.array(x_target, dtype=float)
    interpolated = np.interp(xt, xs, ys)
    return [float(value) for value in interpolated]


def plot_training_loss(meta: ExperimentMeta, rows: list[dict[str, Any]], output_path: Path) -> None:
    steps = [row["step"] for row in rows]
    losses = [float(row["loss"]) for row in rows]
    max_step = max(steps)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle(f"{meta.short_name} Training Loss per 20 Steps across {max_step:,} steps", fontsize=14, y=0.98)
    ax.set_title(meta.eval_subtitle, fontsize=10, pad=10)
    ax.plot(steps, losses, color="tab:blue", marker="o", markersize=3, linewidth=1.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    set_common_axis_format(ax, max_step)
    finalize_figure(fig, output_path)


def metric_color_map(specs: list[tuple[str, tuple[str, ...], str]]) -> dict[str, Any]:
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    return {canonical: colors[idx % len(colors)] for idx, (canonical, _, _) in enumerate(specs)}


def plot_eval_series(
    meta: ExperimentMeta,
    rows: list[dict[str, Any]],
    specs: list[tuple[str, tuple[str, ...], str]],
    output_path: Path,
    title: str,
) -> bool:
    steps = [row["step"] for row in rows]
    max_step = max(steps)
    color_map = metric_color_map(specs)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle(f"{meta.short_name} {title} per 50 Steps across {max_step:,} steps", fontsize=14, y=0.98)
    ax.set_title(meta.eval_subtitle, fontsize=10, pad=10)

    plotted = False
    legend_handles: list[Line2D] = []
    for canonical, keys, label in specs:
        values = [choose_metric_value(row, keys) for row in rows]
        if not any(value is not None for value in values):
            continue
        xs = [step for step, value in zip(steps, values) if value is not None]
        ys = [value for value in values if value is not None]
        color = color_map[canonical]
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.8, color=color)
        legend_handles.append(Line2D([0], [0], color=color, marker="o", linewidth=1.8, label=label))
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlabel("Step")
    ax.set_ylabel("Metric")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    set_common_axis_format(ax, max_step)
    ax.legend(handles=legend_handles, loc="best")
    finalize_figure(fig, output_path)
    return True


def assess_alignment(
    inverted_loss_z: list[float],
    exact_z: list[float],
    f1_z: list[float],
) -> tuple[str, str]:
    if len(inverted_loss_z) < 2 or len(exact_z) < 2 or len(f1_z) < 2:
        return ("insufficient", "Too few aligned checkpoints to say much about shape alignment.")

    loss_arr = np.array(inverted_loss_z, dtype=float)
    exact_arr = np.array(exact_z, dtype=float)
    f1_arr = np.array(f1_z, dtype=float)
    if float(loss_arr.std()) == 0.0 or float(exact_arr.std()) == 0.0 or float(f1_arr.std()) == 0.0:
        return (
            "insufficient",
            "Insufficient variation: one or more z-scored series is nearly flat, so shape correlation is not very informative here.",
        )
    corr_exact = float(np.corrcoef(loss_arr, exact_arr)[0, 1])
    corr_f1 = float(np.corrcoef(loss_arr, f1_arr)[0, 1])
    if math.isnan(corr_exact) or math.isnan(corr_f1):
        return (
            "insufficient",
            "Insufficient variation: one or more series is too flat for a stable correlation-based shape comparison.",
        )
    avg_corr = (corr_exact + corr_f1) / 2.0

    if avg_corr >= 0.75:
        label = "strong"
        summary = (
            f"Strong alignment: the smoothed improvement-from-loss curve co-moves well with eval "
            f"Exact/F1 overall (corr exact={corr_exact:.2f}, corr f1={corr_f1:.2f})."
        )
    elif avg_corr >= 0.4:
        label = "moderate"
        summary = (
            f"Moderate alignment: the broad phases mostly match, but there are noticeable local "
            f"divergences between loss and eval metrics (corr exact={corr_exact:.2f}, corr f1={corr_f1:.2f})."
        )
    else:
        label = "weak"
        summary = (
            f"Weak alignment: loss shape and eval-metric shape diverge enough that loss alone would be "
            f"a poor proxy here (corr exact={corr_exact:.2f}, corr f1={corr_f1:.2f})."
        )
    return label, summary


def plot_shape_comparison(
    meta: ExperimentMeta,
    training_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    output_path: Path,
) -> str | None:
    eval_steps = [row["step"] for row in eval_rows]
    exact = [choose_metric_value(row, ("eval_exact", "eval_exact_match")) for row in eval_rows]
    f1 = [choose_metric_value(row, ("eval_f1",)) for row in eval_rows]
    if not any(value is not None for value in exact) or not any(value is not None for value in f1):
        return None

    train_steps = [row["step"] for row in training_rows]
    train_loss = [float(row["loss"]) for row in training_rows]
    smoothed_loss = moving_average(train_loss, window=5)
    smoothed_loss_at_eval = interpolate_series(train_steps, smoothed_loss, eval_steps)
    inverted_loss_z = zscore([-value for value in smoothed_loss_at_eval])

    exact_values = [float(value) if value is not None else math.nan for value in exact]
    f1_values = [float(value) if value is not None else math.nan for value in f1]
    valid_idx = [
        idx for idx, (e_val, f_val) in enumerate(zip(exact_values, f1_values))
        if not math.isnan(e_val) and not math.isnan(f_val)
    ]
    if len(valid_idx) < 2:
        return None

    valid_steps = [eval_steps[idx] for idx in valid_idx]
    valid_inverted_loss_z = [inverted_loss_z[idx] for idx in valid_idx]
    valid_exact_z = zscore([exact_values[idx] for idx in valid_idx])
    valid_f1_z = zscore([f1_values[idx] for idx in valid_idx])
    alignment_label, explanation = assess_alignment(valid_inverted_loss_z, valid_exact_z, valid_f1_z)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle(
        f"{meta.short_name} Z-scored Shape Comparison per 50 Steps across {max(eval_steps):,} steps",
        fontsize=14,
        y=0.98,
    )
    ax.set_title(
        meta.eval_subtitle + " | Smoothed loss is inverted so upward means improvement.",
        fontsize=10,
        pad=10,
    )
    ax.plot(valid_steps, valid_inverted_loss_z, color="tab:purple", linewidth=2.0, marker="o", label="Smoothed -Loss (z)")
    ax.plot(valid_steps, valid_exact_z, color="tab:blue", linewidth=1.8, marker="o", label="Exact (z)")
    ax.plot(valid_steps, valid_f1_z, color="tab:orange", linewidth=1.8, marker="o", label="F1 (z)")
    ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Step")
    ax.set_ylabel("Z-score")
    set_common_axis_format(ax, max(eval_steps))
    ax.legend(loc="best")
    ax.text(
        0.01,
        0.02,
        f"Assessment: {alignment_label}",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "lightgray"},
    )
    finalize_figure(fig, output_path)
    return explanation


def plot_final_metric_summary(
    experiments: list[tuple[ExperimentMeta, dict[str, Any]]],
    specs: list[tuple[str, tuple[str, ...], str]],
    output_path: Path,
    title: str,
) -> bool:
    if not experiments:
        return False

    color_map = metric_color_map(specs)
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(title, fontsize=14, y=0.98)
    ax.set_title("Each point is the final postprocessed metric for that experiment.", fontsize=10, pad=10)

    exp_numbers = [meta.number for meta, _ in experiments]
    plotted = False
    legend_handles: list[Line2D] = []
    for canonical, keys, label in specs:
        ys = [choose_metric_value(metrics, keys) for _, metrics in experiments]
        if not any(value is not None for value in ys):
            continue
        xs = [num for num, value in zip(exp_numbers, ys) if value is not None]
        vals = [value for value in ys if value is not None]
        color = color_map[canonical]
        ax.plot(xs, vals, marker="o", linewidth=1.8, color=color)
        legend_handles.append(Line2D([0], [0], color=color, marker="o", linewidth=1.8, label=label))
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlabel("Experiment Number")
    ax.set_ylabel("Metric")
    ax.set_xticks(exp_numbers)
    ax.grid(True, alpha=0.25)
    ax.legend(handles=legend_handles, loc="best")
    finalize_figure(fig, output_path)
    return True


def write_experiment_key_json(experiments: list[ExperimentMeta], output_path: Path) -> None:
    payload = [{meta.short_name: meta.key_payload()} for meta in experiments]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_3d_note(experiments: list[tuple[ExperimentMeta, list[dict[str, Any]], dict[str, Any] | None]], output_path: Path) -> None:
    lines = [
        "3D gradient-norm landscape note",
        "",
        "A true loss-landscape or valley plot is not recoverable from these benchmark logs alone.",
        "The logs contain a 1D training trajectory over step, plus scalar values like loss and grad_norm.",
        "They do not contain parameter-space coordinates, Hessian information, or a sampled 2D/3D slice",
        "through model weight space, so Matplotlib cannot reconstruct which basin or valley the best model sits in.",
        "",
        "What is available here:",
        "- step",
        "- loss",
        "- grad_norm",
        "- best_model_restored step",
        "",
        "What would be needed for a real landscape plot:",
        "- saved model checkpoints around the optimum",
        "- a method to define 2 directions in parameter space",
        "- recomputed loss or grad_norm on a grid over those directions",
        "",
        "Suggested tools/workflows:",
        "- Matplotlib or Plotly after generating a sampled loss grid",
        "- loss-landscape style tooling based on checkpoint interpolation",
        "- custom PyTorch evaluation scripts that sweep a 2D slice around the best checkpoint",
        "",
        "Best-model steps observed in these experiments:",
    ]
    for meta, _, best in experiments:
        if best is None:
            lines.append(f"- {meta.short_name}: no best_model_restored event found")
            continue
        lines.append(
            f"- {meta.short_name}: best step={best.get('step')} "
            f"({best.get('metric_name')}={best.get('metric_value')})"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_shape_notes(notes: list[tuple[ExperimentMeta, str]], output_path: Path) -> None:
    lines = [
        "# Z-scored Shape Comparison Notes",
        "",
        "Each plot overlays three z-scored series at eval checkpoints:",
        "- inverted smoothed training loss",
        "- eval Exact",
        "- eval F1",
        "",
        "Because the loss curve is inverted before z-scoring, upward movement means improvement for all three lines.",
        "",
    ]
    for meta, note in notes:
        wrapped = textwrap.fill(note, width=100, subsequent_indent="  ")
        lines.append(f"## {meta.short_name}")
        lines.append(wrapped)
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    bench_root = detect_bench_root(args.bench_root)
    output_root = args.output_root
    eval_output_root = output_root / "eval"
    test_output_root = output_root / "test"
    eval_output_root.mkdir(parents=True, exist_ok=True)
    test_output_root.mkdir(parents=True, exist_ok=True)

    metadata = load_run_metadata(args.run_script)
    bench_dirs = sorted(path.parent for path in bench_root.glob("*/bench.jsonl"))
    experiment_lookup = build_experiment_lookup(metadata, bench_dirs)

    aggregate_eval: list[tuple[ExperimentMeta, dict[str, Any]]] = []
    aggregate_test: list[tuple[ExperimentMeta, dict[str, Any]]] = []
    note_inputs: list[tuple[ExperimentMeta, list[dict[str, Any]], dict[str, Any] | None]] = []
    shape_notes: list[tuple[ExperimentMeta, str]] = []

    for _, (meta, bench_dir) in experiment_lookup.items():
        events = load_bench_events(bench_dir / "bench.jsonl")
        training_rows = collect_training_loss(events)
        eval_rows = collect_eval_metric_logs(events)
        final_eval = find_final_event(events, "eval_postprocessed")
        final_test = find_final_event(events, "test_postprocessed")
        best_model = find_best_model_restored(events)
        note_inputs.append((meta, training_rows, best_model))

        if training_rows:
            plot_training_loss(meta, training_rows, eval_output_root / f"{meta.short_name}_loss.png")

        if eval_rows:
            plot_eval_series(
                meta,
                eval_rows,
                EVAL_SERIES_SPECS,
                eval_output_root / f"{meta.short_name}_eval_exact_f1.png",
                "Eval Exact and F1",
            )
            plot_eval_series(
                meta,
                eval_rows,
                OPTIONAL_ANS_SERIES_SPECS,
                eval_output_root / f"{meta.short_name}_eval_answer_breakdown.png",
                "Eval HasAns and NoAns Metrics",
            )
            if training_rows:
                explanation = plot_shape_comparison(
                    meta,
                    training_rows,
                    eval_rows,
                    eval_output_root / f"{meta.short_name}_shape_comparison_zscore.png",
                )
                if explanation:
                    shape_notes.append((meta, explanation))

        if final_eval:
            aggregate_eval.append((meta, final_eval))
        if final_test:
            aggregate_test.append((meta, final_test))

    metas = [meta for meta, _ in experiment_lookup.values()]
    write_experiment_key_json(metas, output_root / "experiment_key.json")
    plot_final_metric_summary(
        aggregate_eval,
        FINAL_EVAL_METRIC_SPECS,
        output_root / "all_experiments_eval_postprocessed.png",
        "Final Eval Postprocessed Metrics Across Experiments",
    )
    plot_final_metric_summary(
        aggregate_test,
        FINAL_TEST_METRIC_SPECS,
        output_root / "all_experiments_test_postprocessed.png",
        "Final Test Postprocessed Metrics Across Experiments",
    )
    write_3d_note(note_inputs, output_root / "gradient_landscape_note.txt")
    write_shape_notes(shape_notes, output_root / "shape_comparison_notes.md")


if __name__ == "__main__":
    main()
