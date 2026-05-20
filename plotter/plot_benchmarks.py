#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_BENCH_ROOT = Path("/data/data/huggingface/bench")
DEFAULT_OUTPUT_ROOT = Path("results/experiment_plots")

LOSS_METRICS = ["loss"]
PREDICT_VALIDATION_METRICS = ["eval_loss"]
CORE_EVAL_METRICS = ["eval_exact", "eval_exact_match", "eval_f1"]
SQUAD_V2_EVAL_METRICS = [
    "eval_HasAns_exact",
    "eval_HasAns_f1",
    "eval_NoAns_exact",
    "eval_NoAns_f1",
]

DROP_SUFFIXES = ("_per_second", "_total", "_missing", "_thresh", "_runtime")
DROP_EXACT_KEYS = {"elapsed_seconds"}
LINE_COLORS = {
    "exact": "#ff7f0e",
    "f1": "#1f77b4",
    "loss": "#2ca02c",
    "hasans_exact": "#d62728",
    "hasans_f1": "#9467bd",
    "noans_exact": "#8c564b",
    "noans_f1": "#e377c2",
}
OVERALL_METRIC_COLOR_PAIRS = [
    ("#1f77b4", "#6baed6"),
    ("#ff7f0e", "#fdae6b"),
    ("#2ca02c", "#74c476"),
    ("#d62728", "#fb6a4a"),
    ("#9467bd", "#bcbddc"),
    ("#8c564b", "#c49c94"),
    ("#e377c2", "#f7b6d2"),
    ("#7f7f7f", "#c7c7c7"),
]


@dataclass
class ExperimentData:
    name: str
    path: Path
    bench_rows: list[dict[str, Any]]
    best_rows: list[dict[str, Any]]
    final_rows: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-experiment and cross-experiment matplotlib charts from "
            "benchmark JSONL files."
        )
    )
    parser.add_argument(
        "--bench-root",
        type=Path,
        default=DEFAULT_BENCH_ROOT,
        help="Directory containing experiment subdirectories with bench.jsonl/best.jsonl/final.jsonl.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where plots will be written.",
    )
    return parser.parse_args()


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if len(record) != 1:
                continue
            _, payload = next(iter(record.items()))
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def discover_experiments(bench_root: Path) -> list[ExperimentData]:
    experiments: list[ExperimentData] = []
    if not bench_root.exists():
        return experiments

    for experiment_dir in sorted(path for path in bench_root.iterdir() if path.is_dir()):
        experiments.append(
            ExperimentData(
                name=experiment_dir.name,
                path=experiment_dir,
                bench_rows=load_jsonl_records(experiment_dir / "bench.jsonl"),
                best_rows=load_jsonl_records(experiment_dir / "best.jsonl"),
                final_rows=load_jsonl_records(experiment_dir / "final.jsonl"),
            )
        )
    return experiments


def unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def display_experiment_name(name: str) -> str:
    return re.sub(r"_[0-9a-f]{12}$", "", name)


def experiment_number(name: str) -> str:
    match = re.match(r"(exp\d+)", display_experiment_name(name))
    return match.group(1) if match else display_experiment_name(name)


def canonical_metric_name(metric: str) -> str:
    aliases = {
        "eval_exact_match": "eval_exact",
        "test_exact_match": "test_exact",
    }
    return aliases.get(metric, metric)


def should_drop_metric(metric: str) -> bool:
    if metric in DROP_EXACT_KEYS:
        return True
    return any(metric.endswith(suffix) for suffix in DROP_SUFFIXES)


def is_hasans_noans_metric(metric: str) -> bool:
    return "_HasAns_" in metric or "_NoAns_" in metric


def normalized_row(row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        canonical = canonical_metric_name(key)
        result[canonical] = value
    return result


def filter_log_rows(rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    return filter_event_rows(rows, "log", metrics)


def filter_event_rows(rows: list[dict[str, Any]], event_name: str, metrics: list[str]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    canonical_metrics = {canonical_metric_name(metric) for metric in metrics}
    for row in rows:
        row = normalized_row(row)
        if row.get("event") != event_name:
            continue
        if "step" not in row:
            continue
        if any(metric in row for metric in canonical_metrics):
            filtered.append(row)
    return filtered


def extract_series(rows: list[dict[str, Any]], metric: str) -> tuple[list[float], list[float]]:
    canonical = canonical_metric_name(metric)
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        row = normalized_row(row)
        value = row.get(canonical)
        if value is None:
            continue
        xs.append(float(row["step"]))
        ys.append(float(value))
    return xs, ys


def event_payload(rows: list[dict[str, Any]], event_name: str) -> dict[str, Any] | None:
    for row in rows:
        row = normalized_row(row)
        if row.get("event") == event_name:
            return row
    return None


def metric_keys(payload: dict[str, Any] | None, prefix: str, exclude_loss: bool = True) -> list[str]:
    if not payload:
        return []

    keys: list[str] = []
    for key, value in normalized_row(payload).items():
        if not isinstance(value, (int, float)):
            continue
        if not key.startswith(prefix):
            continue
        if should_drop_metric(key):
            continue
        if prefix == "eval_" and is_hasans_noans_metric(key):
            continue
        if exclude_loss and "loss" in key.lower():
            continue
        keys.append(key)
    return sorted(unique_in_order(keys))


def bench_contains_any_metrics(rows: list[dict[str, Any]], metrics: list[str]) -> bool:
    canonical_metrics = {canonical_metric_name(metric) for metric in metrics}
    for row in rows:
        row = normalized_row(row)
        if any(metric in row for metric in canonical_metrics):
            return True
    return False


def filter_metrics_by_bench_presence(bench_rows: list[dict[str, Any]], metrics: list[str]) -> list[str]:
    filtered: list[str] = []
    has_special_metrics = bench_contains_any_metrics(bench_rows, SQUAD_V2_EVAL_METRICS)
    for metric in metrics:
        canonical = canonical_metric_name(metric)
        if canonical in SQUAD_V2_EVAL_METRICS and not has_special_metrics:
            continue
        filtered.append(metric)
    return filtered


def create_figure(num_panels: int, title: str) -> tuple[plt.Figure, np.ndarray]:
    cols = 2 if num_panels > 1 else 1
    rows = math.ceil(num_panels / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows), squeeze=False)
    fig.suptitle(title, fontsize=14)
    return fig, axes


def metric_positions(metrics: list[str]) -> np.ndarray:
    if not metrics:
        return np.array([], dtype=float)

    positions: list[float] = []
    current = 0.0

    for index, metric in enumerate(metrics):
        if index == 0:
            positions.append(current)
            continue
        current += 1.0
        positions.append(current)

    return np.array(positions, dtype=float)


def overall_metric_color_map(metrics: list[str]) -> dict[str, dict[str, str]]:
    color_map: dict[str, dict[str, str]] = {}
    for index, metric in enumerate(metrics):
        best_color, final_color = OVERALL_METRIC_COLOR_PAIRS[index % len(OVERALL_METRIC_COLOR_PAIRS)]
        color_map[metric] = {
            "best": best_color,
            "final": final_color,
        }
    return color_map


def bar_width_for_chart(total_bars: int, normal_width: float = 0.66) -> float:
    if total_bars == 4:
        return normal_width * 0.5
    return normal_width


def save_line_plot(
    rows: list[dict[str, Any]],
    metrics: list[str],
    title: str,
    output_path: Path,
    y_label: str,
    color_map: dict[str, str] | None = None,
) -> None:
    available_metrics = []
    for metric in metrics:
        canonical = canonical_metric_name(metric)
        if canonical in available_metrics:
            continue
        if any(canonical in normalized_row(row) for row in rows):
            available_metrics.append(canonical)

    if not available_metrics:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for metric in available_metrics:
        xs, ys = extract_series(rows, metric)
        if not xs:
            continue
        color = color_map.get(metric) if color_map else None
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=4, label=metric, color=color)

    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_variant_metric_bar_chart(
    experiment_name: str,
    variant_payloads: dict[str, dict[str, Any]],
    metrics: list[str],
    title: str,
    output_path: Path,
) -> None:
    present_metrics = [
        metric
        for metric in unique_in_order(canonical_metric_name(metric) for metric in metrics)
        if any(canonical_metric_name(metric) in normalized_row(payload) for payload in variant_payloads.values())
    ]
    if not present_metrics:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(present_metrics) * 1.8), 5))
    base_positions = metric_positions(present_metrics)
    bar_width = bar_width_for_chart(len(present_metrics) * 2)
    colors = {"best": "#ff7f0e", "final": "#1f77b4"}
    labels_added = {"best": False, "final": False}

    best_payload = normalized_row(variant_payloads.get("best", {}))
    final_payload = normalized_row(variant_payloads.get("final", {}))
    for position, metric in zip(base_positions, present_metrics):
        metric_values = {
            "best": float(best_payload.get(metric, np.nan)),
            "final": float(final_payload.get(metric, np.nan)),
        }
        draw_order = sorted(
            ("best", "final"),
            key=lambda variant: (
                np.isnan(metric_values[variant]),
                -(metric_values[variant] if not np.isnan(metric_values[variant]) else float("-inf")),
            ),
        )
        for variant_name in draw_order:
            value = metric_values[variant_name]
            ax.bar(
                position,
                value,
                width=bar_width,
                label=variant_name if not labels_added[variant_name] else None,
                color=colors[variant_name],
                alpha=1.0,
                linewidth=0,
            )
            labels_added[variant_name] = True

    ax.set_xticks(base_positions)
    ax.set_xticklabels(present_metrics, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Metric Value")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.suptitle(display_experiment_name(experiment_name), fontsize=12, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_combined_variant_metric_bar_chart(
    experiment_name: str,
    eval_variant_payloads: dict[str, dict[str, Any]],
    eval_metrics: list[str],
    test_variant_payloads: dict[str, dict[str, Any]],
    test_metrics: list[str],
    title: str,
    output_path: Path,
) -> None:
    eval_present_metrics = [
        metric
        for metric in unique_in_order(canonical_metric_name(metric) for metric in eval_metrics)
        if any(canonical_metric_name(metric) in normalized_row(payload) for payload in eval_variant_payloads.values())
    ]
    test_present_metrics = [
        metric
        for metric in unique_in_order(canonical_metric_name(metric) for metric in test_metrics)
        if any(canonical_metric_name(metric) in normalized_row(payload) for payload in test_variant_payloads.values())
    ]
    if not eval_present_metrics and not test_present_metrics:
        return

    section_gap = 1.8
    eval_positions = metric_positions(eval_present_metrics)
    if eval_present_metrics:
        test_start = eval_positions[-1] + section_gap + 1.0
    else:
        test_start = 0.0
    test_positions = metric_positions(test_present_metrics) + test_start

    combined_positions = np.concatenate([eval_positions, test_positions]) if len(test_positions) else eval_positions
    total_bars = (len(eval_present_metrics) + len(test_present_metrics)) * 2
    bar_width = bar_width_for_chart(total_bars)
    colors = {"best": "#ff7f0e", "final": "#1f77b4"}
    labels_added = {"best": False, "final": False}

    fig, ax = plt.subplots(figsize=(max(10, len(combined_positions) * 1.8), 5.5))

    def draw_section(
        positions: np.ndarray,
        metrics: list[str],
        variant_payloads: dict[str, dict[str, Any]],
    ) -> None:
        best_payload = normalized_row(variant_payloads.get("best", {}))
        final_payload = normalized_row(variant_payloads.get("final", {}))
        for position, metric in zip(positions, metrics):
            metric_values = {
                "best": float(best_payload.get(metric, np.nan)),
                "final": float(final_payload.get(metric, np.nan)),
            }
            draw_order = sorted(
                ("best", "final"),
                key=lambda variant: (
                    np.isnan(metric_values[variant]),
                    -(metric_values[variant] if not np.isnan(metric_values[variant]) else float("-inf")),
                ),
            )
            for variant_name in draw_order:
                ax.bar(
                    position,
                    metric_values[variant_name],
                    width=bar_width,
                    label=variant_name if not labels_added[variant_name] else None,
                    color=colors[variant_name],
                    alpha=1.0,
                    linewidth=0,
                )
                labels_added[variant_name] = True

    draw_section(eval_positions, eval_present_metrics, eval_variant_payloads)
    draw_section(test_positions, test_present_metrics, test_variant_payloads)

    tick_positions = list(eval_positions) + list(test_positions)
    tick_labels = eval_present_metrics + test_present_metrics
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Metric Value")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    super_positions: list[float] = []
    super_labels: list[str] = []
    if len(eval_positions) > 0:
        super_positions.append(float((eval_positions[0] + eval_positions[-1]) / 2))
        super_labels.append("Validation")
    if len(test_positions) > 0:
        super_positions.append(float((test_positions[0] + test_positions[-1]) / 2))
        super_labels.append("Test")
    if super_positions:
        secondary = ax.secondary_xaxis("top")
        secondary.set_xticks(super_positions)
        secondary.set_xticklabels(super_labels)

    fig.suptitle(display_experiment_name(experiment_name), fontsize=12, y=1.03)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_cross_experiment_variant_chart(
    experiments: list[ExperimentData],
    event_name: str,
    metric_prefix: str,
    title: str,
    output_path: Path,
) -> None:
    metric_union: list[str] = []
    experiment_metric_sets: dict[str, list[str]] = {}
    for experiment in experiments:
        best_payload = event_payload(experiment.best_rows, event_name)
        final_payload = event_payload(experiment.final_rows, event_name)
        experiment_metrics = unique_in_order(
            filter_metrics_by_bench_presence(
                experiment.bench_rows,
                metric_keys(best_payload, metric_prefix) + metric_keys(final_payload, metric_prefix),
            )
        )
        experiment_metric_sets[experiment.name] = experiment_metrics
        for key in experiment_metrics:
            if key not in metric_union:
                metric_union.append(key)

    if not metric_union:
        return

    fig, ax = plt.subplots(figsize=(max(16, len(experiments) * len(metric_union) * 0.8), 7))
    group_gap = 1.8
    total_bars = sum(len(metrics) * 2 for metrics in experiment_metric_sets.values())
    bar_width = bar_width_for_chart(total_bars)
    metric_colors = overall_metric_color_map(metric_union)
    labels_added: set[str] = set()
    experiment_centers: list[float] = []
    experiment_labels: list[str] = []
    current_group_start = 0.0

    for experiment in experiments:
        best_payload = normalized_row(event_payload(experiment.best_rows, event_name) or {})
        final_payload = normalized_row(event_payload(experiment.final_rows, event_name) or {})
        experiment_metrics = experiment_metric_sets[experiment.name]
        metric_offsets = metric_positions(experiment_metrics)
        group_span = (metric_offsets[-1] - metric_offsets[0]) if len(metric_offsets) > 0 else 0.0
        group_start = current_group_start

        if len(experiment_metrics) > 0:
            first_x = group_start + metric_offsets[0]
            last_x = group_start + metric_offsets[-1]
            experiment_centers.append((first_x + last_x) / 2)
            experiment_labels.append(experiment_number(experiment.name))

        for metric_index, metric in enumerate(experiment_metrics):
            base_x = group_start + metric_offsets[metric_index]
            metric_values = {
                "best": float(best_payload.get(metric, np.nan)),
                "final": float(final_payload.get(metric, np.nan)),
            }
            draw_order = sorted(
                ("best", "final"),
                key=lambda variant: (
                    np.isnan(metric_values[variant]),
                    -(metric_values[variant] if not np.isnan(metric_values[variant]) else float("-inf")),
                ),
            )
            for variant_name in draw_order:
                ax.bar(
                    base_x,
                    metric_values[variant_name],
                    width=bar_width,
                    color=metric_colors[metric][variant_name],
                    label=f"{metric} {variant_name}" if f"{metric} {variant_name}" not in labels_added else None,
                    alpha=1.0,
                    linewidth=0,
                )
                labels_added.add(f"{metric} {variant_name}")

        current_group_start = group_start + group_span + group_gap + 1.0
    ax.set_xticks(experiment_centers)
    ax.set_xticklabels(experiment_labels)
    ax.set_title(title)
    ax.set_ylabel("Metric Value")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_per_experiment_plots(experiment: ExperimentData, output_root: Path) -> None:
    experiment_dir = output_root / experiment.name
    log_rows = filter_log_rows(
        experiment.bench_rows,
        LOSS_METRICS + CORE_EVAL_METRICS + SQUAD_V2_EVAL_METRICS,
    )
    predict_rows = filter_event_rows(
        experiment.bench_rows,
        "predict",
        PREDICT_VALIDATION_METRICS + CORE_EVAL_METRICS + SQUAD_V2_EVAL_METRICS,
    )

    save_line_plot(
        rows=filter_log_rows(log_rows, LOSS_METRICS),
        metrics=LOSS_METRICS,
        title=f"{display_experiment_name(experiment.name)} Training Loss",
        output_path=experiment_dir / "training_loss.png",
        y_label="Loss",
        color_map={"loss": LINE_COLORS["loss"]},
    )

    save_line_plot(
        rows=filter_log_rows(log_rows, CORE_EVAL_METRICS),
        metrics=CORE_EVAL_METRICS,
        title=f"{display_experiment_name(experiment.name)} Logged Validation Metrics",
        output_path=experiment_dir / "logged_validation_core.png",
        y_label="Metric Value",
        color_map={
            "eval_exact": LINE_COLORS["exact"],
            "eval_f1": LINE_COLORS["f1"],
        },
    )

    if any(any(canonical_metric_name(metric) in normalized_row(row) for metric in SQUAD_V2_EVAL_METRICS) for row in log_rows):
        save_line_plot(
            rows=filter_log_rows(log_rows, SQUAD_V2_EVAL_METRICS),
            metrics=SQUAD_V2_EVAL_METRICS,
            title=f"{display_experiment_name(experiment.name)} Logged SQuAD v2 Validation Metrics",
            output_path=experiment_dir / "logged_validation_hasans_noans.png",
            y_label="Metric Value",
            color_map={
                "eval_HasAns_exact": LINE_COLORS["hasans_exact"],
                "eval_HasAns_f1": LINE_COLORS["hasans_f1"],
                "eval_NoAns_exact": LINE_COLORS["noans_exact"],
                "eval_NoAns_f1": LINE_COLORS["noans_f1"],
            },
        )

    save_line_plot(
        rows=predict_rows,
        metrics=PREDICT_VALIDATION_METRICS,
        title=f"{display_experiment_name(experiment.name)} Training Validation Metrics",
        output_path=experiment_dir / "training_validation_metrics.png",
        y_label="Metric Value",
        color_map={"eval_loss": LINE_COLORS["loss"]},
    )

    eval_variant_payloads = {
        "best": event_payload(experiment.best_rows, "eval_postprocessed") or {},
        "final": event_payload(experiment.final_rows, "eval_postprocessed") or {},
    }
    test_variant_payloads = {
        "best": event_payload(experiment.best_rows, "test_postprocessed") or {},
        "final": event_payload(experiment.final_rows, "test_postprocessed") or {},
    }
    eval_metrics = filter_metrics_by_bench_presence(
        experiment.bench_rows,
        metric_keys(eval_variant_payloads["best"], "eval_") + metric_keys(eval_variant_payloads["final"], "eval_"),
    )
    test_metrics = filter_metrics_by_bench_presence(
        experiment.bench_rows,
        metric_keys(test_variant_payloads["best"], "test_") + metric_keys(test_variant_payloads["final"], "test_"),
    )
    save_combined_variant_metric_bar_chart(
        experiment_name=experiment.name,
        eval_variant_payloads=eval_variant_payloads,
        eval_metrics=eval_metrics,
        test_variant_payloads=test_variant_payloads,
        test_metrics=test_metrics,
        title="Final Validation And Test Metrics",
        output_path=experiment_dir / "final_metrics.png",
    )


def main() -> int:
    args = parse_args()
    experiments = discover_experiments(args.bench_root)
    if not experiments:
        raise SystemExit(f"No experiment directories found under {args.bench_root}")

    for experiment in experiments:
        generate_per_experiment_plots(experiment, args.output_root)

    save_cross_experiment_variant_chart(
        experiments=experiments,
        event_name="eval_postprocessed",
        metric_prefix="eval_",
        title="All Experiments Final Validation Metrics: Best vs Final",
        output_path=args.output_root / "all_experiments_final_validation.png",
    )

    save_cross_experiment_variant_chart(
        experiments=experiments,
        event_name="test_postprocessed",
        metric_prefix="test_",
        title="All Experiments Final Test Metrics: Best vs Final",
        output_path=args.output_root / "all_experiments_final_test.png",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
