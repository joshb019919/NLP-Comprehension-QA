#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


BENCH_ROOT = Path("/data/data/huggingface/bench")
OUTPUT_ROOT = Path("results/experiment_plots")
COMPARISON_ROOT = OUTPUT_ROOT / "comparisons"

EXPERIMENTS = {
    "exp01": "exp01_bert_triviaqa_4ab8b1e38d83",
    "exp05": "exp05_bert_triviaqa_3epochs_4ab8b1e38d83",
}
LOGGED_VALIDATION_MAX_STEP = 600

EXPERIMENT_COLORS = {
    "exp01": {"exact": "#1f77b4", "f1": "#ff7f0e", "loss": "#2ca02c"},
    "exp05": {"exact": "#6baed6", "f1": "#fdae6b", "loss": "#74c476"},
}

FINAL_BAR_COLORS = {
    "exp01": {"best": "#1f77b4", "final": "#6baed6"},
    "exp05": {"best": "#ff7f0e", "final": "#fdae6b"},
}


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            _, payload = next(iter(record.items()))
            rows.append(payload)
    return rows


def normalize_key(key: str) -> str:
    aliases = {
        "eval_exact_match": "eval_exact",
        "test_exact_match": "test_exact",
    }
    return aliases.get(key, key)


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {normalize_key(key): value for key, value in row.items()}


def logged_validation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        row = normalize_row(row)
        if (
            row.get("event") == "log"
            and ("eval_exact" in row or "eval_f1" in row)
            and float(row.get("step", 0)) <= LOGGED_VALIDATION_MAX_STEP
        ):
            result.append(row)
    return result


def training_loss_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        row = normalize_row(row)
        if row.get("event") == "log" and "loss" in row:
            result.append(row)
    return result


def interpolate_loss_to_eval_steps(
    loss_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    if not loss_rows or not eval_rows:
        return np.array([]), np.array([])

    loss_steps = np.array([float(row["step"]) for row in loss_rows], dtype=float)
    loss_values = np.array([float(row["loss"]) for row in loss_rows], dtype=float)
    eval_steps = np.array([float(row["step"]) for row in eval_rows], dtype=float)

    interpolated_loss = np.interp(eval_steps, loss_steps, loss_values)
    max_loss = float(np.max(interpolated_loss))
    min_loss = float(np.min(interpolated_loss))
    if max_loss == min_loss:
        normalized_inverted = np.ones_like(interpolated_loss)
    else:
        normalized = (interpolated_loss - min_loss) / (max_loss - min_loss)
        normalized_inverted = 1.0 - normalized
    return eval_steps, normalized_inverted


def event_payload(rows: list[dict[str, Any]], event_name: str) -> dict[str, Any]:
    for row in rows:
        row = normalize_row(row)
        if row.get("event") == event_name:
            return row
    return {}


def save_individual_logged_validation_with_loss(experiment_key: str, experiment_dir: str) -> None:
    bench_rows = load_jsonl_rows(BENCH_ROOT / experiment_dir / "bench.jsonl")
    eval_rows = logged_validation_rows(bench_rows)
    loss_rows = training_loss_rows(bench_rows)

    if not eval_rows:
        return

    steps = np.array([float(row["step"]) for row in eval_rows], dtype=float)
    exact = np.array([float(row["eval_exact"]) for row in eval_rows], dtype=float)
    f1 = np.array([float(row["eval_f1"]) for row in eval_rows], dtype=float)
    loss_steps, normalized_inverted_loss = interpolate_loss_to_eval_steps(loss_rows, eval_rows)

    colors = EXPERIMENT_COLORS[experiment_key]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, exact, marker="o", linewidth=2, color=colors["exact"], label="eval_exact")
    ax.plot(steps, f1, marker="o", linewidth=2, color=colors["f1"], label="eval_f1")
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Metric")
    ax.set_title(f"{experiment_key} Logged Validation Metrics")
    ax.grid(True, alpha=0.3)

    if len(loss_steps) > 0:
        ax2 = ax.twinx()
        ax2.plot(
            loss_steps,
            normalized_inverted_loss,
            marker="s",
            linewidth=2,
            linestyle="--",
            color=colors["loss"],
            label="normalized_inverted_loss",
        )
        ax2.set_ylabel("Normalized Inverted Loss")
        lines = ax.get_lines() + ax2.get_lines()
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc="best")
    else:
        ax.legend(loc="best")

    fig.tight_layout()
    output_path = OUTPUT_ROOT / experiment_dir / "logged_validation_core.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_combined_logged_validation() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for experiment_key, experiment_dir in EXPERIMENTS.items():
        bench_rows = load_jsonl_rows(BENCH_ROOT / experiment_dir / "bench.jsonl")
        eval_rows = logged_validation_rows(bench_rows)
        if not eval_rows:
            continue

        steps = np.array([float(row["step"]) for row in eval_rows], dtype=float)
        exact = np.array([float(row["eval_exact"]) for row in eval_rows], dtype=float)
        f1 = np.array([float(row["eval_f1"]) for row in eval_rows], dtype=float)
        colors = EXPERIMENT_COLORS[experiment_key]
        ax.plot(steps, exact, marker="o", linewidth=2, color=colors["exact"], label=f"{experiment_key} eval_exact")
        ax.plot(steps, f1, marker="o", linewidth=2, color=colors["f1"], label=f"{experiment_key} eval_f1")

    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Metric")
    ax.set_title("exp01 vs exp05 Logged Validation Core")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    COMPARISON_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(COMPARISON_ROOT / "exp01_exp05_logged_validation_core.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_combined_final_metrics() -> None:
    sections = [
        ("Validation", "eval_postprocessed", ["eval_exact", "eval_f1"]),
        ("Test", "test_postprocessed", ["test_exact", "test_f1"]),
    ]

    section_gap = 1.8
    experiment_gap = 1.8
    metric_slot_gap = 0.7
    bar_width = 0.66
    fig, ax = plt.subplots(figsize=(12, 6))
    x_tick_positions: list[float] = []
    x_tick_labels: list[str] = []
    super_centers: list[float] = []
    super_labels: list[str] = []
    labels_added: set[str] = set()
    current_x = 0.0

    for section_name, event_name, metrics in sections:
        section_start = current_x
        for experiment_key, experiment_dir in EXPERIMENTS.items():
            rows = load_jsonl_rows(BENCH_ROOT / experiment_dir / "final.jsonl")
            payload = normalize_row(event_payload(rows, event_name))
            center_x = current_x
            metric_offsets = np.linspace(
                -metric_slot_gap / 2,
                metric_slot_gap / 2,
                num=len(metrics),
            ) if len(metrics) > 1 else np.array([0.0])
            positions = center_x + metric_offsets

            for position, metric in zip(positions, metrics):
                values = {
                    "best": float(normalize_row(event_payload(load_jsonl_rows(BENCH_ROOT / experiment_dir / "best.jsonl"), event_name)).get(metric, np.nan)),
                    "final": float(payload.get(metric, np.nan)),
                }
                draw_order = sorted(
                    ("best", "final"),
                    key=lambda variant: (
                        np.isnan(values[variant]),
                        -(values[variant] if not np.isnan(values[variant]) else float("-inf")),
                    ),
                )
                for variant_name in draw_order:
                    label = f"{experiment_key} {variant_name}"
                    ax.bar(
                        position,
                        values[variant_name],
                        width=bar_width,
                        color=FINAL_BAR_COLORS[experiment_key][variant_name],
                        alpha=1.0,
                        linewidth=0,
                        label=label if label not in labels_added else None,
                    )
                    labels_added.add(label)

            x_tick_positions.append(center_x)
            x_tick_labels.append(experiment_key)
            current_x = center_x + experiment_gap

        section_end = current_x - experiment_gap
        super_centers.append((section_start + section_end) / 2)
        super_labels.append(section_name)
        current_x += section_gap

    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels)
    ax.set_ylabel("Metric Value")
    ax.set_title("exp01 vs exp05 Final Metrics")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best")

    secondary = ax.secondary_xaxis("top")
    secondary.set_xticks(super_centers)
    secondary.set_xticklabels(super_labels)

    fig.tight_layout()
    COMPARISON_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(COMPARISON_ROOT / "exp01_exp05_final_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    for experiment_key, experiment_dir in EXPERIMENTS.items():
        save_individual_logged_validation_with_loss(experiment_key, experiment_dir)
    save_combined_logged_validation()
    save_combined_final_metrics()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
