from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

class BenchLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.counter = 0
        self.start_time = time.perf_counter()

    def elapsed_seconds(self) -> float:
        return float(time.perf_counter() - self.start_time)

    def write(self, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        # Store a monotonic elapsed timestamp in seconds for every event.
        payload.setdefault("elapsed_seconds", self.elapsed_seconds())
        line = {str(self.counter): payload}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        self.counter += 1


class BenchCallback(TrainerCallback):
    def __init__(self, logger: BenchLogger) -> None:
        self.logger = logger

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.logger.write({"event": "train_begin"})

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.logger.write({"event": "train_end", "global_step": state.global_step})

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        payload = {"event": "log", "step": state.global_step}
        if logs:
            payload.update(logs)
        self.logger.write(payload)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        payload = {"event": "evaluate", "step": state.global_step}
        if metrics:
            payload.update(metrics)
        self.logger.write(payload)

    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        payload = {"event": "predict", "step": state.global_step}
        if metrics:
            payload.update(metrics)
        self.logger.write(payload)
