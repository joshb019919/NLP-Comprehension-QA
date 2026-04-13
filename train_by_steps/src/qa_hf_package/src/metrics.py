from __future__ import annotations

from typing import Any

import evaluate


def normalize_answer_text(text: str) -> str:
    return " ".join(str(text).strip().split()).lower()


class QAMetricBundle:
    def __init__(self, version_2_with_negative: bool = False) -> None:
        self.version_2_with_negative = version_2_with_negative
        metric_name = "squad_v2" if version_2_with_negative else "squad"
        self.metric = evaluate.load(metric_name)

    def compute(self, predictions: list[dict[str, Any]], references: list[dict[str, Any]]) -> dict[str, float]:
        normalized_predictions: list[dict[str, Any]] = []
        normalized_references: list[dict[str, Any]] = []

        for p, r in zip(predictions, references):
            normalized_pred = {
                "id": str(p.get("id", "")),
                "prediction_text": str(p.get("prediction_text", "")),
            }
            if self.version_2_with_negative:
                normalized_pred["no_answer_probability"] = float(p.get("no_answer_probability", 0.0))

            answers = r.get("answers") if isinstance(r, dict) else None
            if not isinstance(answers, dict):
                answers = {"text": [], "answer_start": []}
            text = answers.get("text", [])
            answer_start = answers.get("answer_start", [])
            normalized_ref = {
                "id": str(r.get("id", "")) if isinstance(r, dict) else "",
                "answers": {
                    "text": list(text) if isinstance(text, list) else [],
                    "answer_start": list(answer_start) if isinstance(answer_start, list) else [],
                },
            }

            # The plain SQuAD metric cannot score examples with no ground-truth answers.
            if (not self.version_2_with_negative) and len(normalized_ref["answers"]["text"]) == 0:
                continue

            normalized_predictions.append(normalized_pred)
            normalized_references.append(normalized_ref)

        if len(normalized_predictions) == 0:
            # Keep downstream logging stable even when a split has no scorable labels.
            return {"exact_match": 0.0, "f1": 0.0, "num_scored_examples": 0.0}

        result = self.metric.compute(predictions=normalized_predictions, references=normalized_references)
        scored = {k: float(v) for k, v in result.items()}
        scored["num_scored_examples"] = float(len(normalized_predictions))
        return scored
