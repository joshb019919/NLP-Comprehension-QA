from datasets import Dataset
from typing import Any
import collections
import numpy as np


def postprocess_triviaqa_predictions(
    examples: Dataset,
    features: Dataset,
    raw_predictions: tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_start_logits, all_end_logits = raw_predictions

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)

    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = []
    references = []

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        best_text = ""
        best_score = -1e9
        min_null_score = None

        context = example["context"]
        answers = example.get("answers", {"text": [], "answer_start": []})

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            cls_score = float(start_logits[0] + end_logits[0])
            if min_null_score is None or cls_score < min_null_score:
                min_null_score = cls_score

            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                        continue
                    if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue
                    if end_index < start_index:
                        continue
                    if end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    text = context[start_char:end_char]
                    score = float(start_logits[start_index] + end_logits[end_index])

                    if score > best_score:
                        best_score = score
                        best_text = text

        final_text = best_text
        if version_2_with_negative and min_null_score is not None:
            score_diff = min_null_score - best_score
            if score_diff > null_score_diff_threshold:
                final_text = ""

        predictions.append({
            "id": example["question_id"],
            "prediction_text": final_text,
            "score": best_score,
            "no_answer_probability": 0.0,
        })

        references.append({
            "id": example["question_id"],
            "answers": answers,
        })

    return predictions, references


def postprocess_squad_predictions(
    examples: Dataset,
    features: Dataset,
    raw_predictions: tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_start_logits, all_end_logits = raw_predictions

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)

    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = []
    references = []

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        best_text = ""
        best_score = -1e9
        min_null_score = None

        context = example["context"]
        answers = example["answers"]

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            cls_score = float(start_logits[0] + end_logits[0])
            if min_null_score is None or cls_score < min_null_score:
                min_null_score = cls_score

            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                        continue
                    if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue
                    if end_index < start_index:
                        continue
                    if end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    text = context[start_char:end_char]
                    score = float(start_logits[start_index] + end_logits[end_index])

                    if score > best_score:
                        best_score = score
                        best_text = text

        final_text = best_text
        if version_2_with_negative and min_null_score is not None:
            score_diff = min_null_score - best_score
            if score_diff > null_score_diff_threshold:
                final_text = ""

        predictions.append({
            "id": example["id"],
            "prediction_text": final_text,
            "no_answer_probability": 0.0,
        })

        references.append({
            "id": example["id"],
            "answers": answers,
        })

    return predictions, references