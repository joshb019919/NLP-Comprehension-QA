import re
import string
from collections import Counter
from typing import Iterable, List, Dict, Any

from datasets import Dataset
import numpy as np
from common import release_memory, shutdown_trainer_dataloader
from postprocess import (
    postprocess_triviaqa_predictions,
    postprocess_squad_predictions,
)
from transformers import Trainer
import evaluate


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.UNICODE)


def triviaqa_normalize_answer(text: str) -> str:
    if text is None:
        return ""
    text = text.lower()
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)
    text = _ARTICLES_RE.sub(" ", text)
    text = " ".join(text.split())
    return text


def triviaqa_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = triviaqa_normalize_answer(prediction).split()
    gold_tokens = triviaqa_normalize_answer(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def triviaqa_exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(triviaqa_normalize_answer(prediction) == triviaqa_normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: Iterable[str]) -> float:
    scores = [metric_fn(prediction, gt) for gt in ground_truths if gt is not None]
    return max(scores) if scores else 0.0


def extract_ground_truths_from_answer(answer: Dict[str, Any]) -> List[str]:
    values: List[str] = []

    value = answer.get("value")
    if value:
        values.append(value)

    for alias in answer.get("aliases", []) or []:
        if alias:
            values.append(alias)

    seen = set()
    deduped = []
    for x in values:
        nx = triviaqa_normalize_answer(x)
        if nx not in seen:
            seen.add(nx)
            deduped.append(x)

    return deduped


def score_triviaqa_prediction(prediction: str, ground_truths: List[str]) -> Dict[str, float]:
    em = metric_max_over_ground_truths(triviaqa_exact_match_score, prediction, ground_truths)
    f1 = metric_max_over_ground_truths(triviaqa_f1_score, prediction, ground_truths)
    return {"exact_match": em, "f1": f1}


def evaluate_triviaqa(
    predictions: Dict[str, str],
    references: Dict[str, List[str]],
) -> Dict[str, float]:
    total = 0
    exact_match = 0.0
    f1 = 0.0
    missing = 0

    for qid, golds in references.items():
        total += 1
        pred = predictions.get(qid)
        if pred is None:
            missing += 1
            pred = ""

        scores = score_triviaqa_prediction(pred, golds)
        exact_match += scores["exact_match"]
        f1 += scores["f1"]

    if total == 0:
        return {"exact_match": 0.0, "f1": 0.0, "total": 0, "missing": 0}

    return {
        "exact_match": 100.0 * exact_match / total,
        "f1": 100.0 * f1 / total,
        "total": total,
        "missing": missing,
    }


def aggregate_predictions_by_question_id(predictions: list[dict[str, Any]]) -> Dict[str, str]:
    best_by_qid: dict[str, dict[str, Any]] = {}
    for p in predictions:
        qid = str(p["id"])
        score = float(p.get("score", float("-inf")))
        if qid not in best_by_qid or score > float(best_by_qid[qid].get("score", float("-inf"))):
            best_by_qid[qid] = p
    return {qid: item["prediction_text"] for qid, item in best_by_qid.items()}


def build_triviaqa_reference_map_from_raw(raw_examples: Dataset) -> Dict[str, List[str]]:
    ref_map: Dict[str, List[str]] = {}
    for ex in raw_examples:
        qid = str(ex["question_id"])
        ref_map[qid] = extract_ground_truths_from_answer(ex["answer"])
    return ref_map


def evaluate_squad_family(
    predictions: list[dict[str, Any]],
    references: list[dict[str, Any]],
    version_2_with_negative: bool,
) -> dict[str, float]:
    if len(references) == 0:
        if version_2_with_negative:
            return {
                "exact": 0.0,
                "f1": 0.0,
                "total": 0.0,
                "HasAns_exact": 0.0,
                "HasAns_f1": 0.0,
                "HasAns_total": 0.0,
                "NoAns_exact": 0.0,
                "NoAns_f1": 0.0,
                "NoAns_total": 0.0,
                "best_exact": 0.0,
                "best_exact_thresh": 0.0,
                "best_f1": 0.0,
                "best_f1_thresh": 0.0,
            }
        return {
            "exact_match": 0.0,
            "f1": 0.0,
        }

    metric_name = "squad_v2" if version_2_with_negative else "squad"
    metric = evaluate.load(metric_name)
    normalized_predictions = []
    for prediction in predictions:
        item = {
            "id": prediction["id"],
            "prediction_text": prediction["prediction_text"],
        }
        if version_2_with_negative:
            item["no_answer_probability"] = float(prediction.get("no_answer_probability", 0.0))
        normalized_predictions.append(item)
    result = metric.compute(predictions=normalized_predictions, references=references)
    return {k: float(v) for k, v in result.items()}


def unpack_qa_predictions(predictions: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(predictions, tuple) and len(predictions) == 2:
        return predictions[0], predictions[1]

    if isinstance(predictions, list) and len(predictions) == 2:
        return predictions[0], predictions[1]

    if isinstance(predictions, np.ndarray):
        if predictions.ndim >= 2 and predictions.shape[0] == 2:
            return predictions[0], predictions[1]
        if predictions.ndim >= 3 and predictions.shape[1] == 2:
            return predictions[:, 0, ...], predictions[:, 1, ...]

    raise ValueError(
        "Expected QA predictions as (start_logits, end_logits); "
        f"got type={type(predictions).__name__}"
        + (f", shape={predictions.shape}" if hasattr(predictions, "shape") else "")
    )


def run_postprocessed_eval(
    trainer: Trainer,
    dataset_name: str,
    version_2_with_negative: bool,
    raw_examples: Dataset,
    eval_examples: Dataset,
    eval_features_for_postprocess: Dataset,
    prefix: str,
) -> dict[str, float]:
    if len(eval_examples) == 0 or len(eval_features_for_postprocess) == 0:
        if dataset_name == "mandarjoshi/trivia_qa":
            qa_scores = evaluate_triviaqa({}, build_triviaqa_reference_map_from_raw(raw_examples))
        elif dataset_name in {"rajpurkar/squad", "rajpurkar/squad_v2"}:
            qa_scores = evaluate_squad_family(
                predictions=[],
                references=[],
                version_2_with_negative=version_2_with_negative,
            )
        else:
            raise ValueError(f"Unsupported dataset_name={dataset_name!r}")

        return {f"{prefix}_{k}": float(v) for k, v in qa_scores.items()}

    def predict_for_postprocessing():
        removable = [c for c in ["example_id", "offset_mapping", "question_id"] if c in eval_features_for_postprocess.column_names]
        eval_features_for_predict = eval_features_for_postprocess.remove_columns(removable)

        original_drop_last = trainer.args.dataloader_drop_last
        trainer.args.dataloader_drop_last = False
        try:
            return trainer.predict(eval_features_for_predict, metric_key_prefix=prefix)
        finally:
            trainer.args.dataloader_drop_last = original_drop_last
            del eval_features_for_predict
            shutdown_trainer_dataloader(trainer, "_test_dataloader")
            release_memory()

    pred_output = predict_for_postprocessing()

    if pred_output.predictions is None:
        if dataset_name == "mandarjoshi/trivia_qa":
            qa_scores = evaluate_triviaqa({}, build_triviaqa_reference_map_from_raw(raw_examples))
        elif dataset_name in {"rajpurkar/squad", "rajpurkar/squad_v2"}:
            qa_scores = evaluate_squad_family(
                predictions=[],
                references=[],
                version_2_with_negative=version_2_with_negative,
            )
        else:
            raise ValueError(f"Unsupported dataset_name={dataset_name!r}")

        combined = {f"{prefix}_{k}": float(v) for k, v in qa_scores.items()}
        for key, value in (pred_output.metrics or {}).items():
            if isinstance(value, (int, float)):
                combined[key] = float(value)
        pred_output = None
        release_memory()
        return combined

    all_start_logits, all_end_logits = unpack_qa_predictions(pred_output.predictions)
    aligned_count = min(len(eval_features_for_postprocess), len(all_start_logits), len(all_end_logits))

    eval_features_for_postprocess = eval_features_for_postprocess.select(range(aligned_count))
    raw_predictions = (all_start_logits[:aligned_count], all_end_logits[:aligned_count])

    if dataset_name == "mandarjoshi/trivia_qa":
        predictions, _ = postprocess_triviaqa_predictions(
            examples=eval_examples,
            features=eval_features_for_postprocess,
            raw_predictions=raw_predictions,
            version_2_with_negative=version_2_with_negative,
        )
        pred_map = aggregate_predictions_by_question_id(predictions)
        ref_map = build_triviaqa_reference_map_from_raw(raw_examples)
        qa_scores = evaluate_triviaqa(pred_map, ref_map)
    elif dataset_name in {"rajpurkar/squad", "rajpurkar/squad_v2"}:
        predictions, references = postprocess_squad_predictions(
            examples=eval_examples,
            features=eval_features_for_postprocess,
            raw_predictions=raw_predictions,
            version_2_with_negative=version_2_with_negative,
        )
        qa_scores = evaluate_squad_family(
            predictions=predictions,
            references=references,
            version_2_with_negative=version_2_with_negative,
        )
    else:
        raise ValueError(f"Unsupported dataset_name={dataset_name!r}")

    combined = {f"{prefix}_{k}": float(v) for k, v in qa_scores.items()}
    for key, value in (pred_output.metrics or {}).items():
        if isinstance(value, (int, float)):
            combined[key] = float(value)
    pred_output = None
    raw_predictions = None
    all_start_logits = None
    all_end_logits = None
    release_memory()
    return combined
