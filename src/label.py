from find_spans import find_best_answer_span
from normalize_text import normalize_for_answer_matching
from normalize_text import normalize_text
from typing import Any


def _extract_contexts(source: Any, context_key: str) -> list[str]:
    """
    Normalize nested TriviaQA page/result structures produced by HF datasets.

    In batched mode, sequence-of-struct columns may arrive either as:
    - list[dict[str, Any]]
    - dict[str, list[Any]]
    """
    if not source:
        return []

    if isinstance(source, dict):
        values = source.get(context_key, [])
        if isinstance(values, list):
            return [value for value in values if isinstance(value, str)]
        if isinstance(values, str):
            return [values]
        return []

    if isinstance(source, list):
        contexts = []
        for item in source:
            if isinstance(item, dict):
                value = item.get(context_key, "")
                if isinstance(value, str):
                    contexts.append(value)
            elif isinstance(item, str):
                contexts.append(item)
        return contexts

    if isinstance(source, str):
        return [source]

    return []


def expand_and_label_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """
    Input: batch of original TriviaQA examples
    Output: expanded rows, one per question-context pair that contains an answer
    """
    out = {
        "id": [],
        "question_id": [],
        "question": [],
        "context": [],
        "source_type": [],
        "answers": [],
        "matched_alias_source": [],
        "all_answers": [],
    }

    n = len(batch["question"])

    for i in range(n):
        question = normalize_text(batch["question"][i])
        qid = batch["question_id"][i]
        answer = batch["answer"][i]

        candidate_answers = []
        if answer.get("value"):
            candidate_answers.append(normalize_text(answer["value"]))
        for a in answer.get("aliases", []) or []:
            if a:
                candidate_answers.append(normalize_text(a))

        # dedupe candidate answers by normalized matching form
        seen = set()
        deduped_answers = []
        for a in candidate_answers:
            key = normalize_for_answer_matching(a)
            if key and key not in seen:
                seen.add(key)
                deduped_answers.append(a)

        entity_pages = _extract_contexts(batch["entity_pages"][i], "wiki_context")
        search_results = _extract_contexts(batch["search_results"][i], "search_context")

        # wiki contexts
        for j, context in enumerate(entity_pages):
            if not context or not context.strip():
                continue

            match = find_best_answer_span(context, deduped_answers)
            if match is None:
                continue

            start, end, matched_text, matched_alias = match
            out["all_answers"].append(deduped_answers)
            out["id"].append(f"{qid}_wiki_{j}")
            out["question_id"].append(qid)
            out["question"].append(question)
            out["context"].append(context)
            out["source_type"].append("wiki")
            out["answers"].append({
                "text": [matched_text],
                "answer_start": [start],
            })
            out["matched_alias_source"].append(matched_alias)

        # search contexts
        for j, context in enumerate(search_results):
            if not context or not context.strip():
                continue

            match = find_best_answer_span(context, deduped_answers)
            if match is None:
                continue

            start, end, matched_text, matched_alias = match
            out["all_answers"].append(deduped_answers)
            out["id"].append(f"{qid}_search_{j}")
            out["question_id"].append(qid)
            out["question"].append(question)
            out["context"].append(context)
            out["source_type"].append("search")
            out["answers"].append({
                "text": [matched_text],
                "answer_start": [start],
            })
            out["matched_alias_source"].append(matched_alias)

    return out
