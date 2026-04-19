from normalize_text import normalize_for_answer_matching
from normalize_text import _build_normalized_char_map
from typing import Optional


def find_exact_raw_span(context: str, answer: str) -> Optional[tuple[int, int, str]]:
    start = context.find(answer)
    if start == -1:
        return None
    end = start + len(answer)
    return start, end, context[start: end]


def find_loose_span(context: str, answer: str):
    norm_context = normalize_for_answer_matching(context)
    norm_answer = normalize_for_answer_matching(answer)

    idx = norm_context.find(norm_answer)
    if idx == -1:
        return None

    # fallback approximation:
    raw_idx = context.lower().find(answer.lower())
    if raw_idx == -1:
        return None

    return raw_idx, raw_idx + len(answer), context[raw_idx:raw_idx + len(answer)]


def find_first_matching_answer_span(context: str, answers: list[str]):
    answers = sorted(answers, key=len, reverse=True)

    for ans in answers:
        exact = find_exact_raw_span(context, ans)
        if exact:
            return exact[0], exact[1], exact[2], ans

    for ans in answers:
        loose = find_loose_span(context, ans)
        if loose:
            return loose[0], loose[1], loose[2], ans

    return None


def find_normalized_span(
    context: str,
    answer: str
) -> Optional[tuple[int, int, str]]:
    norm_context, norm_to_orig = _build_normalized_char_map(context)
    norm_answer = normalize_for_answer_matching(answer)

    if not norm_context or not norm_answer:
        return None

    match_start = norm_context.find(norm_answer)
    if match_start == -1:
        return None

    match_end = match_start + len(norm_answer) - 1
    orig_start = norm_to_orig[match_start]
    orig_end = norm_to_orig[match_end] + 1
    return orig_start, orig_end, context[orig_start:orig_end]


def find_best_answer_span(
    context: str,
    candidate_answers: list[str],
    keep_apostrophes: bool = True,
) -> Optional[tuple[int, int, str, str]]:
    """
    Deterministic policy:
    1. deduplicate by normalized form
    2. try longer answers first
    3. exact raw match first
    4. normalized char-map match second
    """
    deduped = []
    seen = set()
    for ans in candidate_answers:
        if not ans:
            continue
        key = normalize_for_answer_matching(ans)
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ans)

    deduped.sort(key=lambda x: len(normalize_for_answer_matching(x)), reverse=True)

    for ans in deduped:
        exact = find_exact_raw_span(context, ans)
        if exact is not None:
            s, e, txt = exact
            return s, e, txt, ans

    for ans in deduped:
        loose = find_normalized_span(context, ans)
        if loose is not None:
            s, e, txt = loose
            return s, e, txt, ans

    return None
