import re
import unicodedata
from typing import List, Optional, Tuple


def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", text)

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }

    for src, dst in replacements.items():
        text = text.replace(src, dst)

    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_for_answer_matching(text: str) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"_", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_normalized_char_map(
    text: str,
    keep_apostrophes: bool = True,
) -> Tuple[str, List[int]]:
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    out_chars: List[str] = []
    norm_to_orig: List[int] = []
    prev_space = True

    for orig_idx, ch in enumerate(text):
        ch = ch.lower()
        if ch.isalnum() or ch == "_":
            out_chars.append(ch)
            norm_to_orig.append(orig_idx)
            prev_space = False
        elif keep_apostrophes and ch == "'":
            out_chars.append(ch)
            norm_to_orig.append(orig_idx)
            prev_space = False
        else:
            if not prev_space:
                out_chars.append(" ")
                norm_to_orig.append(orig_idx)
                prev_space = True

    start = 0
    while start < len(out_chars) and out_chars[start] == " ":
        start += 1
    end = len(out_chars)
    while end > start and out_chars[end - 1] == " ":
        end -= 1

    return "".join(out_chars[start:end]), norm_to_orig[start:end]
