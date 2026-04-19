from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding


def load_qa_raw_split(
    dataset_name: str,
    dataset_config_name: str | None,
    split: str,
    max_examples: int | None = None,
    seed: int = 42,
):
    """
    Load one QA split from TriviaQA, SQuAD 1.1, or SQuAD 2.0.

    Supported:
      - mandarjoshi/trivia_qa + rc
      - rajpurkar/squad
      - rajpurkar/squad_v2
    """
    if dataset_name == "mandarjoshi/trivia_qa":
        raw_ds = load_dataset(dataset_name, dataset_config_name or "rc", split=split)
    elif dataset_name == "rajpurkar/squad":
        raw_ds = load_dataset("rajpurkar/squad", split=split)
    elif dataset_name == "rajpurkar/squad_v2":
        raw_ds = load_dataset("rajpurkar/squad_v2", split=split)
    else:
        raise ValueError(
            f"Unsupported dataset_name={dataset_name!r}. "
            "Expected one of: 'mandarjoshi/trivia_qa', 'rajpurkar/squad', 'rajpurkar/squad_v2'."
        )

    if max_examples is not None:
        subset_size = min(max_examples, len(raw_ds))
        raw_ds = raw_ds.shuffle(seed=seed).select(range(subset_size))

    return raw_ds


def build_tokenizer_and_collator(
    model_name: str,
    cache_dir: str | None = None,
    revision: str | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        cache_dir=cache_dir,
        revision=revision,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )
    return tokenizer, data_collator
