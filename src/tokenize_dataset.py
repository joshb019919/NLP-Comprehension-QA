from typing import Any


def prepare_train_features_triviaqa(
    examples: dict[str, list[Any]],
    tokenizer,
    max_length: int = 384,
    doc_stride: int = 64,
) -> dict[str, list[Any]]:
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )

    sample_mapping = tokenized["overflow_to_sample_mapping"]
    offset_mapping = tokenized["offset_mapping"]

    example_ids = []
    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized.sequence_ids(i)
        sample_idx = sample_mapping[i]
        example_ids.append(examples["id"][sample_idx])

        answers = examples["answers"][sample_idx]
        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if offsets[token_start_index][0] > start_char or offsets[token_end_index][1] < end_char:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        start_positions.append(token_start_index - 1)

        while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        end_positions.append(token_end_index + 1)

    tokenized["example_id"] = example_ids
    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    tokenized.pop("offset_mapping")
    return tokenized


def prepare_validation_features_triviaqa(
    examples: dict[str, list[Any]],
    tokenizer,
    max_length: int = 384,
    doc_stride: int = 64,
) -> dict[str, list[Any]]:
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )

    sample_mapping = tokenized["overflow_to_sample_mapping"]
    offset_mapping = tokenized["offset_mapping"]

    example_ids = []
    question_ids = []
    masked_offset_mapping = []

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized.sequence_ids(i)
        sample_idx = sample_mapping[i]

        example_ids.append(examples["id"][sample_idx])
        question_ids.append(examples["question_id"][sample_idx])

        masked_offsets = []
        for k, offset in enumerate(offsets):
            if sequence_ids[k] == 1:
                masked_offsets.append(offset)
            else:
                masked_offsets.append(None)
        masked_offset_mapping.append(masked_offsets)

        answers = examples["answers"][sample_idx]
        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if offsets[token_start_index][0] > start_char or offsets[token_end_index][1] < end_char:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        start_positions.append(token_start_index - 1)

        while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        end_positions.append(token_end_index + 1)

    tokenized["example_id"] = example_ids
    tokenized["question_id"] = question_ids
    tokenized["offset_mapping"] = masked_offset_mapping
    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


def prepare_test_features_triviaqa(
    examples: dict[str, list[Any]],
    tokenizer,
    max_length: int = 384,
    doc_stride: int = 64,
) -> dict[str, list[Any]]:
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )

    sample_mapping = tokenized["overflow_to_sample_mapping"]
    offset_mapping = tokenized["offset_mapping"]

    example_ids = []
    question_ids = []
    masked_offset_mapping = []

    for i, offsets in enumerate(offset_mapping):
        sequence_ids = tokenized.sequence_ids(i)
        sample_idx = sample_mapping[i]

        example_ids.append(examples["id"][sample_idx])
        question_ids.append(examples["question_id"][sample_idx])

        masked_offsets = []
        for k, offset in enumerate(offsets):
            if sequence_ids[k] == 1:
                masked_offsets.append(offset)
            else:
                masked_offsets.append(None)
        masked_offset_mapping.append(masked_offsets)

    tokenized["example_id"] = example_ids
    tokenized["question_id"] = question_ids
    tokenized["offset_mapping"] = masked_offset_mapping
    return tokenized


def prepare_train_features_squad(
    examples: dict[str, list[Any]],
    tokenizer,
    max_length: int = 384,
    doc_stride: int = 128,
    version_2_with_negative: bool = False,
) -> dict[str, list[Any]]:
    questions = [q.lstrip() for q in examples["question"]]

    tokenized = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )

    sample_mapping = tokenized["overflow_to_sample_mapping"]
    offset_mapping = tokenized["offset_mapping"]

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized.sequence_ids(i)
        sample_idx = sample_mapping[i]

        answers = examples["answers"][sample_idx]

        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if offsets[token_start_index][0] > start_char or offsets[token_end_index][1] < end_char:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        start_positions.append(token_start_index - 1)

        while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        end_positions.append(token_end_index + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    tokenized.pop("offset_mapping")
    return tokenized


def prepare_validation_features_squad(
    examples: dict[str, list[Any]],
    tokenizer,
    max_length: int = 384,
    doc_stride: int = 128,
    version_2_with_negative: bool = False,
) -> dict[str, list[Any]]:
    questions = [q.lstrip() for q in examples["question"]]

    tokenized = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )

    sample_mapping = tokenized["overflow_to_sample_mapping"]
    offset_mapping = tokenized["offset_mapping"]

    example_ids = []
    masked_offset_mapping = []

    for i, offsets in enumerate(offset_mapping):
        sequence_ids = tokenized.sequence_ids(i)
        sample_idx = sample_mapping[i]

        example_ids.append(examples["id"][sample_idx])

        masked_offsets = []
        for k, offset in enumerate(offsets):
            if sequence_ids[k] == 1:
                masked_offsets.append(offset)
            else:
                masked_offsets.append(None)
        masked_offset_mapping.append(masked_offsets)

    tokenized["example_id"] = example_ids
    tokenized["offset_mapping"] = masked_offset_mapping
    return tokenized


def prepare_test_features_squad(
    examples: dict[str, list[Any]],
    tokenizer,
    max_length: int = 384,
    doc_stride: int = 128,
    version_2_with_negative: bool = False,
) -> dict[str, list[Any]]:
    return prepare_validation_features_squad(
        examples=examples,
        tokenizer=tokenizer,
        max_length=max_length,
        doc_stride=doc_stride,
        version_2_with_negative=version_2_with_negative,
    )
