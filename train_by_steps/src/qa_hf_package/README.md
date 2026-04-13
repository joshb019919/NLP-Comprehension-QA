# Hugging Face Extractive QA Package

A conservative, disk-backed package for fine-tuning and evaluating Hugging Face extractive QA models on:

- `trivia_qa` with config `rc`
- `squad`
- `squad_v2`

Included model configs:

- `bert-base-uncased` (default)
- `distilbert-base-uncased`
- `dmis-lab/biobert-base-cased-v1.1`

## Highlights

- CUDA assumed present and used automatically
- Sets:
  - `torch.backends.cudnn.benchmark = True`
  - `torch.backends.cuda.matmul.allow_tf32 = True`
  - `torch.backends.cudnn.allow_tf32 = True`
- Disk-backed dataset and tokenized caches under `/data/data/huggingface`
- Conservative defaults for 12 GB system RAM and RTX 3060 12 GB VRAM
- Non-parallel defaults, with one config knob each for:
  - `streaming`
  - `torch_compile`
  - `preprocessing_num_proc`
  - `dataloader_num_workers`
- Optional process memory limit
- `bench.jsonl` written during training/eval with sequential integer keys
- TriviaQA expansion into one row per context, with alias-aware matching

## Layout

- `train_qa.py` — main entry point
- `src/` — implementation
- `configs/models/` — model configs
- `configs/datasets/` — dataset configs
- `configs/runs/` — ready-to-run merged references

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run examples

TriviaQA RC with BERT:

```bash
python train_qa.py --run-config configs/runs/bert_trivia_qa_rc.json
```

Small smoke test using max examples override:

```bash
python train_qa.py \
  --run-config configs/runs/bert_trivia_qa_rc.json \
  --set max_examples=512 \
  --set output_name=bert_trivia_smoke
```

SQuAD v2 with DistilBERT:

```bash
python train_qa.py --run-config configs/runs/distilbert_squad_v2.json
```

BioBERT on SQuAD 1.1:

```bash
python train_qa.py --run-config configs/runs/biobert_squad.json
```

## Notes

### Streaming

Set `streaming` in the run config to `true` or `false`.

- `false` is the default and gives full disk-backed caching, save-to-disk tokenized datasets, and the most predictable Trainer behavior.
- `true` uses HF streaming for dataset reads. In streaming mode, tokenized datasets are not saved with `save_to_disk()` because iterable datasets do not support it the same way.

### Torch compile

Set `torch_compile` in the run config.

Default is `false`.

### Raising process counts

Use:

- `preprocessing_num_proc`
- `dataloader_num_workers`

These are independent knobs.

### Dynamic shard sizing

`save_max_shard_size` is computed dynamically per save target as approximately half the estimated size of one preprocessing chunk, rather than being set statically.

### TriviaQA handling

The package expands each TriviaQA item into multiple rows: one row per wiki or search context. It carries the question, one context, and normalized aliases including the canonical answer value.

For extractive training labels, it tries to locate an answer span in the context using the canonical answer and aliases. Rows without a match are excluded from training by default, but they can still be evaluated or predicted on.

### Download and cache recovery

- Existing valid caches are reused.
- If tokenized cache directories are incomplete, they are rebuilt.
- If dataset/model download caches appear broken because a load fails, the code retries with a forced redownload.

## Output directories

By default, all Hugging Face and package data are rooted at:

```text
/data/data/huggingface
```

Important subfolders include:

- `/data/data/huggingface/datasets`
- `/data/data/huggingface/models`
- `/data/data/huggingface/tokenized`
- `/data/data/huggingface/runs`
- `/data/data/huggingface/bench`

## Bench log format

`bench.jsonl` contains one JSON object per line. Each line has a single top-level sequential key beginning at `0`, for example:

```json
{"0": {"event": "log", "step": 10, "loss": 2.1, "samples_per_second": 12.3}}
```

## CLI override format

You can override scalar config values with repeated `--set key=value` arguments.

Examples:

```bash
python train_qa.py \
  --run-config configs/runs/bert_squad.json \
  --set batch_size=8 \
  --set preprocessing_num_proc=2 \
  --set streaming=false
```
