# NLP-Comprehension-QA

## Overview

This exists as a repository for ten experiments performed on Huggingface models and datasets in Python.  The main experiments are using TriviaQA to fine-tune BERT, and the other three experiments are fine-tuning BERT or DistilBERT on SQuAD 1.1 or SQuAD 2.0.  If the model was tuned on TriviaQA, it was tested against SQuAD 1.1.  If it was tuned on SQuAD 1.1 or 2.0, it was tested against TriviaQA.

This repo contains the configurations for downloading and fine-tuning BioBERT, too, but no experiment is set up for it and it has not been tested.

*All F1 and exact match results are very low due to using only a small subset of training examples (20000) and eval and test examples (5000).*

## Huggingface Models and Datasets

### Models

- BERT (bert-base-uncased).
  - Basic transformer-based natural language processing model.
  - Text vectors with self-supervised learning.
  - Originally pre-trained on Wikipedia and BookCorpus.
- DistilBERT (distilbert-base-uncased).
  - Smaller and faster than BERT, with small accuracy loss.
  - No token-type embeddings (fast and lightweight, poor at next-sentence prediction).
- BioBERT (dmis-lab/biobert-base-cased-v1.1).
  - Used in biomedical research learning.
  - Pretrained on PubMed and PMC articles.
 
### Datasets

- TriviaQA (mandarjoshi/trivia_qa).
  - 173K examples (train, validation, test splits).
- SQuAD 1.1 (rajpurkar/squad).
  - 100K examples (train, validation splits, all answerable).
- SQuAD 2.0 (rajpurkar/squad_v2).
  - 150K examples (SQuAD 1.1 + 50K unanaswerable, train and validation).
 
### Terms
 
- SQuAD --- Stanford Question and Answering Dataset.
- BERT --- bidirectional encoder representations from transformers.
- BioBERT --- BERT for Biomedical Text Mining.
- EM --- exact match metric for perfectly matching predicted answer text to ground truth for that example.

## Important Paths

### / (root)

- Smoke-test runner, `run.sh`.
- Full experiment runner, `run_experiments.sh`.
- Virtual environment setup runners for Windows or Linux, `setup_venv_bertqa.bat` and `setup_venv_bertqa.sh`.
- Special scripts to run validation or testing separately, in case these runs fail for any experiment.

### plotter/

- Matplotlib plotting engine, `plot_benchmarks.py`.

### results/

- A JSON file detailing experiment settings.
- Figures visualizing exact match, f1, and loss scores for each experiment.
- Figures plotting metric scores against inverted, normalized loss.
- An analysis text detailing any importance of comparing loss and metrics.

### src/

- JSON configurations for each model, datset, and experiment run.
- Python scripts for running the experiments.
  - bench.py (logs benchmarks).
  - common.py (contains lines common to all scripts).
  - config.py (contains default configurations overridden by JSON).
  - evaluation.py (controls eval runs).
  - find_spans.py (discovers and saves answer spans).
  - label.py (expands and labels each example row).
  - main.py (main entrypoint).
  - normalize_text.py (ensures consistency).
  - paths.py (specific to each machine, where to save/load things).
  - postprocess.py (mid-training and post-training metric wiring).
  - qa_trainer.py (custom transformers-inheriting Trainer class).
  - test.py (governs test runs).
  - tokenize_dataset.py (tokenizes dataset and saves to path).
  - train.py (main training logic).

The directory pathways in paths.py must be altered to fit your own setup and desired save location(s).

## Main Entrypoints

### Run an individual experiment:

- `python src/main.py [--run-config src/configs/runs/<run-config>.json] [--set <override-config-name>=<value>]`.

### Run the smoke test:

- `./run.sh`.

### Run all experiments:

- `./run_experiments.sh`.

### Run validation only over all experiments:

- `python val_only.py`.

### Run test only over all experiments:

- `python test_only.py`.

## Brief Results

All seeds are set to *The Answer*: 42.

### Poor Results

#### Experiment 4

Used stochastic gradient descent with momentum.  Loss reveals early peak, SGD with momentum is not the appropriate optimizer for transformers.

#### Experiment 7

Uses aggressive learning rate and weight decay.  Loss peaks early and remains high.  No real improvement in EM or F1 over baseline.

### Moderate Results

#### Experiment 1

Exists as a baseline with standard hyperparameters.  LR 1e-5, weight decay 1e-8, gradient clip [-1, 1], and AdamW Torch Fused optimizer.

#### Experiment 2

Slightly better results than experiment 1 by clipping all gradients to [-0.5, 0.5].

#### Experiment 3



#### Experiment 5



#### Experiment 10


### Excellent Results

#### Experiment 6



#### Experiment 8



#### Experiment 9

