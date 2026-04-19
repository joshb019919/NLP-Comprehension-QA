# NLP-Comprehension-QA

## Overview

This exists as a repository for ten experiments performed on Huggingface models and datasets in Python.  The main experiments are using TriviaQA to fine-tune BERT, and the other three experiments are fine-tuning BERT or DistilBERT on SQuAD 1.1 or SQuAD 2.0.  If the model was tuned on TriviaQA, it was tested against SQuAD 1.1.  If it was tuned on SQuAD 1.1 or 2.0, it was tested against TriviaQA.

This repo contains the configurations for downloading and fine-tuning BioBERT, too, but no experiment is set up for it and it has not been tested.

## Important Paths

/ (root)
- Smoke-test runner, `run.sh`.
- Full experiment runner, `run_experiments.sh`.
- Virtual environment setup runners for Windows or Linux, `setup_venv_bertqa.bat` and `setup_venv_bertqa.sh`.
- Special scripts to run validation or testing separately, in case these runs fail for any experiment.

plotter/

- Matplotlib plotting engine, `plot_benchmarks.py`.

results/

- A JSON file detailing experiment settings.
- Figures visualizing exact match, f1, and loss scores for each experiment.
- Figures plotting metric scores against inverted, normalized loss.
- An analysis text detailing any importance of comparing loss and metrics.

src/

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
