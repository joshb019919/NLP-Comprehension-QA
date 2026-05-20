# NLP-Comprehension-QA
A natural language processing tool using BERT transformer to perform reading comprehension question-answering on passages.

## Experiment Performance Report

This report compares experiments using:

- `Training-validation`: the best logged in-training validation snapshot from `bench.jsonl`
- `Final validation`: the postprocessed final validation run from `final.jsonl`
- `Test validation`: the postprocessed final test run from `final.jsonl`

Primary comparison is by F1, with exact match included as supporting context.

### Quick Ranking

#### Training-validation F1

1. `exp08` - `61.634`
2. `exp10` - `51.007`
3. `exp05` - `8.645`
4. `exp01` - `8.105`
5. `exp06` - `8.043`
6. `exp02` - `7.941`
7. `exp03` - `7.890`
8. `exp09` - `5.959`
9. `exp04` - `2.260`
10. `exp07` - `0.940`

#### Final validation F1

1. `exp08` - `61.624`
2. `exp10` - `50.717`
3. `exp01` - `7.963`
4. `exp02` - `7.840`
5. `exp03` - `7.642`
6. `exp05` - `7.522`
7. `exp06` - `7.453`
8. `exp09` - `5.929`
9. `exp04` - `1.939`
10. `exp07` - `0.710`

### Final test F1

1. `exp05` - `19.993`
2. `exp01` - `19.929`
3. `exp02` - `19.410`
4. `exp06` - `16.528`
5. `exp03` - `14.988`
6. `exp09` - `12.240`
7. `exp08` - `8.628`
8. `exp04` - `5.843`
9. `exp10` - `5.086`
10. `exp07` - `0.531`

### Experiment Notes

#### `exp01`

Baseline setup:

- epochs: `2`
- learning rate: `5e-5`
- weight decay: `1e-8`
- gradient norm clip: `1.0`
- optimizer: `adamw_torch_fused`

Performance:

- training-validation: exact `6.084`, F1 `8.105`
- final validation: exact `6.054`, F1 `7.963`
- test validation: exact `14.257`, F1 `19.929`

Interpretation:

- Strong baseline for the TriviaQA-trained BERT runs.
- It stayed near the top on final validation and was second-best on final test.

#### `exp02`

What changed from `exp01`:

- gradient norm clip: `0.5` instead of `1.0`

Effect:

- training-validation: worse than `exp01` (`7.941` vs `8.105` F1)
- final validation: worse than `exp01` (`7.840` vs `7.963` F1)
- test validation: worse than `exp01` (`19.410` vs `19.929` F1)

Interpretation:

- Tightening clipping from `1.0` to `0.5` mildly hurt all three tracked outcomes.

#### `exp03`

What changed from `exp02`:

- gradient norm clip: `0` instead of `0.5`

Effect:

- training-validation: slightly worse than `exp02` (`7.890` vs `7.941` F1)
- final validation: worse than `exp02` (`7.642` vs `7.840` F1)
- test validation: much worse than `exp02` (`14.988` vs `19.410` F1)

Interpretation:

- Removing clipping hurt generalization, especially on the final test set.

#### `exp04`

What changed from `exp03`:

- optimizer: `sgd` instead of `adamw_torch_fused`
- gradient norm clip: back to `1.0`

Effect:

- training-validation: far worse than `exp03` (`2.260` vs `7.890` F1)
- final validation: far worse than `exp03` (`1.939` vs `7.642` F1)
- test validation: far worse than `exp03` (`5.843` vs `14.988` F1)

Interpretation:

- Switching to SGD was a major regression for this setup.

#### `exp05`

What changed from `exp04`:

- epochs: `3` instead of `2`
- optimizer: back to `adamw_torch_fused`

Effect:

- training-validation: much better than `exp04` (`8.645` vs `2.260` F1)
- final validation: much better than `exp04` (`7.522` vs `1.939` F1)
- test validation: much better than `exp04` (`19.993` vs `5.843` F1)

Interpretation:

- This was the best final test run overall.
- More training helped in-training validation, but final validation still did not beat the original baseline.

#### `exp06`

What changed from `exp05`:

- epochs: `2` instead of `3`
- weight decay: `1e-7` instead of `1e-8`

Effect:

- training-validation: worse than `exp05` (`8.043` vs `8.645` F1)
- final validation: slightly worse than `exp05` (`7.453` vs `7.522` F1)
- test validation: worse than `exp05` (`16.528` vs `19.993` F1)

Interpretation:

- Higher weight decay plus dropping back to 2 epochs reduced performance across the board.

#### `exp07`

What changed from `exp06`:

- learning rate: `5e-4` instead of `5e-5`
- weight decay: `1e-7` instead of `1e-8`

Effect:

- training-validation: dramatically worse than `exp06` (`0.940` vs `8.043` F1)
- final validation: dramatically worse than `exp06` (`0.710` vs `7.453` F1)
- test validation: dramatically worse than `exp06` (`0.531` vs `16.528` F1)

Interpretation:

- The 10x learning-rate increase destabilized this configuration.

#### `exp08`

What changed from `exp07`:

- training/eval data: `SQuAD 2.0` instead of `TriviaQA`
- test data: `TriviaQA` instead of `SQuAD 1.1`
- learning rate: back to `5e-5`
- weight decay: back to `1e-8`
- model remains `BERT-base`
- added `null_score_diff_threshold: 2.0`

Effect:

- training-validation: much better than `exp07` (`61.634` vs `0.940` F1)
- final validation: much better than `exp07` (`61.624` vs `0.710` F1)
- test validation: better than `exp07` (`8.628` vs `0.531` F1), but much worse than the best TriviaQA-trained BERT test results

Interpretation:

- This was the best run on SQuAD 2.0 validation by a large margin.
- Its cross-dataset final test result on TriviaQA was much lower than the best TriviaQA-trained runs.

#### `exp09`

What changed from `exp08`:

- model: `DistilBERT` instead of `BERT-base`
- training/eval data: back to `TriviaQA`
- test data: back to `SQuAD 1.1`
- batch size: `128` instead of `64`
- removed `null_score_diff_threshold`

Effect:

- training-validation: much worse than `exp08` (`5.959` vs `61.634` F1)
- final validation: much worse than `exp08` (`5.929` vs `61.624` F1)
- test validation: better than `exp08` (`12.240` vs `8.628` F1)

Interpretation:

- Relative to `exp08`, the task/domain change dominates the comparison.
- Within the TriviaQA-trained family, DistilBERT underperformed the BERT baseline on both validation and test.

#### `exp10`

What changed from `exp09`:

- training/eval data: `SQuAD 2.0` instead of `TriviaQA`
- test data: `TriviaQA` instead of `SQuAD 1.1`
- added `null_score_diff_threshold: 2.0`

Effect:

- training-validation: much better than `exp09` (`51.007` vs `5.959` F1)
- final validation: much better than `exp09` (`50.717` vs `5.929` F1)
- test validation: worse than `exp09` (`5.086` vs `12.240` F1)

Interpretation:

- DistilBERT on SQuAD 2.0 validated strongly on its own domain, but transferred poorly to the TriviaQA test set.
- It trailed `exp08` on all three metrics, so BERT-base remained the stronger SQuAD 2.0 configuration here.

### Bottom Line

- Best TriviaQA-family final test result: `exp05`
- Best TriviaQA-family final validation result: `exp01`
- Best SQuAD 2.0-family validation result: `exp08`
- Strongest evidence of harmful changes:
  - `exp04`: switching to SGD
  - `exp07`: raising learning rate to `5e-4`
- Strongest evidence of helpful change for test transfer inside the BERT TriviaQA family:
  - `exp05`: training for `3` epochs
