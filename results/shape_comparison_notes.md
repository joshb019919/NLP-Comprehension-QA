# Z-scored Shape Comparison Notes

Each plot overlays three z-scored series at eval checkpoints:
- inverted smoothed training loss
- eval Exact
- eval F1

Because the loss curve is inverted before z-scoring, upward movement means improvement for all three lines.

## exp01
Strong alignment: the smoothed improvement-from-loss curve co-moves well with eval Exact/F1 overall
  (corr exact=0.84, corr f1=0.88).

## exp02
Strong alignment: the smoothed improvement-from-loss curve co-moves well with eval Exact/F1 overall
  (corr exact=0.82, corr f1=0.84).

## exp03
Strong alignment: the smoothed improvement-from-loss curve co-moves well with eval Exact/F1 overall
  (corr exact=0.81, corr f1=0.82).

## exp04
Weak alignment: loss shape and eval-metric shape diverge enough that loss alone would be a poor
  proxy here (corr exact=0.32, corr f1=-0.86).

## exp05
Strong alignment: the smoothed improvement-from-loss curve co-moves well with eval Exact/F1 overall
  (corr exact=0.77, corr f1=0.80).

## exp06
Strong alignment: the smoothed improvement-from-loss curve co-moves well with eval Exact/F1 overall
  (corr exact=0.81, corr f1=0.84).

## exp07
Insufficient variation: one or more z-scored series is nearly flat, so shape correlation is not very
  informative here.

## exp08
Strong alignment: the smoothed improvement-from-loss curve co-moves well with eval Exact/F1 overall
  (corr exact=0.76, corr f1=0.85).

## exp09
Strong alignment: the smoothed improvement-from-loss curve co-moves well with eval Exact/F1 overall
  (corr exact=0.77, corr f1=0.74).

## exp10
Moderate alignment: the broad phases mostly match, but there are noticeable local divergences
  between loss and eval metrics (corr exact=0.51, corr f1=0.69).
