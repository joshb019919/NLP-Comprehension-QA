# Z-scored Shape Comparison Notes

Each plot overlays three z-scored series at eval checkpoints:
- inverted smoothed training loss
- eval Exact
- eval F1

Because the loss curve is inverted before z-scoring, upward movement means improvement for all three lines.

## exp01
Moderate alignment: the broad phases mostly match, but there are noticeable local divergences
  between loss and eval metrics (corr exact=0.74, corr f1=0.70).

## exp02
Strong alignment: the smoothed improvement-from-loss curve co-moves well with eval Exact/F1 overall
  (corr exact=0.80, corr f1=0.76).

## exp03
Strong alignment: the smoothed improvement-from-loss curve co-moves well with eval Exact/F1 overall
  (corr exact=0.82, corr f1=0.86).

## exp04
Weak alignment: loss shape and eval-metric shape diverge enough that loss alone would be a poor
  proxy here (corr exact=-0.87, corr f1=-0.98).

## exp05
Moderate alignment: the broad phases mostly match, but there are noticeable local divergences
  between loss and eval metrics (corr exact=0.66, corr f1=0.59).

## exp06
Moderate alignment: the broad phases mostly match, but there are noticeable local divergences
  between loss and eval metrics (corr exact=0.69, corr f1=0.57).

## exp07
Weak alignment: loss shape and eval-metric shape diverge enough that loss alone would be a poor
  proxy here (corr exact=-0.35, corr f1=0.13).

## exp08
Strong alignment: the smoothed improvement-from-loss curve co-moves well with eval Exact/F1 overall
  (corr exact=0.78, corr f1=0.85).

## exp09
Strong alignment: the smoothed improvement-from-loss curve co-moves well with eval Exact/F1 overall
  (corr exact=0.92, corr f1=0.85).

## exp10
Weak alignment: loss shape and eval-metric shape diverge enough that loss alone would be a poor
  proxy here (corr exact=-0.42, corr f1=-0.01).
