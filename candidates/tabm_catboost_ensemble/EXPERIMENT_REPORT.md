# TabM v1 + submit_v5_8 CatBoost ensemble experiment

## Decision

Do not replace `submit_v5_8` with the current TabM ensemble. The honest rolling
validation assigns at most 5% to TabM and improves mean fold Brier by only about
0.0000064. This is too small relative to the validation approximations and
season shift to justify the added model and inference dependency.

If a blend must be tested on the server, the only supported conservative probe
is 95% CatBoost and 5% TabM, followed by one Platt calibrator fit on the blended
raw OOF. The 92.39%/7.61% in-sample OOF optimum is diagnostic only.

## Data and protocol

- Temporal folds: train on seasons earlier than 2022, 2023, or 2024 and validate
  on that season.
- Aligned validation rows: 746,504.
- Metric: fold-wise Brier score, with equal weight for the three seasons.
- CatBoost proxy: seed 42, up to 600 trees, early stopping with patience 60.
- TabM: seed 42, batch size 4096, one epoch on CPU.
- Weight grid: CatBoost weight from 0.00 to 1.00 at 0.05 intervals, plus the
  closed-form row-weighted raw optimum.
- Calibration: rolling Platt for selection; global Platt is an in-sample
  diagnostic and is not used as validation evidence.

## Base model results

| model | 2022 | 2023 | 2024 | mean | worst |
|---|---:|---:|---:|---:|---:|
| CatBoost raw proxy | 0.24315017 | 0.24991173 | 0.24774630 | 0.24693607 | 0.24991173 |
| TabM raw, epoch 1 | 0.24379437 | 0.25244986 | 0.24816434 | 0.24813619 | 0.25244986 |

The residual correlation is 0.99723, so the models have very little independent
error despite their architectural differences.

## Blend results

The full-OOF diagnostic optimum is 92.3871% CatBoost and 7.6129% TabM:

- Raw mean Brier: 0.24692794, versus CatBoost 0.24693607.
- Progressive-Platt mean Brier: 0.24689575, versus CatBoost 0.24691014.
- The gain is selected and evaluated on the same OOF, so it is optimistic.

The rolling nested procedure chooses weights using earlier folds only:

| validation season | CatBoost | TabM | Brier |
|---:|---:|---:|---:|
| 2022 | 1.00 | 0.00 | 0.24315017 |
| 2023 | 1.00 | 0.00 | 0.24990388 |
| 2024 | 0.95 | 0.05 | 0.24765703 |

Its mean Brier is 0.24690369, compared with 0.24691014 for CatBoost rolling
Platt: an improvement of about 0.0000064. The worst fold is unchanged.

## TabM convergence diagnostic

Full-size CPU training takes approximately 7.5, 11.1, and 13.8 minutes per
epoch for the three folds, excluding some preprocessing and validation time.
The worst fold, 2023, was rerun for two epochs:

- Epoch 1: validation Brier 0.252450.
- Epoch 2: validation Brier 0.254698.

Training loss improved from 0.682766 to 0.680725 while temporal validation
worsened materially. This is distribution-shift overfitting, not merely a lack
of convergence, so longer runs were stopped.

## Limitations

- Original v5_8 OOF arrays were not committed. The CatBoost OOF was retrained
  with one seed; the final v5_8 artifact averages five seeds.
- The original v5_8 validation removed 116 hand-mismatch rows. This experiment
  uses the official `train.csv` because the derived cleaned CSV is absent.
- CatBoost features are reconstructed with feature and TrackMan state stored in
  the final full-training artifact. This introduces a small approximation to the
  original fold-specific preprocessing.
- Both learners use the target fold for early stopping, matching the existing
  candidate code but not a fully nested model-selection protocol.
- The generated `ensemble_calibrator.joblib` is diagnostic and must not be put
  into a submission package without retraining the selected final TabM model.

## Next useful TabM experiment

Before trying the ensemble again, stabilize TabM itself on 2023: lower the
learning rate, reduce or regularize high-cardinality player/team embeddings,
and select architecture settings on rolling 2022-to-2023 validation. Only rerun
the blend after TabM improves the 2023 fold without degrading 2022 or 2024.
