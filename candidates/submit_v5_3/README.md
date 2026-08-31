# Submit v5-3

submit v5-3 is the currently confirmed best leaderboard submission in this workspace.

- Server score: `1014.5598924165`
- Description: v5-2 plus cleaned train data and current-season success-rate features
- Task: predict the probability that each pitch has `control_success = 1`

## What Changed From v5-2

- Uses hand-cleaned train data
- Adds current-season pitcher success reconstruction features
- Keeps Trackman prior-season pitcher summary features
- Keeps the CatBoost + LightGBM ensemble

## Model

- CatBoostClassifier: weight `0.8`
- LightGBM LGBMClassifier: weight `0.2`
- Calibration: Platt calibration stored in `model/calibration_model.joblib`
- Trackman strategy: `mapping_all_shrink`
- Trackman feature set: `tm_metadata_physical_pitchmix`

The final probability is:

```text
p = 0.8 * catboost.predict_proba(x) + 0.2 * lightgbm.predict_proba(x)
p_final = Platt(p)
```

## Files

- `submit.zip`: exact leaderboard submission archive
- `script.py`: inference entry point
- `requirements.txt`: inference dependencies
- `model/`: trained model bundle, calibration model, feature schema, ensemble config
- `train.py`, `build_submit.py`, `full_validation.py`: reproduction-oriented training/build scripts from the v5-3 workspace
- `features/`: Trackman helper code and derived mapping files
- `data_processing/`: preprocessing summaries
- `reports/`: validation, score, compliance, and analysis reports

## Additional Data Used During Training

Committed:

- `features/pitcher_id_mapping_clean.csv`
- `features/batter_id_mapping_clean.csv`
- `features/trackman_features.py`

Not committed:

- official `train.csv`
- official `test.csv`
- official `trackman_history.csv`
- `train_hand_trackman_clean.csv`

Those files are excluded because they are official or large derivative data.

## Feature Summary

v5-3 uses original game/count/runner/team/player features, official `asof_*` cumulative features, smoothed rates, recent gap/range/std features, hand matchup, count state, pressure state, Trackman prior-season pitcher summary features, and reconstructed current-season pitcher success features.

## Preprocessing

Categorical features are passed natively to CatBoost and integer-encoded with train-time maps for LightGBM. Numeric features are not globally scaled. Missing rate features are smoothed using train priors and missing indicators are generated for key `asof_*` columns. Inference validates test schema against the saved input column list.

## Training

Expected data placement in the original workspace:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
data/trackman_history.csv
hand_cleaning_analysis/train_hand_trackman_clean.csv
train_trackman/pitcher_id_mapping_clean.csv
```

Run from the original v5-3-style workspace:

```bash
python build_submit.py
```

The copied scripts are reproduction candidates tied to the original workspace layout.

## Inference

`script.py` is designed for the competition runner. It reads:

```text
data/test.csv
data/sample_submission.csv
```

and writes:

```text
output/submission.csv
```

## submit.zip Structure

```text
model/
script.py
requirements.txt
```

## Rule Compliance

The package does not use external data, remote APIs, forbidden pretrained weights, current-pitch results, current-pitch Trackman measurements, 2025 Trackman, or test-internal aggregate features. Test rows are predicted independently.
