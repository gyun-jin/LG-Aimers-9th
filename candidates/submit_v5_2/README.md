# Submit v5-2 Trackman Model

## Summary

- Version: `submit v5-2`
- Server leaderboard score: `957.6800554874`
- Task: predict the probability that each pitch has `control_success = 1`
- Final model: CatBoost + LightGBM weighted ensemble
- Final weights: CatBoost `0.8`, LightGBM `0.2`
- Calibration: Platt calibration
- Main improvement: prior-season Trackman pitcher summary features via `mapping_all_shrink`

## Package Layout

```text
candidates/submit_v5_2/
├── model/
├── script.py
├── requirements.txt
├── submit.zip
├── train.py
├── full_validation.py
├── build_submit.py
├── features/
├── notebooks/
├── output/
├── reports/
├── README.md
├── MODEL_CARD.md
├── FEATURE_ENGINEERING.md
├── TRACKMAN_USAGE.md
├── PREPROCESSING.md
├── TRAINING_AND_INFERENCE.md
├── RULE_COMPLIANCE.md
└── SCORE_REPORT.md
```

## Model

The model keeps the v5 no-id tabular feature pipeline and appends 29 Trackman features. The final prediction is:

```text
raw_pred = 0.8 * CatBoost_proba + 0.2 * LightGBM_proba
final_pred = Platt(raw_pred)
```

The final feature matrix has 118 columns.

## Data

Included:

- `features/pitcher_id_mapping_clean.csv`
- `features/trackman_features.py`
- `features/README.md`
- `features/trackman_reuse_guide.md`

Not included:

- `train.csv`
- `test.csv`
- `trackman_history.csv`

`trackman_history.csv` is an official large source file and must be placed under `data/` or `train_trackman/` when retraining. The submission does not reread the original Trackman CSV; it loads the Trackman state saved inside `model/final_model.joblib`.

## Trackman Usage

Trackman is used as a prior-season pitcher summary. For each pitch row, the code maps `pitcher_id` to `pitcher_trackman_id`, then uses Trackman records from seasons earlier than the row season. Current-pitch Trackman measurements and 2025 Trackman data are not used.

Final Trackman settings:

- strategy: `mapping_all_shrink`
- feature_set: `5-2_server_957`
- Trackman feature count: `29`
- categorical Trackman feature: `tm_mapping_confidence_bucket`

## Preprocessing

- Categorical features are string-filled with `__MISSING__`.
- CatBoost receives categorical columns natively.
- LightGBM uses train-fitted integer maps for categorical columns.
- Numeric values are converted with `pd.to_numeric`; missing values are handled through model-native behavior or explicit priors.
- Train/test schema is checked against saved `input_columns`.
- `pitcher_id` and `batter_id` raw IDs are dropped in the final v5-2 matrix.

## Training

Training candidates:

```bash
python train.py
python full_validation.py
python build_submit.py
```

Required local files for retraining:

```text
data/train.csv
data/trackman_history.csv
features/pitcher_id_mapping_clean.csv
```

The copied `features/trackman_features.py` is a reusable wrapper. The actual v5-2 training code also imports the Trackman generation logic from the original 5-1 code path.

## Inference

The evaluation server runs:

```bash
python script.py
```

The script reads `./open/test.csv` and `./open/sample_submission.csv` first, then falls back to `./data/`. It writes:

```text
output/submission.csv
```

## submit.zip

`submit.zip` contains exactly:

```text
model/
model/final_model.joblib
model/calibration_model.joblib
model/feature_schema.json
model/ensemble_config.json
script.py
requirements.txt
```

## Rule Compliance

- Uses official data only.
- Does not use remote APIs.
- Does not use pretrained external weights.
- Does not use test-row aggregates.
- Does not use future outcome, current pitch result, current pitch location, or current-pitch Trackman measurements.
- Does not include large original source CSVs in this Git package.
