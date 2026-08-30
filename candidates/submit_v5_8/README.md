# Submit v5-8

v5-8 is a CatBoost-only control-success probability model. It uses the validated v5-8 training code, saved preprocessing/schema, final model artifact, and submission archive.

## Results

- Validation: chronological folds for 2022, 2023, and 2024
- Calibrated mean Brier: 0.24684710
- 5-3 CatBoost-only reference mean Brier: 0.24688214
- Reported server score: see `FINAL_SUMMARY.md`

## Reproduction

Place the official data files in the original project layout described by `FINAL_SUMMARY.md`, then run `train.py`. The committed source data is intentionally excluded.

For inference, run `script.py`; it writes `output/submission.csv` with columns `row_id,control_success`.

## Included

- CatBoost training and inference code
- Requirements and final feature schema
- Final model and calibration artifact
- `submit.zip`
- Validation summaries

## Excluded

Official train/test/Trackman data, virtual environments, logs, caches, private keys, and intermediate fold artifacts are excluded from the repository.
