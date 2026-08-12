# RULE_COMPLIANCE

## Checklist

- [x] Uses official competition data only.
- [x] Does not use external data.
- [x] Does not call remote APIs.
- [x] Does not download external assets during inference.
- [x] Does not use pretrained external models or weights.
- [x] Does not use other rows from `test.csv` to build test aggregates.
- [x] Does not use future outcome information.
- [x] Does not use current pitch actual location/course.
- [x] Does not use current pitch judgment/result.
- [x] Does not use current pitch `control_success`.
- [x] Does not use current pitch actual pitch type.
- [x] Does not use current-pitch Trackman measurements.
- [x] Does not use 2025 Trackman data.
- [x] Uses Trackman only as prior-season historical pitcher summaries.
- [x] Stores Trackman state in the model bundle for inference.
- [x] Keeps `submit.zip` structure compatible with the code-submission format.
- [x] Does not commit original `train.csv`, `test.csv`, or `trackman_history.csv`.

## submit.zip Structure

```text
model/
model/final_model.joblib
model/calibration_model.joblib
model/feature_schema.json
model/ensemble_config.json
script.py
requirements.txt
```

## Known Notes

`pitcher_id_mapping_clean.csv` is included as a small derived mapping file. The original large official `trackman_history.csv` is excluded from Git and must be provided separately for retraining.
