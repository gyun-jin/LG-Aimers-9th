# MODEL CARD

## Model

- Name: submit v5-3
- Server score: `1014.5598924165`
- Status: best confirmed server submission in this workspace

## Architecture

- CatBoostClassifier: enabled
- LightGBM LGBMClassifier: enabled
- Ensemble: weighted average, CatBoost `0.8`, LightGBM `0.2`
- Calibration: Platt calibration

CatBoost handles categorical structure strongly and is the main model. LightGBM is a lower-weight auxiliary component.

## Stored Files

- `model/final_model.joblib`: model bundle with ensemble components, feature state, Trackman state, encoders, and weights
- `model/calibration_model.joblib`: Platt calibration model
- `model/feature_schema.json`: input columns, feature columns, categorical columns, selected feature set
- `model/ensemble_config.json`: ensemble configuration and selected strategy

## Why v5-3 Was Selected

v5-3 improved full 3-fold local validation versus v5-2 and achieved the best observed server score. v5-4 and v5-5 had small local Brier improvements in some checks but lower server scores, so v5-3 remains the safest shared model.

## Limitations

The 2023 fold remains difficult, with weak separation compared with 2022. Trackman features help define the selected server model but are not uniformly positive in every quick local ablation.
