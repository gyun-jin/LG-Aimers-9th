# MODEL_CARD

## Model Identity

- Model name: `submit v5-2`
- Server score: `957.6800554874`
- Target: `control_success`
- Output: probability in `[0, 1]`

## Models Used

| Model | Used | Role |
|---|---|---|
| CatBoost | Yes | Main tabular classifier, weight `0.8` |
| LightGBM | Yes | Secondary tabular classifier, weight `0.2` |
| RandomForest | No | Not used |
| Transformer | No | Not used in v5-2 |
| TabM | No | Not used in v5-2 |

## Ensemble And Calibration

The final ensemble is a fixed weighted average:

```text
raw_pred = 0.8 * CatBoost + 0.2 * LightGBM
```

The raw ensemble probability is then passed through Platt calibration fitted from OOF predictions.

## Stored Files

| File | Description |
|---|---|
| `model/final_model.joblib` | CatBoost/LightGBM model bundle, feature states, Trackman state |
| `model/calibration_model.joblib` | Platt calibration model |
| `model/feature_schema.json` | Input columns, final feature columns, selected Trackman feature set |
| `model/ensemble_config.json` | Model weights and calibration selection |
| `script.py` | Evaluation-time inference entry point |
| `requirements.txt` | Runtime dependencies |

## Inference-Time Loading

`script.py` loads the bundle, rebuilds row-wise base features, appends Trackman features from saved `trackman_state`, applies each model, ensembles probabilities, calibrates, and writes `output/submission.csv`.
