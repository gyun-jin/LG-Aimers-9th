# PREPROCESSING

## Missing Values

- Categorical columns are converted to strings and missing values are filled with `__MISSING__`.
- Numeric columns are converted with `pd.to_numeric`.
- Rate features are smoothed with train-fitted priors.
- Trackman missing history is represented by metadata flags and hand/season priors.

## Categorical Encoding

- CatBoost receives categorical columns directly.
- LightGBM receives integer-encoded categorical values from train-fitted maps.
- `tm_mapping_confidence_bucket` is treated as categorical.

## Numeric Scaling

No global numeric scaling is applied for CatBoost or LightGBM. Tree models use raw numeric magnitudes.

## ID Handling

- Raw `pitcher_id` and `batter_id` are dropped from the final v5-2 matrix.
- Team IDs are kept as categorical context.
- Pitcher identity is represented indirectly through official `asof_*` history and Trackman summary mapping.

## Rare Categories

The final v5-2 model does not use low-frequency raw pitcher/batter ID bucketing because raw pitcher/batter IDs are removed. Unknown LightGBM categories are mapped to `-1`.

## Schema Alignment

`script.py` checks that test columns match saved `input_columns`. It then selects saved feature columns in the exact order expected by each model component.

## Column-Missing Protection

The inference script raises an error if required input columns are missing or if sample/test row IDs do not match.
