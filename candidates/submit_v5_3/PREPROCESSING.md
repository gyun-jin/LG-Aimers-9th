# PREPROCESSING

## Hand Cleaning

v5-3 trains on `train_hand_trackman_clean.csv`, which removes conservative hand mismatch rows backed by Trackman hand majority checks. The large cleaned CSV is not committed.

## Current-Season Reconstruction

Pitcher current-season features are reconstructed from row-level official `asof_*` values plus train-time prior season-ending cumulative constants.

## Missing Values

Rate columns are smoothed with train priors and count denominators. Missing flags are generated for important `asof_*` columns.

## Categorical Encoding

CatBoost receives categorical columns directly. LightGBM receives train-time integer maps stored in the model artifact. Unknown categories map to fallback values.

## Numeric Scaling

No global numeric scaling is used. Log transforms and smoothing are applied where useful, for example `log1p_asof_pitcher_n`.

## ID Handling

The selected v5-3 ensemble follows the established v5 no-ID behavior where ID columns are dropped from component frames when required by the variant. Team and context categorical fields remain.

## Schema Safety

`script.py` checks test input columns against the saved schema. Missing or reordered input schema raises an error before prediction.

## Data Quality Handling

v5-3 reflects the earlier hand-cleaning result and current-season reconstruction. Later broader data-quality flags were analyzed after v5-3 and are not part of this highest-scoring submission.
