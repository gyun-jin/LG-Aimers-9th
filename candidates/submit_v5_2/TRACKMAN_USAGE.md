# TRACKMAN_USAGE

## Files

- `pitcher_id_mapping_clean.csv`: maps official `pitcher_id` to `pitcher_trackman_id`
- `trackman_history.csv`: official historical Trackman data, not committed here because it is a large source file
- `trackman_features.py`: reusable feature-generation wrapper

## Mapping

The feature builder joins:

```text
train/test pitcher_id -> pitcher_id_mapping_clean.csv -> pitcher_trackman_id
```

Then it looks up historical Trackman rows by `pitcher_trackman_id`.

## Strategy

- strategy: `mapping_all_shrink`
- feature_set: `5-2_server_957`
- Trackman feature count: `29`

`mapping_all_shrink` uses every available mapping and shrinks low-history summaries toward hand/season priors. It also emits metadata features so the model knows whether the Trackman information is strong, weak, or missing.

## Temporal Rule

For each row, only Trackman rows from seasons earlier than the row's `season` are summarized.

Examples:

- 2023 row uses 2019-2022 Trackman history.
- 2024 row uses 2019-2023 Trackman history.
- 2025 test row uses 2019-2024 Trackman history.

## Not Used

- Current pitch Trackman measurement
- 2025 Trackman data
- Current pitch actual location
- Current pitch judged result
- Current pitch actual pitch type
- 1:1 joining of current train/test pitch rows to current Trackman rows

## Submission Behavior

The submission does not reread `trackman_history.csv`. The training process builds `tm_state` and stores it inside `model/final_model.joblib`. At inference, `script.py` uses that saved state to rebuild Trackman features for test rows.
