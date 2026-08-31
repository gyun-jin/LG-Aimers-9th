# TRACKMAN USAGE

## Mapping Files

- `pitcher_id_mapping_clean.csv`: maps competition `pitcher_id` to `pitcher_trackman_id`
- `batter_id_mapping_clean.csv`: derived batter mapping from hand-cleaning analysis; included for documentation and hand cleaning context

v5-3 Trackman model features are pitcher-oriented. Batter mapping is not used as a direct inference-time Trackman feature source.

## Strategy

v5-3 uses `mapping_all_shrink`. Trackman logs are summarized by pitcher and season history, then shrunk toward priors. The selected feature set is `tm_metadata_physical_pitchmix`.

The model uses prior-season or historical summaries only. It does not join a train/test row to the current pitch's Trackman row and does not use current-pitch Trackman measurements.

## Safety Rules

- No 2025 Trackman data is used.
- `trackman_history.csv` is not included in Git.
- `script.py` does not read raw `trackman_history.csv`; train-time Trackman state is stored inside `model/final_model.joblib`.
- The model does not use test-set aggregate Trackman statistics.

## Feature Count

The final v5-3 schema reports `138` total features. The Trackman feature set includes metadata, physical core summaries, pitch mix summaries, and pitch type entropy.
