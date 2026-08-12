# FEATURE_ENGINEERING

## Original Features

The pipeline starts from the official test schema, including season, inning, count, base state, score, win expectancy, leverage index, pitcher/batter IDs, team IDs, handedness, and `asof_*` cumulative history columns.

## asof Features

`asof_*` columns describe information available before the current pitch:

- pitcher cumulative success/reverse/middle/ball/strike rates
- pitcher recent 1/3/5-game success and middle rates
- batter cumulative success and middle rates
- pitcher pitch mix rates

Smoothing is applied with train-fitted priors:

```text
smoothed_rate = (count * observed_rate + alpha * prior) / (count + alpha)
```

## Count Features

- `count_state`
- balls and strikes
- recent gaps against cumulative pitcher rates

These features capture pitch-count pressure. Hitter-friendly counts usually make command harder; pitcher-friendly counts can change pitch selection and risk tolerance.

## Runner And Base Features

- `base_state`
- `runner_out_state`
- runner flags
- score and leverage context

They represent pressure and game situation before the pitch.

## Hand Match Features

- `pitcher_hand`
- `batter_hand`
- `hand_matchup`

These capture handedness interaction and matchup effects.

## Game Type Features

`game_type` is retained as a categorical input. It allows the model to separate regular/playoff or other game contexts if they have different command distributions.

## Pitcher/Batter/Team Features

The final v5-2 matrix drops raw `pitcher_id` and `batter_id`, but keeps team IDs and historical `asof_*` pitcher/batter summaries. This avoids memorizing raw pitcher/batter IDs while still using prior performance information.

## Trackman 5-2 Features

Final Trackman feature count: `29`.

### Metadata

- `tm_has_mapping`
- `tm_has_pitcher_history`
- `tm_is_low_history`
- `tm_is_rookie_or_no_history`
- `tm_hist_pitch_count`
- `tm_hist_game_count`
- `tm_hist_season_count`
- `tm_mapping_exact_game_count`
- `tm_mapping_margin`
- `tm_mapping_ratio`
- `tm_mapping_n_candidates`
- `tm_mapping_confidence`
- `tm_mapping_confidence_bucket`
- `tm_mapping_is_high_confidence`
- `tm_mapping_selected`

### Physical Core

- `tm_hist_release_speed_mean_shrunk`
- `tm_hist_release_speed_std_shrunk`
- `tm_hist_spin_rate_mean_shrunk`
- `tm_hist_spin_rate_std_shrunk`
- `tm_hist_release_pos_x_mean_shrunk`
- `tm_hist_release_pos_x_std_shrunk`
- `tm_hist_release_pos_z_mean_shrunk`
- `tm_hist_release_pos_z_std_shrunk`
- `tm_hist_extension_mean_shrunk`
- `tm_hist_extension_std_shrunk`

### Pitch Mix

- `tm_hist_pitch_group_fastball_shrunk`
- `tm_hist_pitch_group_breaking_shrunk`
- `tm_hist_pitch_group_offspeed_shrunk`
- `tm_hist_pitch_type_entropy`

## tm_mapping_confidence_bucket

`tm_mapping_confidence_bucket` bins the mapping confidence into discrete categories. It tells CatBoost whether Trackman summaries are high-confidence, medium-confidence, or mostly fallback/prior values.

## Predictive Meaning

Trackman physical summaries describe the pitcher's historical release, velocity, spin, extension, and pitch-mix profile. These do not directly reveal the current pitch outcome, but they help separate pitchers with stable mechanics/history from pitchers where the model should rely more on priors and game context.
