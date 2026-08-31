# 5-3 Trackman 제한 피처 quick 실험

## 결론

- best candidate: `v5_no_trackman_recheck`
- selected calibration: `platt`
- best mean Brier: `0.24696972`
- best worst Brier: `0.24973173`
- stable selected candidate: `v5_no_trackman_recheck`
- stable selected calibration: `platt`
- stable selected mean/worst Brier: `0.24696972` / `0.24973173`
- `/5` reference mean/worst: `0.24717070` / `0.24979486`
- stable selected `/5` 대비 mean 차이: `-0.00020098`
- stable selected `/5` 대비 worst 차이: `-0.00006313`

## 설계

- `/5`와 같은 no_ids CatBoost 0.8 + LightGBM 0.2 구조를 유지했다.
- `/5`와 같은 quick validation sample 크기인 train 180,000 / valid 70,000을 사용했다.
- Trackman은 `mapping_all_shrink` 방식으로 prior-season pitcher summary만 만들었다.
- 현재 투구 단위 Trackman 결합, 2025 Trackman, test 내부 통계는 사용하지 않았다.

## 후보별 결과

| feature set | selected calibration | mean Brier | worst Brier | raw mean Brier | raw worst Brier | mean AUC | mean_pred | mean_target |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v5_no_trackman_recheck | platt | 0.24696972 | 0.24973173 | 0.24698118 | 0.24972270 | 0.553427 | 0.510095 | 0.506105 |
| tm_metadata_physical_pitchmix | prior_correction | 0.24698039 | 0.24985762 | 0.24698775 | 0.24985008 | 0.554009 | 0.510605 | 0.506105 |
| tm_metadata_physical_core | platt | 0.24698188 | 0.24980606 | 0.24699124 | 0.24978660 | 0.553642 | 0.510186 | 0.506105 |
| tm_metadata_pitchmix | platt | 0.24698750 | 0.24981527 | 0.24699513 | 0.24978561 | 0.554013 | 0.510687 | 0.506105 |
| tm_metadata_only | prior_correction | 0.24700277 | 0.24982940 | 0.24701735 | 0.24982910 | 0.553521 | 0.511168 | 0.506105 |

## Feature set 구성

### v5_no_trackman_recheck
- Trackman 피처 없음. `/5` no_ids baseline 재확인용.

### tm_metadata_only
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

### tm_metadata_pitchmix
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
- `tm_hist_pitch_group_fastball_shrunk`
- `tm_hist_pitch_group_breaking_shrunk`
- `tm_hist_pitch_group_offspeed_shrunk`
- `tm_hist_pitch_type_entropy`

### tm_metadata_physical_core
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

### tm_metadata_physical_pitchmix
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
- `tm_hist_pitch_group_fastball_shrunk`
- `tm_hist_pitch_group_breaking_shrunk`
- `tm_hist_pitch_group_offspeed_shrunk`
- `tm_hist_pitch_type_entropy`

## 판단

- 판단: `전체 fold 재검증 필요`
- `/5` 기준을 평균과 최악 fold에서 동시에 넘어야 `submit.zip`을 만들 후보로 본다.
- quick 기준 eligible 후보: `['v5_no_trackman_recheck']`
