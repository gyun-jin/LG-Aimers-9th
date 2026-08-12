# 5-2 Trackman 제한 피처 실험

## 결론

- best candidate: `tm_metadata_only`
- selected calibration: `platt`
- best mean Brier: `0.24710060`
- best worst Brier: `0.24989225`
- stable selected candidate: `tm_metadata_physical_pitchmix`
- stable selected calibration: `platt`
- stable selected mean/worst Brier: `0.24710121` / `0.24974433`
- `/5` reference mean/worst: `0.24717070` / `0.24979486`
- stable selected `/5` 대비 mean 차이: `-0.00006949`
- stable selected `/5` 대비 worst 차이: `-0.00005053`

## 설계

- `/5`와 같은 no_ids CatBoost 0.8 + LightGBM 0.2 구조를 유지했다.
- `/5`와 같은 quick validation sample 크기인 train 180,000 / valid 70,000을 사용했다.
- Trackman은 `mapping_all_shrink` 방식으로 prior-season pitcher summary만 만들었다.
- 현재 투구 단위 Trackman 결합, 2025 Trackman, test 내부 통계는 사용하지 않았다.

## 후보별 결과

| feature set | selected calibration | mean Brier | worst Brier | raw mean Brier | raw worst Brier | mean AUC | mean_pred | mean_target |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| tm_metadata_only | platt | 0.24710060 | 0.24989225 | 0.24711726 | 0.24988907 | 0.550558 | 0.509544 | 0.504643 |
| tm_metadata_pitchmix | prior_correction | 0.24710102 | 0.24981299 | 0.24711732 | 0.24982309 | 0.551287 | 0.509487 | 0.504643 |
| tm_metadata_physical_pitchmix | platt | 0.24710121 | 0.24974433 | 0.24711073 | 0.24975944 | 0.550460 | 0.508425 | 0.504643 |
| tm_metadata_physical_core | platt | 0.24710923 | 0.24975636 | 0.24712163 | 0.24976863 | 0.550331 | 0.508978 | 0.504643 |
| v5_no_trackman_recheck | prior_correction | 0.24715062 | 0.24985105 | 0.24716838 | 0.24985813 | 0.550368 | 0.509809 | 0.504643 |

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
- quick 기준 eligible 후보: `['tm_metadata_physical_pitchmix', 'tm_metadata_physical_core']`
