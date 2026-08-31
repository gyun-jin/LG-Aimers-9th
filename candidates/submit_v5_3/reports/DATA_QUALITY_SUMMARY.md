# DATA QUALITY SUMMARY

## 1. 전체 요약

- 원본 파일은 읽기 전용으로 분석했고, 산출물은 `data_quality_check` 아래에만 저장했다.

- train은 1,475,092행, test 샘플은 5행, Trackman은 1,793,078행이다.

- 핵심 이상 후보: pitcher hand 불일치 ID 15개, batter hand 불일치/스위치 후보 ID 32개, Trackman hand mismatch 후보 724건, count/state 이상 row 0건, current-season 복원 이상 row 1,533건.

- `test.csv`는 배포본 형식 확인용 5행이므로 test 분포 기반 판단은 모델 보정에 쓰면 안 된다.


## 2. 사용 파일과 행/컬럼 수

| file | rows | cols |
| --- | --- | --- |
| train | 1475092 | 49 |
| test | 5 | 48 |
| sample_submission | 5 | 2 |
| trackman_history | 1793078 | 30 |


- `control_success` in train: True

- `control_success` in test: False

- train row_id 중복: 0; test row_id 중복: 0; sample row_id 중복: 0


상위 결측률 컬럼은 `output/summary_counts.csv`에 저장했다. dtype은 train/test schema 섹션과 `output/train_test_schema_quality.csv`에 정리했다.


## 3. hand 관련 이상 탐지

- 양손처럼 보이는 pitcher_id: 15개. 목록 상위: 22233, 22254, 22277, 22345, 22525, 23101, 23238, 23288, 23418, 23478, 23599, 23627, 23819, 23887, 24570

- 스위치/불확실 batter_id: 32개. 목록 상위: 21835, 22064, 22066, 22208, 22514, 22521, 22606, 22737, 22773, 22808, 22830, 22853, 22924, 23274, 23298, 23413, 23481, 23548, 23746, 23805, 23810, 23811, 23817, 23980, 24005, 24023, 24075, 24263, 24304, 24464, ... (+2)

- 명백한 오류로 보이는 pitcher 소수 hand 후보 ID: 11개. 기준은 다수 hand가 압도적이고 소수 row 비율이 2% 이하/100행 이하인 경우다.

- batter는 실제 switch hitter 가능성이 있어 `delete`보다 `flag`가 기본적으로 안전하다.

- 기존 `hand_cleaning_analysis` 참고 파일: `{"pitcher_hand_cleaning_summary.csv": {"rows": 760, "cols": 12}, "batter_hand_cleaning_summary.csv": {"rows": 735, "cols": 16}, "removed_hand_mismatch_rows_all.csv": {"rows": 116, "cols": 51}}`


상위 pitcher hand 불일치:

| pitcher_id | rows | left_rows | right_rows | unique_hand_count | hands | majority_hand | majority_rows | minority_rows | minority_rate | season_hand_changes | team_hand_changes | seasons | teams | severity | candidate_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 22277 | 12743 | 13 | 12730 | 2 | Left, Right | Right | 12730 | 13 | 0.0010201679353370478 | 1 | 1 | 2019, 2020, 2021, 2022, 2023, 2024 | 13 | clear_error | delete_candidate |
| 23288 | 4742 | 4735 | 7 | 2 | Left, Right | Left | 4735 | 7 | 0.0014761703922395613 | 1 | 1 | 2019, 2020, 2021, 2022, 2023, 2024 | 14, 23 | clear_error | delete_candidate |
| 23238 | 4377 | 2 | 4375 | 2 | Left, Right | Right | 4375 | 2 | 0.0004569339730408956 | 2 | 1 | 2019, 2020, 2021, 2022, 2023, 2024 | 12 | clear_error | delete_candidate |
| 23478 | 5026 | 1 | 5025 | 2 | Left, Right | Right | 5025 | 1 | 0.00019896538002387584 | 1 | 1 | 2019, 2020, 2021, 2022, 2023, 2024 | 20, 23 | clear_error | delete_candidate |
| 22254 | 4953 | 1 | 4952 | 2 | Left, Right | Right | 4952 | 1 | 0.0002018978396931153 | 1 | 1 | 2019, 2020, 2021, 2022, 2023, 2024 | 18 | clear_error | delete_candidate |
| 22525 | 3989 | 3988 | 1 | 2 | Left, Right | Left | 3988 | 1 | 0.000250689395838556 | 1 | 1 | 2019, 2020, 2021, 2022, 2023, 2024 | 12, 13 | clear_error | delete_candidate |
| 23418 | 2947 | 1 | 2946 | 2 | Left, Right | Right | 2946 | 1 | 0.00033932813030200206 | 1 | 1 | 2021, 2022, 2023, 2024 | 13 | clear_error | delete_candidate |
| 24570 | 2832 | 2831 | 1 | 2 | Left, Right | Left | 2831 | 1 | 0.00035310734463276836 | 1 | 1 | 2024 | 14 | clear_error | delete_candidate |
| 23627 | 2597 | 2596 | 1 | 2 | Left, Right | Left | 2596 | 1 | 0.0003850596842510589 | 1 | 1 | 2019, 2020, 2021, 2022, 2023, 2024 | 15, 17, 21 | clear_error | delete_candidate |
| 22233 | 322 | 321 | 1 | 2 | Left, Right | Left | 321 | 1 | 0.003105590062111801 | 1 | 1 | 2019, 2020 | 17 | clear_error | delete_candidate |
| 23599 | 200 | 199 | 1 | 2 | Left, Right | Left | 199 | 1 | 0.005 | 1 | 1 | 2020, 2021 | 19 | clear_error | delete_candidate |
| 23101 | 1412 | 47 | 1365 | 2 | Left, Right | Right | 1365 | 47 | 0.03328611898016997 | 1 | 1 | 2019, 2020, 2021, 2022, 2023 | 21 | suspicious | flag_candidate |
| 23887 | 121 | 37 | 84 | 2 | Left, Right | Right | 84 | 37 | 0.30578512396694213 | 1 | 1 | 2020, 2021 | 15 | suspicious | flag_candidate |
| 23819 | 292 | 36 | 256 | 2 | Left, Right | Right | 256 | 36 | 0.1232876712328767 | 1 | 1 | 2024 | 25 | suspicious | flag_candidate |
| 22345 | 23 | 7 | 16 | 2 | Left, Right | Right | 16 | 7 | 0.30434782608695654 | 1 | 1 | 2021 | 17 | suspicious | flag_candidate |


상위 batter hand 불일치/스위치 후보:

| batter_id | rows | left_rows | right_rows | unique_hand_count | hands | majority_hand | majority_rows | minority_rows | minority_rate | season_hand_changes | team_hand_changes | seasons | teams | severity | candidate_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 23274 | 1863 | 10 | 1853 | 2 | Left, Right | Right | 1853 | 10 | 0.005367686527106817 | 1 | 1 | 2021, 2022, 2023, 2024 | 19 | clear_error | delete_candidate |
| 22737 | 676 | 668 | 8 | 2 | Left, Right | Left | 668 | 8 | 0.011834319526627219 | 1 | 1 | 2019 | 13 | clear_error | delete_candidate |
| 23298 | 2016 | 2011 | 5 | 2 | Left, Right | Left | 2011 | 5 | 0.00248015873015873 | 1 | 1 | 2019, 2020, 2021, 2022, 2023, 2024 | 15, 16, 23 | clear_error | delete_candidate |
| 24005 | 1238 | 5 | 1233 | 2 | Left, Right | Right | 1233 | 5 | 0.004038772213247173 | 1 | 1 | 2020 | 17 | clear_error | delete_candidate |
| 22808 | 4472 | 4468 | 4 | 2 | Left, Right | Left | 4468 | 4 | 0.0008944543828264759 | 1 | 1 | 2019, 2020, 2021, 2022, 2023, 2024 | 13, 19 | clear_error | delete_candidate |
| 22773 | 679 | 3 | 676 | 2 | Left, Right | Right | 676 | 3 | 0.004418262150220913 | 1 | 1 | 2019, 2020 | 13 | clear_error | delete_candidate |
| 22066 | 12781 | 12780 | 1 | 2 | Left, Right | Left | 12780 | 1 | 7.824113919098662e-05 | 1 | 1 | 2019, 2020, 2021, 2022, 2023, 2024 | 15, 19 | clear_error | delete_candidate |
| 22606 | 10659 | 10658 | 1 | 2 | Left, Right | Left | 10658 | 1 | 9.381743127873159e-05 | 1 | 1 | 2019, 2020, 2021, 2022, 2023, 2024 | 19 | clear_error | delete_candidate |
| 22853 | 3573 | 3572 | 1 | 2 | Left, Right | Left | 3572 | 1 | 0.000279876854184159 | 1 | 1 | 2019, 2020, 2021, 2022, 2023, 2024 | 14, 23 | clear_error | delete_candidate |
| 22208 | 991 | 1 | 990 | 2 | Left, Right | Right | 990 | 1 | 0.0010090817356205853 | 1 | 1 | 2019, 2020, 2021, 2022 | 17, 20 | clear_error | delete_candidate |
| 23746 | 349 | 1 | 348 | 2 | Left, Right | Right | 348 | 1 | 0.0028653295128939827 | 1 | 1 | 2019, 2020 | 14 | clear_error | delete_candidate |
| 23548 | 95 | 1 | 94 | 2 | Left, Right | Right | 94 | 1 | 0.010526315789473684 | 1 | 1 | 2019 | 21 | clear_error | delete_candidate |
| 23413 | 7433 | 5093 | 2340 | 2 | Left, Right | Left | 5093 | 2340 | 0.314812323422575 | 3 | 1 | 2019, 2020, 2024 | 20 | suspicious | flag_candidate |
| 24075 | 5953 | 4307 | 1646 | 2 | Left, Right | Left | 4307 | 1646 | 0.2764992440786158 | 4 | 1 | 2021, 2022, 2023, 2024 | 19 | suspicious | flag_candidate |
| 24526 | 2442 | 1643 | 799 | 2 | Left, Right | Left | 1643 | 799 | 0.3271908271908272 | 1 | 1 | 2024 | 15 | suspicious | flag_candidate |


## 4. Trackman hand mismatch 분석

- 후보 724건. mapping confidence가 낮거나 ambiguous한 경우는 명백 오류로 단정하지 않았다.

- high confidence 후보: 31, medium: 356, low: 337


| entity | train_id | trackman_id | mapping_file | mapping_confidence | exact_game_count | ratio | margin | train_left_rows | train_right_rows | trackman_left_rows | trackman_right_rows | train_majority_hand | trackman_majority_hand | mismatch_rows_vs_trackman_majority | mismatch_rate_vs_trackman_majority | severity | recommended_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batter | 24162.0 | 52124.0 | hand_cleaning_analysis/batter_id_mapping_clean.csv | low | 1.0 | 1.0 | 1.0 | 1256 | 0 | 6 | 994 | Left | Right | 1256 | 1.0 | flag_only | flag_candidate |
| batter | 23980.0 | 578628.0 | hand_cleaning_analysis/batter_id_mapping_clean.csv | medium | 8.0 | 8.0 | 8.0 | 199 | 147 | 364 | 214 | Left | Left | 147 | 0.42485549132947975 | flag_only | flag_candidate |
| pitcher | 23446.0 | 68391.0 | train_trackman/pitcher_id_mapping_clean.csv | low | 2.0 | 2.0 | 1.0 | 0 | 123 | 265 | 0 | Right | Left | 123 | 1.0 | flag_only | flag_candidate |
| pitcher | 23887.0 | 50563.0 | train_trackman/pitcher_id_mapping_clean.csv | medium | 8.0 | 8.0 | 7.0 | 37 | 84 | 558 | 0 | Right | Left | 84 | 0.6942148760330579 | flag_only | flag_candidate |
| batter | 23481.0 | 68069.0 | hand_cleaning_analysis/batter_id_mapping_clean.csv | low | 2.0 | 2.0 | 2.0 | 41 | 111 | 16 | 1804 | Right | Right | 41 | 0.26973684210526316 | flag_only | flag_candidate |
| batter | 22064.0 | 78566.0 | hand_cleaning_analysis/batter_id_mapping_clean.csv | low | 3.0 | 3.0 | 3.0 | 59 | 36 | 453 | 156 | Left | Left | 36 | 0.37894736842105264 | flag_only | flag_candidate |
| pitcher | 23444.0 | 68362.0 | train_trackman/pitcher_id_mapping_clean.csv | low | 1.0 | inf | 1.0 | 0 | 22 | 22 | 0 | Right | Left | 22 | 1.0 | flag_only | flag_candidate |
| batter | 21835.0 | 76869.0 | hand_cleaning_analysis/batter_id_mapping_clean.csv | medium | 16.0 | 16.0 | 16.0 | 293 | 19 | 327 | 43 | Left | Left | 19 | 0.060897435897435896 | flag_only | flag_candidate |
| pitcher | 22345.0 | 61208.0 | train_trackman/pitcher_id_mapping_clean.csv | low | 3.0 | 3.0 | 2.0 | 7 | 16 | 12 | 11 | Right | Left | 16 | 0.6956521739130435 | flag_only | flag_candidate |
| pitcher | 22277.0 | 61101.0 | train_trackman/pitcher_id_mapping_clean.csv | high | 148.0 | 148.0 | 147.0 | 13 | 12730 | 0 | 12499 | Right | Right | 13 | 0.0010201679353370478 | flag_only | flag_candidate |
| batter | 23274.0 | 67905.0 | hand_cleaning_analysis/batter_id_mapping_clean.csv | high | 107.0 | 107.0 | 107.0 | 10 | 1853 | 4 | 3677 | Right | Right | 10 | 0.005367686527106817 | flag_only | flag_candidate |
| batter | 22737.0 | 64117.0 | hand_cleaning_analysis/batter_id_mapping_clean.csv | high | 22.0 | 22.0 | 22.0 | 668 | 8 | 408 | 0 | Left | Left | 8 | 0.011834319526627219 | flag_only | flag_candidate |
| pitcher | 23288.0 | 67391.0 | train_trackman/pitcher_id_mapping_clean.csv | high | 228.0 | 28.5 | 220.0 | 4735 | 7 | 4943 | 0 | Left | Left | 7 | 0.0014761703922395613 | flag_only | flag_candidate |
| batter | 23298.0 | 67644.0 | hand_cleaning_analysis/batter_id_mapping_clean.csv | high | 100.0 | 100.0 | 100.0 | 2011 | 5 | 3267 | 0 | Left | Left | 5 | 0.00248015873015873 | flag_only | flag_candidate |
| batter | 24005.0 | 488681.0 | hand_cleaning_analysis/batter_id_mapping_clean.csv | high | 44.0 | 44.0 | 44.0 | 5 | 1233 | 5 | 1155 | Right | Right | 5 | 0.004038772213247173 | flag_only | flag_candidate |
| batter | 22808.0 | 64101.0 | hand_cleaning_analysis/batter_id_mapping_clean.csv | high | 124.0 | 124.0 | 124.0 | 4468 | 4 | 4452 | 7 | Left | Left | 4 | 0.0008944543828264759 | flag_only | flag_candidate |
| batter | 22773.0 | 65120.0 | hand_cleaning_analysis/batter_id_mapping_clean.csv | medium | 15.0 | 15.0 | 15.0 | 3 | 676 | 1 | 546 | Right | Right | 3 | 0.004418262150220913 | flag_only | flag_candidate |
| pitcher | 23238.0 | 67266.0 | train_trackman/pitcher_id_mapping_clean.csv | high | 243.0 | 48.6 | 238.0 | 2 | 4375 | 0 | 4487 | Right | Right | 2 | 0.0004569339730408956 | flag_only | flag_candidate |
| pitcher | 22254.0 | 60146.0 | train_trackman/pitcher_id_mapping_clean.csv | high | 273.0 | 39.0 | 266.0 | 1 | 4952 | 0 | 5059 | Right | Right | 1 | 0.0002018978396931153 | flag_only | flag_candidate |
| pitcher | 22525.0 | 63248.0 | train_trackman/pitcher_id_mapping_clean.csv | high | 192.0 | 32.0 | 186.0 | 3988 | 1 | 4006 | 0 | Left | Left | 1 | 0.000250689395838556 | flag_only | flag_candidate |


## 5. 경기/선수/team 이상 탐지

- train에는 명시적 game_id/game_date가 없어 row 순서와 1회초 0-0 첫 카운트 조건으로 `inferred_game_id`를 복원했다. 따라서 이 섹션은 삭제 판단이 아니라 extreme flag 용도다.

- inferred game 수: 7,079; roster/team extreme 후보: 33건.


| season | inferred_game_id | rows | unique_pitcher_teams | unique_batter_teams | unique_all_teams | unique_pitchers | unique_batters | max_inning | issue_type | severity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2019 | 122 | 453 | 2 | 2 | 2 | 16 | 25 | 12 | extreme_row_count | flag_only |
| 2019 | 190 | 431 | 2 | 2 | 2 | 18 | 24 | 11 | extreme_row_count | flag_only |
| 2019 | 211 | 421 | 2 | 2 | 2 | 15 | 26 | 11 | extreme_row_count | flag_only |
| 2020 | 1566 | 418 | 2 | 2 | 2 | 16 | 25 | 12 | extreme_row_count | flag_only |
| 2020 | 1698 | 431 | 2 | 2 | 2 | 12 | 27 | 9 | extreme_row_count | flag_only |
| 2020 | 1740 | 450 | 2 | 2 | 2 | 20 | 26 | 12 | extreme_row_count | flag_only |
| 2020 | 1885 | 457 | 2 | 2 | 2 | 19 | 27 | 12 | extreme_row_count | flag_only |
| 2020 | 1956 | 460 | 2 | 2 | 2 | 18 | 27 | 12 | extreme_row_count | flag_only |
| 2020 | 2135 | 453 | 2 | 2 | 2 | 16 | 32 | 12 | extreme_row_count | flag_only |
| 2021 | 2374 | 447 | 2 | 2 | 2 | 14 | 23 | 12 | extreme_row_count | flag_only |
| 2021 | 2524 | 440 | 2 | 2 | 2 | 13 | 27 | 12 | extreme_row_count | flag_only |
| 2021 | 2656 | 448 | 2 | 2 | 2 | 17 | 30 | 12 | extreme_row_count | flag_only |
| 2021 | 2831 | 451 | 2 | 2 | 2 | 18 | 26 | 12 | extreme_row_count | flag_only |
| 2022 | 3584 | 426 | 2 | 2 | 2 | 18 | 25 | 12 | extreme_row_count | flag_only |
| 2022 | 3789 | 430 | 2 | 2 | 2 | 15 | 24 | 12 | extreme_row_count | flag_only |
| 2022 | 4603 | 422 | 2 | 2 | 2 | 13 | 25 | 9 | extreme_row_count | flag_only |
| 2023 | 4741 | 419 | 2 | 2 | 2 | 18 | 23 | 11 | extreme_row_count | flag_only |
| 2023 | 4825 | 437 | 2 | 2 | 2 | 17 | 25 | 12 | extreme_row_count | flag_only |
| 2023 | 5024 | 439 | 2 | 2 | 2 | 14 | 25 | 12 | extreme_row_count | flag_only |
| 2023 | 5169 | 441 | 2 | 2 | 2 | 15 | 25 | 12 | extreme_row_count | flag_only |


## 6. count/inning/score 이상 탐지

- count/state anomaly rows: 0; clear_error rows: 0.

- `balls/strikes/outs` 범위 위반, runner/base_state 불일치, run_total 불일치는 제거 후보다. win expectancy 범위 위반은 clip 후보, 큰 score_diff는 flag 후보로 분리한다.


(없음)


## 7. asof 누적 피처 이상 탐지

- rate류는 0~1 범위, n/count류는 음수 여부, n=0인데 rate가 non-null인 경우, 결측률과 train/test 샘플 평균 차이를 점검했다.


| column | dtype | train_missing | train_missing_rate | test_missing | test_missing_rate | train_min | train_max | train_mean | test_mean | rate_outside_0_1_rows | negative_rows | n_zero_but_rate_nonnull_rows | severity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| asof_pitcher_prev1_game_success_rate | float64 | 29185 | 0.019785206617621138 | 3 | 0.6 | 0.0 | 1.0 | 0.5233543069768666 | 0.39586849999999996 | 0 | 0 | 0 | no_action |
| asof_pitcher_prev3_game_success_rate | float64 | 29185 | 0.019785206617621138 | 3 | 0.6 | 0.0 | 1.0 | 0.523847505939317 | 0.4115025 | 0 | 0 | 0 | no_action |
| asof_pitcher_prev5_game_success_rate | float64 | 29185 | 0.019785206617621138 | 3 | 0.6 | 0.0 | 1.0 | 0.5244111593601802 | 0.407593 | 0 | 0 | 0 | no_action |
| asof_pitcher_prev1_game_middle_rate | float64 | 29185 | 0.019785206617621138 | 3 | 0.6 | 0.0 | 1.0 | 0.15060582464190297 | 0.1918505 | 0 | 0 | 0 | no_action |
| asof_pitcher_prev3_game_middle_rate | float64 | 29185 | 0.019785206617621138 | 3 | 0.6 | 0.0 | 1.0 | 0.14939043487541034 | 0.18394749999999999 | 0 | 0 | 0 | no_action |
| asof_pitcher_prev5_game_middle_rate | float64 | 29185 | 0.019785206617621138 | 3 | 0.6 | 0.0 | 1.0 | 0.14889194615302365 | 0.18582100000000001 | 0 | 0 | 0 | no_action |
| asof_batter_success_rate | float64 | 830 | 0.0005626767686354478 | 2 | 0.4 | 0.0 | 1.0 | 0.5397311017763466 | 0.5156763333333333 | 0 | 0 | 0 | no_action |
| asof_batter_middle_rate | float64 | 830 | 0.0005626767686354478 | 2 | 0.4 | 0.0 | 1.0 | 0.14050634325036526 | 0.153331 | 0 | 0 | 0 | no_action |
| asof_pitcher_success_rate | float64 | 792 | 0.0005369156635653912 | 1 | 0.2 | 0.0 | 1.0 | 0.5352282202778269 | 0.42497925000000003 | 0 | 0 | 0 | no_action |
| asof_pitcher_reverse_rate | float64 | 792 | 0.0005369156635653912 | 1 | 0.2 | 0.0 | 1.0 | 0.2152099238230009 | 0.282621 | 0 | 0 | 0 | no_action |
| asof_pitcher_middle_rate | float64 | 792 | 0.0005369156635653912 | 1 | 0.2 | 0.0 | 1.0 | 0.1418648379080716 | 0.22207325 | 0 | 0 | 0 | no_action |
| asof_pitcher_ball_rate | float64 | 792 | 0.0005369156635653912 | 1 | 0.2 | 0.0 | 1.0 | 0.37224288439137226 | 0.39866375 | 0 | 0 | 0 | no_action |
| asof_pitcher_strike_rate | float64 | 792 | 0.0005369156635653912 | 1 | 0.2 | 0.0 | 1.0 | 0.44187568922315673 | 0.407876 | 0 | 0 | 0 | no_action |
| asof_pitcher_fastball_rate | float64 | 792 | 0.0005369156635653912 | 1 | 0.2 | 0.0 | 1.0 | 0.5543834851488842 | 0.5281629999999999 | 0 | 0 | 194 | no_action |
| asof_pitcher_breaking_rate | float64 | 792 | 0.0005369156635653912 | 1 | 0.2 | 0.0 | 1.0 | 0.28927426308661713 | 0.32915675 | 0 | 0 | 4442 | no_action |
| asof_pitcher_offspeed_rate | float64 | 792 | 0.0005369156635653912 | 1 | 0.2 | 0.0 | 1.0 | 0.1563422500024134 | 0.142680575 | 0 | 0 | 40525 | no_action |
| asof_pitcher_n | int64 | 0 | 0.0 | 0 | 0.0 | 0.0 | 15449.0 | 2661.3230401900355 | 809.2 | 0 | 0 | 0 | no_action |
| asof_batter_n | int64 | 0 | 0.0 | 0 | 0.0 | 0.0 | 13927.0 | 3270.8473966369556 | 5159.4 | 0 | 0 | 0 | no_action |
| asof_pitcher_pitchmix_n | int64 | 0 | 0.0 | 0 | 0.0 | 0.0 | 15449.0 | 2661.3230401900355 | 809.2 | 0 | 0 | 0 | no_action |


## 8. current-season 복원 가능성 및 이상 탐지

- 복원 이상 row: 1,533건.

- `asof_pitcher_n * asof_pitcher_success_rate`는 통산 성공 수 근사로 사용할 수 있으나, 시즌 종료 누적 차감은 row 정렬/라운딩/공식 asof 정의에 민감하다.

- 이상 row는 삭제보다 flag 및 clipping 대상이다. small sample은 `current_season_n < 30` 같은 별도 cold-start flag를 두는 편이 안전하다.


| row_id | season | pitcher_id | issue_type | asof_pitcher_n | asof_pitcher_success_rate | current_season_n_est | current_season_success_count_est | severity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TRAIN_0237924 | 2020 | 21961 | current_season_success_count_negative | 2699 | 0.609114 | 1.0 | -0.0006339999997635459 | suspicious |
| TRAIN_0728589 | 2022 | 21961 | current_season_success_count_negative | 5649 | 0.590901 | 1.0 | -0.002139000000170199 | suspicious |
| TRAIN_0728590 | 2022 | 21961 | current_season_success_count_negative | 5650 | 0.590796 | 2.0 | -0.00448800000049232 | suspicious |
| TRAIN_0988549 | 2023 | 21961 | current_season_success_count_negative | 8505 | 0.566843 | 1.0 | -0.0029250000006868504 | suspicious |
| TRAIN_0988550 | 2023 | 21961 | current_season_success_count_negative | 8506 | 0.566776 | 2.0 | -0.005984000001262757 | suspicious |
| TRAIN_0988551 | 2023 | 21961 | current_season_success_count_negative | 8507 | 0.56671 | 3.0 | -0.0006700000003547757 | suspicious |
| TRAIN_0988552 | 2023 | 21961 | current_season_success_count_negative | 8508 | 0.566643 | 4.0 | -0.003996000000370259 | suspicious |
| TRAIN_1225265 | 2024 | 21961 | current_season_success_count_gt_n | 11256 | 0.552683 | 1.0 | 1.0028830000001108 | suspicious |
| TRAIN_0241792 | 2020 | 23422 | current_season_success_count_negative | 2746 | 0.482156 | 1.0 | -0.0009640000000672444 | suspicious |
| TRAIN_0496974 | 2021 | 23496 | current_season_success_count_gt_n | 1839 | 0.550299 | 3.0 | 3.0003569999998945 | suspicious |
| TRAIN_0496976 | 2021 | 23496 | current_season_success_count_gt_n | 1841 | 0.550788 | 5.0 | 5.00120400000003 | suspicious |
| TRAIN_0815735 | 2022 | 23496 | current_season_success_count_gt_n | 2269 | 0.551785 | 1.0 | 1.0008489999997892 | suspicious |
| TRAIN_1080724 | 2023 | 23496 | current_season_success_count_gt_n | 2679 | 0.550952 | 1.0 | 1.0008559999996578 | suspicious |
| TRAIN_1080740 | 2023 | 23496 | current_season_success_count_gt_n | 2681 | 0.551287 | 3.0 | 3.000894999999673 | suspicious |
| TRAIN_0482592 | 2021 | 23665 | current_season_success_count_negative | 1424 | 0.501404 | 1.0 | -0.0009150000000772707 | suspicious |
| TRAIN_0729050 | 2022 | 23614 | current_season_success_count_negative | 2167 | 0.546377 | 1.0 | -0.0016209999998864077 | suspicious |
| TRAIN_0729051 | 2022 | 23614 | current_season_success_count_negative | 2168 | 0.546125 | 2.0 | -0.001579999999876236 | suspicious |
| TRAIN_0977457 | 2023 | 23614 | current_season_success_count_gt_n | 2828 | 0.542079 | 1.0 | 1.0000529999999799 | suspicious |
| TRAIN_0977458 | 2023 | 23614 | current_season_success_count_gt_n | 2829 | 0.542241 | 2.0 | 2.0004300000000512 | suspicious |
| TRAIN_0977459 | 2023 | 23614 | current_season_success_count_gt_n | 2830 | 0.542403 | 3.0 | 3.0011309999999867 | suspicious |


## 9. Trackman mapping 품질 분석

- mapping 없음/low sample/ambiguous mapping은 삭제보다 flag 또는 Trackman feature fallback이 적절하다.


| entity | mapping_file | status | mapping_rows | train_unique_ids | trackman_unique_ids | mapped_train_ids | mapped_trackman_ids | train_ids_without_mapping | ambiguous_train_ids_multi_tm | ambiguous_tm_ids_multi_train | high_confidence | medium_confidence | low_confidence | exact_game_count_min | exact_game_count_median | ratio_min | ratio_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pitcher | train_trackman/pitcher_id_mapping_clean.csv | available | 760 | 792 | 906 | 760 | 760 | 32 | 0 | 0 | 392 | 202 | 166 | 1.0 | 22.0 | 1.0 | 25.416666666666664 |
| pitcher_data | data/pitcher_id_mapping_clean.csv | available | 760 | 792 | 906 | 760 | 760 | 32 | 0 | 0 | 392 | 202 | 166 | 1.0 | 22.0 | 1.0 | 25.416666666666664 |
| batter | hand_cleaning_analysis/batter_id_mapping_clean.csv | available | 735 | 830 | 913 | 735 | 735 | 95 | 0 | 0 | 410 | 154 | 171 | 1.0 | 31.0 | 1.0 | 30.0 |


## 10. Trackman 물성 outlier 분석

- Trackman season 범위: 2019~2024; 2025 Trackman 존재 여부: False. 2025 Trackman이 있으면 사용 금지로 표시해야 하나, 현재 파일에는 없다.

- 물성 극단값은 개별 pitch 삭제보다 robust aggregation, winsor 후보, outlier flag로 처리하는 것이 안전하다.


| metric | rows | missing | min | p001 | median | p999 | max | impossible_or_rule_outlier_rows | iqr_extreme_rows | severity | unique_values | top_values | contains_2025_trackman | low_sample_lt_30_pitchers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rel_speed | 1785461 | 7617.0 | 71.6818 | 103.13246 | 137.412 | 156.10454000000027 | 160.664 | 0.0 | 40.0 | flag_only |  |  |  |  |
| spin_rate | 1780613 | 12465.0 | 434.9 | 768.365448 | 2223.17 | 3129.3794000000134 | 3695.92 | 0.0 | 31561.0 | flag_only |  |  |  |  |
| induced_vert_break | 1783016 | 10062.0 | -81.8097 | -54.191367 | 30.104950000000002 | 71.72924900000034 | 153.326 | 6.0 | 1.0 | suspicious |  |  |  |  |
| horz_break | 1782745 | 10333.0 | -78.9232 | -55.7731328 | 12.9918 | 62.372621600001025 | 103.699 | 2.0 | 0.0 | suspicious |  |  |  |  |
| extension | 1785362 | 7716.0 | -0.387376 | 1.22614361 | 1.76003 | 2.257792780000004 | 3.84514 | 4.0 | 32.0 | suspicious |  |  |  |  |
| rel_height | 1785461 | 7617.0 | 0.0971153 | 0.31852737999999997 | 1.76115 | 2.1082370000000137 | 2.5121 | 0.0 | 97134.0 | flag_only |  |  |  |  |
| rel_side | 1785460 | 7618.0 | -2.28866 | -1.13748951 | 0.4305135 | 1.30680705000001 | 1.66635 | 0.0 | 1.0 | flag_only |  |  |  |  |
| zone_speed | 1785157 | 7921.0 | 62.7546 | 94.22300919999999 | 125.915 | 142.607 | 148.296 | 0.0 | 67.0 | flag_only |  |  |  |  |
| pitch_type_group | 1793078 | 0.0 |  |  |  |  |  |  |  | no_action | 4.0 | fastball:931120; breaking:512851; offspeed:326809; other:22298 |  |  |
| tagged_pitch_type | 1793078 | 0.0 |  |  |  |  |  |  |  | no_action | 17.0 | Fastball:835886; Slider:359226; Curveball:174359; ChangeUp:170859; Splitter:101926; Sinker:101734; Cutter:37907; Unde... |  |  |
| auto_pitch_type | 1793078 | 72.0 |  |  |  |  |  |  |  | no_action | 11.0 | Fastball:417848; Four-Seam:336375; Slider:328182; Changeup:185311; Curveball:184666; Sinker:121539; ChangeUp:90324; C... |  |  |
| pitcher_hand | 1793078 | 0.0 |  |  |  |  |  |  |  | no_action | 2.0 | Right:1343118; Left:449960 |  |  |
| batter_hand | 1793078 | 0.0 |  |  |  |  |  |  |  | no_action | 2.0 | Right:967537; Left:825541 |  |  |
| season_coverage | 1793078 |  |  |  |  |  |  |  |  | no_action | 6.0 | 2019, 2020, 2021, 2022, 2023, 2024 | False |  |
| pitcher_trackman_sample_size | 906 |  | 2.0 |  | 1150.0 |  | 15671.0 |  |  | flag_only |  |  |  | 46.0 |


## 11. test.csv와 train.csv schema 차이

- 공통 컬럼 48개, train 전용 1개: control_success, test 전용 0개: .

- test row 수와 sample_submission row 수 일치: True (5 vs 5).

- 평가 서버의 실제 test는 교체되므로, test 전체 분포/빈도/순서/rolling을 이용한 예측값 보정은 금지다. 이번 분석은 구조 확인 목적이다.


| column | in_train | in_test | train_dtype | test_dtype | dtype_same | train_unique | test_unique | test_only_categories |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_success | True | False | int64 |  | False | 2 |  |  |


## 12. 모델 전처리에 반영할 추천 action

| issue_type | affected_rows | affected_ids | severity | recommended_action | reason | leakage_risk | model_impact_expected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pitcher_hand_inconsistent | 157 | 15 | suspicious | flag | 소수 hand row가 섞인 투수 ID가 있어 원본 hand 신뢰도가 흔들림 | low | hand split/interaction feature 안정화 |
| batter_hand_inconsistent_or_switch | 7396 | 32 | flag_only | flag | 스위치 타자 가능성이 있어 일괄 삭제는 위험 | low | 타자 hand interaction의 노이즈 격리 |
| trackman_hand_mismatch | 8953 | 718 | suspicious | flag | mapping confidence별로 hand mismatch가 존재; 낮은 confidence는 오류 단정 불가 | medium_if_mapping_uses_future_games | Trackman join feature의 hand 조건부 bias 완화 |
| count_state_invalid | 0 |  | no_action | remove_or_clip_by_issue | 야구 규칙/상태 일관성 위반 row | low | 명백 오류 제거 시 안정화 |
| current_season_reconstruction_anomaly | 1533 | 450 | suspicious | flag | 시즌 시작 누적 차감 방식에서 음수/초과 성공 수 발생 가능 row | low_if_only_asof_used | current-season 파생 피처 clipping/flag 필요성 확인 |
| trackman_physical_outlier | 12 |  | suspicious | flag | Trackman 물성 값 일부가 물리/룰 기준 밖이거나 극단값 | low_for_2019_2024_only | robust aggregation winsor/flag 후보 |


## 13. 삭제 후보와 flag 후보 구분

- 바로 제거 검토 가능한 명백 오류: count/state clear_error row 0개, pitcher hand 소수 row clear_error ID 11개. 실제 제거 전에는 `row_id` 단위 CSV를 재확인해야 한다.

- 삭제하면 위험해서 flag 처리해야 하는 항목: batter hand 양손 후보, low/medium confidence Trackman mismatch, inferred game roster extreme, current-season 복원 이상, Trackman 물성 extreme.

- v5-3/5-6 후보: hand mismatch flag, Trackman mapping confidence flag, current-season cold-start/clip flag, count-state clear_error exclusion/clip.

- 10-3 AutoInt 후보: hand/mapping confidence embedding 또는 binary flags, asof missing/cold-start indicators, Trackman robust aggregate outlier flags.



## 14. 다음 단계 제안

- 삭제 판단 없이 이번 CSV의 `severity`와 `recommended_action` 기준으로 ablation 실험 명령을 별도로 만든다.

- hand는 pitcher clear minority row와 batter switch-like row를 분리해 각각 remove/flag/no-action ablation을 비교한다.

- Trackman은 high confidence mapping만 쓰는 버전과 low/medium fallback 버전을 비교한다.

- current-season 파생 피처는 `current_season_n` small sample smoothing, 음수/초과 clipping, anomaly flag를 포함한 버전만 실험한다.

- 다음 작업 명령어에는 원본 CSV 수정 금지, test 내부 분포 사용 금지, 산출물 row_id audit 저장을 명시한다.



## 생성 산출물

- `output/summary_counts.csv`
- `output/hand_inconsistency_by_pitcher.csv`
- `output/hand_inconsistency_by_batter.csv`
- `output/trackman_hand_mismatch_candidates.csv`
- `output/game_roster_anomaly_candidates.csv`
- `output/count_state_anomaly_rows.csv`
- `output/asof_anomaly_summary.csv`
- `output/current_season_reconstruction_anomaly.csv`
- `output/trackman_mapping_quality_summary.csv`
- `output/trackman_physical_outlier_summary.csv`
- `output/recommended_cleaning_actions.csv`
- `output/train_test_schema_quality.csv`
