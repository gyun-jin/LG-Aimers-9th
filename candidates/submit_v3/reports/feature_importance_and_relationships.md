# LG Aimers 9기 해커톤 - 데이터 분석 및 피처 중요도 결과

본 문서는 `agy.md`의 요청에 따라 공식 자료와 데이터를 기반으로 작성되었습니다.

## 1. 데이터 구조 및 기본 현황 정리

- **train.csv**: 1,475,092 행, 49 열
- **test.csv**: 5 행, 48 열

### 주요 데이터 타입
- int64: 26 개
- float64: 19 개
- object: 4 개

## 2. 주요 변수의 의미와 분포 및 결측치 현황

| 변수명 | 타입 | 결측치 수 | 고유값 수 | 예시 (상위 3개) |
|---|---|---|---|---|
| row_id | object | 0 | 1,475,092 | TRAIN_0000001, TRAIN_0983403, TRAIN_0983401 |
| season | int64 | 0 | 6 | 2024, 2022, 2021 |
| game_month | int64 | 0 | 8 | 5, 6, 9 |
| game_dayofweek | int64 | 0 | 7 | 5, 6, 4 |
| inning | int64 | 0 | 13 | 1, 7, 8 |
| top_bottom | object | 0 | 2 | T, B |
| game_type | object | 0 | 2 | R, F |
| balls_before | int64 | 0 | 4 | 0, 1, 2 |
| strikes_before | int64 | 0 | 3 | 0, 1, 2 |
| outs_before | int64 | 0 | 3 | 0, 1, 2 |
| run_top_before | int64 | 0 | 30 | 0, 1, 2 |
| run_bot_before | int64 | 0 | 25 | 0, 1, 2 |
| run_total_before | int64 | 0 | 38 | 0, 2, 1 |
| score_diff_home | int64 | 0 | 47 | 0, -1, 1 |
| score_diff_pitcher_team | int64 | 0 | 52 | 0, 1, -1 |
| runner_on_1b | int64 | 0 | 2 | 0, 1 |
| runner_on_2b | int64 | 0 | 2 | 0, 1 |
| runner_on_3b | int64 | 0 | 2 | 0, 1 |
| num_runners_on | int64 | 0 | 4 | 0, 1, 2 |
| base_state | object | 0 | 8 | ___, 1__, 12_ |
| home_win_expectancy | float64 | 0 | 977 | 50.0, 52.2, 54.8 |
| away_win_expectancy | float64 | 0 | 981 | 50.0, 47.8, 45.2 |
| li | float64 | 0 | 502 | 0.87, 0.62, 0.01 |
| pitcher_id | int64 | 0 | 792 | 23633, 23719, 22759 |
| batter_id | int64 | 0 | 830 | 22660, 22816, 23420 |
| pitcher_hand | int64 | 0 | 2 | 2, 1 |
| batter_hand | int64 | 0 | 2 | 2, 1 |
| pitcher_team_id | int64 | 0 | 13 | 13, 17, 21 |
| batter_team_id | int64 | 0 | 13 | 13, 14, 17 |
| asof_pitcher_n | int64 | 0 | 15,450 | 0, 1, 2 |
| asof_pitcher_success_rate | float64 | 792 | 220,896 | 0.5, 0.666667, 0.6 |
| asof_pitcher_reverse_rate | float64 | 792 | 228,112 | 0.0, 0.25, 0.2 |
| asof_pitcher_middle_rate | float64 | 792 | 120,993 | 0.0, 0.142857, 0.125 |
| asof_pitcher_ball_rate | float64 | 792 | 144,809 | 0.333333, 0.4, 0.5 |
| asof_pitcher_strike_rate | float64 | 792 | 131,532 | 0.5, 0.4, 0.428571 |
| asof_pitcher_prev1_game_success_rate | float64 | 29,185 | 1,924 | 0.5, 0.666667, 0.6 |
| asof_pitcher_prev3_game_success_rate | float64 | 29,185 | 6,319 | 0.5, 0.571429, 0.6 |
| asof_pitcher_prev5_game_success_rate | float64 | 29,185 | 9,314 | 0.5, 0.6, 0.571429 |
| asof_pitcher_prev1_game_middle_rate | float64 | 29,185 | 1,337 | 0.0, 0.166667, 0.2 |
| asof_pitcher_prev3_game_middle_rate | float64 | 29,185 | 4,653 | 0.166667, 0.142857, 0.2 |
| asof_pitcher_prev5_game_middle_rate | float64 | 29,185 | 7,296 | 0.142857, 0.125, 0.166667 |
| asof_batter_n | int64 | 0 | 13,928 | 0, 1, 2 |
| asof_batter_success_rate | float64 | 830 | 166,832 | 0.5, 0.666667, 0.6 |
| asof_batter_middle_rate | float64 | 830 | 92,883 | 0.0, 0.142857, 0.125 |
| asof_pitcher_pitchmix_n | int64 | 0 | 15,450 | 0, 1, 2 |
| asof_pitcher_fastball_rate | float64 | 792 | 337,588 | 0.5, 0.666667, 1.0 |
| asof_pitcher_breaking_rate | float64 | 792 | 386,613 | 0.0, 0.333333, 0.25 |
| asof_pitcher_offspeed_rate | float64 | 792 | 449,454 | 0.0, 0.2, 0.166667 |
| control_success | int64 | 0 | 2 | 1, 0 |

## 3. target(control_success)과 상관관계가 큰 변수 추출

### Target 상관계수 (Top 15)
| 변수명 | 상관계수 |
|---|---|
| asof_pitcher_success_rate | 0.0843 |
| asof_pitcher_prev5_game_success_rate | 0.0816 |
| asof_pitcher_reverse_rate | -0.0795 |
| asof_pitcher_prev3_game_success_rate | 0.0778 |
| asof_pitcher_prev1_game_success_rate | 0.0617 |
| asof_batter_success_rate | 0.0589 |
| season | -0.0483 |
| asof_batter_n | -0.0363 |
| asof_pitcher_middle_rate | -0.0361 |
| asof_batter_middle_rate | -0.0338 |
| asof_pitcher_prev5_game_middle_rate | -0.0320 |
| asof_pitcher_prev3_game_middle_rate | -0.0283 |
| asof_pitcher_prev1_game_middle_rate | -0.0195 |
| asof_pitcher_ball_rate | -0.0177 |
| asof_pitcher_offspeed_rate | 0.0138 |

## 4. 수치형 변수 간 상관관계 분석 (다중공선성 의심 변수)

상관계수의 절대값이 0.7 이상인 변수 쌍입니다.

| 변수 1 | 변수 2 | 상관계수 |
|---|---|---|
| asof_pitcher_n | asof_pitcher_pitchmix_n | 1.0000 |
| home_win_expectancy | away_win_expectancy | -1.0000 |
| score_diff_home | away_win_expectancy | -0.9048 |
| score_diff_home | home_win_expectancy | 0.9048 |
| asof_pitcher_prev3_game_success_rate | asof_pitcher_prev5_game_success_rate | 0.8822 |
| asof_pitcher_prev3_game_middle_rate | asof_pitcher_prev5_game_middle_rate | 0.8463 |
| asof_pitcher_success_rate | asof_pitcher_reverse_rate | -0.8099 |
| run_top_before | run_total_before | 0.8042 |
| run_bot_before | run_total_before | 0.7797 |
| asof_pitcher_ball_rate | asof_pitcher_strike_rate | -0.7734 |
| runner_on_1b | num_runners_on | 0.7457 |
| runner_on_2b | num_runners_on | 0.7051 |

## 5. 범주형 변수와 target 간 관계 정리

### top_bottom
| 카테고리 | 제구 성공률(mean) | 표본 수(count) |
|---|---|---|
| T | 0.5231 | 752,812 |
| B | 0.5244 | 722,280 |

### game_type
| 카테고리 | 제구 성공률(mean) | 표본 수(count) |
|---|---|---|
| R | 0.5140 | 1,314,088 |
| F | 0.6033 | 161,004 |

### base_state
| 카테고리 | 제구 성공률(mean) | 표본 수(count) |
|---|---|---|
| ___ | 0.5230 | 777,248 |
| 1__ | 0.5253 | 293,724 |
| 12_ | 0.5213 | 125,831 |
| _2_ | 0.5249 | 113,627 |
| 1_3 | 0.5279 | 48,839 |
| 123 | 0.5161 | 48,408 |
| __3 | 0.5361 | 35,513 |
| _23 | 0.5249 | 31,902 |

## 6. 상황별 변수 영향도 분석

주요 상황 변수(볼카운트, 이닝, 주자 상황 등)에 따른 제구 성공률 변화입니다.

### balls_before 에 따른 제구 성공률
| 값 | 성공률 | 표본 수 |
|---|---|---|
| 0 | 0.5276 | 652,052 |
| 1 | 0.5259 | 444,780 |
| 2 | 0.5208 | 254,237 |
| 3 | 0.5021 | 124,023 |

### strikes_before 에 따른 제구 성공률
| 값 | 성공률 | 표본 수 |
|---|---|---|
| 0 | 0.5249 | 608,121 |
| 1 | 0.5276 | 447,852 |
| 2 | 0.5180 | 419,119 |

### inning 에 따른 제구 성공률
| 값 | 성공률 | 표본 수 |
|---|---|---|
| 1 | 0.5321 | 172,012 |
| 2 | 0.5313 | 162,861 |
| 3 | 0.5295 | 163,893 |
| 4 | 0.5266 | 163,809 |
| 5 | 0.5239 | 164,439 |
| 6 | 0.5241 | 166,686 |
| 7 | 0.5172 | 168,702 |
| 8 | 0.5145 | 168,021 |
| 9 | 0.5150 | 124,717 |
| 10 | 0.5041 | 11,643 |
| 11 | 0.5038 | 5,441 |
| 12 | 0.5101 | 2,811 |
| 13 | 0.5088 | 57 |

### num_runners_on 에 따른 제구 성공률
| 값 | 성공률 | 표본 수 |
|---|---|---|
| 0 | 0.5230 | 777,248 |
| 1 | 0.5260 | 442,864 |
| 2 | 0.5234 | 206,572 |
| 3 | 0.5161 | 48,408 |

## 7. 변수 간 상호작용 분석

두 가지 변수가 결합되었을 때의 제구 성공률입니다.

### 투수 손(pitcher_hand) × 타자 손(batter_hand)
| 투수 손 | 타자 손 | 성공률 | 표본 수 |
|---|---|---|---|
| 1 | 1 | 0.4909 | 170,292 |
| 1 | 2 | 0.5375 | 211,059 |
| 2 | 1 | 0.5307 | 525,455 |
| 2 | 2 | 0.5221 | 568,286 |

## 8. 모델 기반 변수 중요도 계산

결측치를 처리하고 범주형 변수를 인코딩한 후, RandomForest (또는 LightGBM)을 이용한 트리 기반 변수 중요도입니다. (메모리 및 시간 절약을 위해 10만 개 샘플링 학습)

| 순위 | 변수명 | 중요도(Gini) |
|---|---|---|
| 1 | asof_pitcher_reverse_rate | 0.0645 |
| 2 | asof_pitcher_success_rate | 0.0610 |
| 3 | asof_batter_success_rate | 0.0534 |
| 4 | asof_pitcher_prev5_game_success_rate | 0.0504 |
| 5 | asof_pitcher_prev3_game_success_rate | 0.0472 |
| 6 | asof_pitcher_ball_rate | 0.0412 |
| 7 | asof_batter_middle_rate | 0.0402 |
| 8 | asof_pitcher_strike_rate | 0.0339 |
| 9 | asof_pitcher_fastball_rate | 0.0328 |
| 10 | asof_batter_n | 0.0324 |
| 11 | asof_pitcher_middle_rate | 0.0317 |
| 12 | asof_pitcher_n | 0.0315 |
| 13 | asof_pitcher_prev1_game_success_rate | 0.0314 |
| 14 | asof_pitcher_breaking_rate | 0.0310 |
| 15 | asof_pitcher_pitchmix_n | 0.0310 |
| 16 | asof_pitcher_offspeed_rate | 0.0309 |
| 17 | asof_pitcher_prev5_game_middle_rate | 0.0283 |
| 18 | asof_pitcher_prev3_game_middle_rate | 0.0273 |
| 19 | away_win_expectancy | 0.0253 |
| 20 | li | 0.0253 |

## 9. 대회 규칙상 금지된 변수 및 데이터 누수(Data Leakage) 위험 변수 구분

대회 규칙에 명시된 바에 따라 다음 항목은 예측에 사용해서는 안 됩니다:
- **미래/현재 정보**: 현재 투구의 구종, 위치, 결과, Trackman 기록 (속도, 회전수 등)은 사용 불가.
- **Data Leakage 주의**: `test.csv` 내부의 다른 행을 이용한 통계(Target Encoding, Rolling 등) 생성 금지.
- **Trackman History**: `trackman_history.csv`의 데이터는 오직 **과거 시점**의 통계(선수별 평균 구속, 회전수 등)를 사전 계산하는 데만 사용해야 하며, 현재 투구와 직접 결합할 수 없음.

## 10. 성능 개선을 위한 피처 엔지니어링 아이디어 (우선순위별)

### 1순위 (높은 효과 예상)
- `asof_pitcher_success_rate` 등 누적 확률 피처의 신뢰도 보정: 표본 수(`asof_pitcher_n`)가 적은 경우 전체 평균으로 Bayesian Smoothing (Cold-start 문제 해결).
- `pitcher_id` 및 `batter_id` 기반 파생 변수: Trackman History를 활용해 해당 투수의 주력 구종, 평균 구속/무브먼트 피처를 생성 후 병합.
- 상황 압박도 결합 피처: `li` (상황 중요도)가 높을 때 해당 투수의 과거 제구 성공률 변화량(Clutch 능력) 피처화.

### 2순위 (상호작용 및 파생 변수)
- 카운트 변수 세분화: `balls_before` - `strikes_before`를 통해 유리한/불리한 카운트(Count Advantage) 명시적 생성.
- 피로도 지표: `inning`과 `asof_pitcher_n`(누적 투구 수)의 비율 또는 곱을 통해 현재 피로도 추정.
- 투타 상성: `pitcher_hand`와 `batter_hand` 조합에 따른 이점(예: L vs L)을 원-핫 인코딩 또는 가중치로 부여.

### 3순위 (고급 이력 요약 - Trackman 활용)
- 최근 폼 지표: 최근 N경기 제구 성공률(`asof_pitcher_prev1_game_success_rate` 등)간의 차이(Momentum)를 구해 상승세/하락세 반영.
- 구종 다양성: `pitchmix_n`, `fastball_rate` 등의 비율 정보로 엔트로피(Entropy)를 계산하여 투수의 패턴 단순화 정도 파악.
