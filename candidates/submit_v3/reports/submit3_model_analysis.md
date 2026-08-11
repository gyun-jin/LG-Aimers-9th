# 3/submit.zip Model Analysis

작성 기준: `3/submit.zip` 내부 파일을 직접 확인했다. zip 바깥의 `solution_v3/` 파일과 값이 다를 수 있으므로, 이 문서는 실제 제출 압축본만 기준으로 한다.

## 1. 제출 파일 구성

`3/submit.zip` 내부 구성은 다음 6개 파일이다.

| 파일 | 역할 | 크기 |
|---|---:|---:|
| `model/final_model.joblib` | 최종 앙상블 모델 bundle | 1,098,263 bytes |
| `model/calibration_model.joblib` | Platt calibration 모델 | 574 bytes |
| `model/feature_schema.json` | 입력 컬럼, component별 피처/인코딩 schema | 143,489 bytes |
| `model/ensemble_config.json` | 앙상블 가중치와 선택 기준 | 248 bytes |
| `script.py` | 평가 서버용 독립 추론 스크립트 | 12,895 bytes |
| `requirements.txt` | 추론 의존성 | 48 bytes |

의존성은 다음 버전으로 고정되어 있다.

```text
catboost==1.2.10
lightgbm==4.7.0
xgboost==3.2.0
```

## 2. 최종 앙상블과 가중치

`model/ensemble_config.json` 및 `model/final_model.joblib`를 모두 확인한 실제 가중치는 다음과 같다.

| 모델 | 실제 가중치 | 최종 예측 반영 여부 |
|---|---:|---|
| CatBoostClassifier | 0.65 | 반영 |
| LGBMClassifier | 0.00 | bundle에는 있으나 최종 예측에는 영향 없음 |
| XGBClassifier | 0.35 | 반영 |

선택된 앙상블 이름은 `catboost_lightgbm_xgboost`다. 다만 LightGBM 가중치가 0이므로 수식상 최종 raw prediction은 아래와 같다.

```text
raw_pred = 0.65 * catboost_pred + 0.00 * lightgbm_pred + 0.35 * xgboost_pred
```

선택 기준은 `2024 temporal OOF Brier`이고, 가중치 탐색 간격은 0.05다.

## 3. 보정 Calibration

최종 raw prediction에는 Platt scaling을 적용한다.

| 항목 | 값 |
|---|---|
| calibration kind | `platt` |
| 모델 | `sklearn.linear_model.LogisticRegression` |
| 입력 | raw probability를 logit으로 변환한 1개 feature |
| 계수 | 1.26020599 |
| 절편 | -0.04793917 |

추론 코드의 보정 흐름은 다음과 같다.

```text
clipped = clip(raw_pred, 1e-6, 1 - 1e-6)
logit = log(clipped / (1 - clipped))
final_pred = LogisticRegression.predict_proba(logit)[:, 1]
final_pred = clip(final_pred, 0, 1)
```

## 4. 입력 Schema

입력 컬럼은 `row_id`를 제외하고 정확히 47개이며, 컬럼 순서까지 강제한다. `script.py`는 test 컬럼 순서가 schema와 다르면 즉시 오류를 낸다.

```text
season
game_month
game_dayofweek
inning
top_bottom
game_type
balls_before
strikes_before
outs_before
run_top_before
run_bot_before
run_total_before
score_diff_home
score_diff_pitcher_team
runner_on_1b
runner_on_2b
runner_on_3b
num_runners_on
base_state
home_win_expectancy
away_win_expectancy
li
pitcher_id
batter_id
pitcher_hand
batter_hand
pitcher_team_id
batter_team_id
asof_pitcher_n
asof_pitcher_success_rate
asof_pitcher_reverse_rate
asof_pitcher_middle_rate
asof_pitcher_ball_rate
asof_pitcher_strike_rate
asof_pitcher_prev1_game_success_rate
asof_pitcher_prev3_game_success_rate
asof_pitcher_prev5_game_success_rate
asof_pitcher_prev1_game_middle_rate
asof_pitcher_prev3_game_middle_rate
asof_pitcher_prev5_game_middle_rate
asof_batter_n
asof_batter_success_rate
asof_batter_middle_rate
asof_pitcher_pitchmix_n
asof_pitcher_fastball_rate
asof_pitcher_breaking_rate
asof_pitcher_offspeed_rate
```

타깃은 `control_success`다. schema에 저장된 타깃 정의는 "현재 투구가 가운데 위험 코스, 스트라이크존 대폭 이탈, 포수 요구 반대 방향에 해당하지 않는 유효한 제구 성공의 사전 확률"이다.

## 5. 모델별 객체와 주요 파라미터

### 5.1 CatBoost

| 항목 | 값 |
|---|---|
| class | `catboost.core.CatBoostClassifier` |
| weight | 0.65 |
| feature 수 | 91 |
| categorical feature 수 | 14 |
| classes | `[0, 1]` |
| iterations / tree_count | 108 / 108 |
| learning_rate | 0.03 |
| depth | 9 |
| loss_function | `Logloss` |
| eval_metric | `BrierScore` |
| l2_leaf_reg | 20.0 |
| random_seed | 42 |
| encoder | 없음. CatBoost native categorical 처리 |
| redundant raw columns | 제거 |

### 5.2 LightGBM

| 항목 | 값 |
|---|---|
| class | `lightgbm.sklearn.LGBMClassifier` |
| weight | 0.00 |
| feature 수 | 96 |
| categorical feature 수 | 14 |
| classes | `[0, 1]` |
| n_estimators | 100 |
| learning_rate | 0.04 |
| num_leaves | 31 |
| max_depth | -1 |
| min_child_samples | 500 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| reg_lambda | 20.0 |
| objective | `binary` |
| random_state | 42 |
| n_jobs | 6 |
| encoder | category code, unknown은 -1 |
| redundant raw columns | 유지 |

LightGBM은 bundle에 포함되어 있고 추론 시 predict도 수행되지만, weight가 0.00이라 최종 raw prediction에는 영향을 주지 않는다.

### 5.3 XGBoost

| 항목 | 값 |
|---|---|
| class | `xgboost.sklearn.XGBClassifier` |
| weight | 0.35 |
| feature 수 | 91 |
| categorical feature 수 | 14 |
| classes | `[0, 1]` |
| n_estimators | 100 |
| learning_rate | 0.04 |
| max_depth | 7 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| reg_lambda | 10.0 |
| objective | `binary:logistic` |
| eval_metric | `logloss` |
| min_child_weight | 20 |
| random_state | 42 |
| n_jobs | 6 |
| encoder | frequency encoding, unknown은 0.0 |
| redundant raw columns | 제거 |

## 6. 범주형 컬럼

세 모델이 공통으로 사용하는 범주형 컬럼은 14개다.

```text
top_bottom
game_type
base_state
game_dayofweek
pitcher_id
batter_id
pitcher_hand
batter_hand
pitcher_team_id
batter_team_id
hand_matchup
count_state
runner_out_state
pressure_state
```

모델별 처리 방식은 다음과 같다.

| 모델 | 처리 방식 |
|---|---|
| CatBoost | 문자열로 변환 후 native categorical feature로 전달 |
| LightGBM | 저장된 category code mapping 적용, 미등록 범주는 -1 |
| XGBoost | 저장된 frequency mapping 적용, 미등록 범주는 0.0 |

저장된 mapping 크기는 다음과 같다.

| 컬럼 | mapping 개수 |
|---|---:|
| `top_bottom` | 2 |
| `game_type` | 2 |
| `base_state` | 8 |
| `game_dayofweek` | 7 |
| `pitcher_id` | 792 |
| `batter_id` | 830 |
| `pitcher_hand` | 2 |
| `batter_hand` | 2 |
| `pitcher_team_id` | 13 |
| `batter_team_id` | 13 |
| `hand_matchup` | 4 |
| `count_state` | 12 |
| `runner_out_state` | 24 |
| `pressure_state` | 8 |

## 7. 파생피처

`script.py`의 `build_features()`가 생성하는 파생피처는 총 49개다. test 내부 집계나 test 분포 학습 없이, 각 row의 입력값과 저장된 학습 prior만 사용한다.

### 7.1 상황 조합 피처

| 피처 | 정의 |
|---|---|
| `hand_matchup` | `pitcher_hand + "_" + batter_hand` |
| `count_state` | `balls_before + "-" + strikes_before` |
| `runner_out_state` | `base_state + "_o" + outs_before` |
| `close_game` | `abs(score_diff_pitcher_team) <= 2` |
| `late_inning` | `inning >= 7` |
| `pressure_state` | `late{late_inning}_close{close_game}_li{li >= 1.5}` |

### 7.2 표본 수 로그 피처

| 피처 | 정의 |
|---|---|
| `log1p_asof_pitcher_n` | `log1p(max(asof_pitcher_n, 0))` |
| `log1p_asof_batter_n` | `log1p(max(asof_batter_n, 0))` |

### 7.3 Smoothed rate 피처

아래 10개 rate에 대해 smoothing 피처를 만든다.

```text
asof_pitcher_success_rate_smoothed
asof_pitcher_reverse_rate_smoothed
asof_pitcher_middle_rate_smoothed
asof_pitcher_ball_rate_smoothed
asof_pitcher_strike_rate_smoothed
asof_batter_success_rate_smoothed
asof_batter_middle_rate_smoothed
asof_pitcher_fastball_rate_smoothed
asof_pitcher_breaking_rate_smoothed
asof_pitcher_offspeed_rate_smoothed
```

공식은 다음과 같다.

```text
smoothed = (count * observed_rate + alpha * prior) / (count + alpha)
alpha = 50.0
```

rate별 count 컬럼은 다음과 같다.

| rate 컬럼 | count 컬럼 |
|---|---|
| pitcher success/reverse/middle/ball/strike rate | `asof_pitcher_n` |
| batter success/middle rate | `asof_batter_n` |
| pitcher fastball/breaking/offspeed rate | `asof_pitcher_pitchmix_n` |

저장된 prior는 다음과 같다.

| rate | prior |
|---|---:|
| `asof_pitcher_success_rate` | 0.5237659752747625 |
| `asof_pitcher_reverse_rate` | 0.21850676683367742 |
| `asof_pitcher_middle_rate` | 0.14517203514148141 |
| `asof_pitcher_ball_rate` | 0.36262177743797613 |
| `asof_pitcher_strike_rate` | 0.44816401187508975 |
| `asof_batter_success_rate` | 0.5237659752747625 |
| `asof_batter_middle_rate` | 0.1444071141311529 |
| `asof_pitcher_fastball_rate` | 0.5278964420740577 |
| `asof_pitcher_breaking_rate` | 0.2944599032085815 |
| `asof_pitcher_offspeed_rate` | 0.1776436547623318 |

### 7.4 최근 경기 rate 요약

최근 1, 3, 5경기 pitcher success rate에서 생성한다.

```text
recent_success_mean_1_3_5
recent_success_range_1_3_5
recent_success_std_1_3_5
recent_gap_1
recent_gap_3
recent_gap_5
```

`recent_gap_{1,3,5}`는 각 최근 성공률에서 `asof_pitcher_success_rate_smoothed`를 뺀 값이다.

최근 1, 3, 5경기 pitcher middle rate에서 생성한다.

```text
recent_middle_mean_1_3_5
recent_middle_range_1_3_5
recent_middle_vs_cumulative
```

`recent_middle_vs_cumulative`는 `recent_middle_mean_1_3_5 - asof_pitcher_middle_rate_smoothed`다.

### 7.5 결측 플래그

학습 시 결측이 있었던 16개 컬럼에 대해 `__missing` 플래그를 만든다.

```text
asof_batter_middle_rate__missing
asof_batter_success_rate__missing
asof_pitcher_ball_rate__missing
asof_pitcher_breaking_rate__missing
asof_pitcher_fastball_rate__missing
asof_pitcher_middle_rate__missing
asof_pitcher_offspeed_rate__missing
asof_pitcher_prev1_game_middle_rate__missing
asof_pitcher_prev1_game_success_rate__missing
asof_pitcher_prev3_game_middle_rate__missing
asof_pitcher_prev3_game_success_rate__missing
asof_pitcher_prev5_game_middle_rate__missing
asof_pitcher_prev5_game_success_rate__missing
asof_pitcher_reverse_rate__missing
asof_pitcher_strike_rate__missing
asof_pitcher_success_rate__missing
```

### 7.6 Cold-start, pitch mix, score 피처

| 피처 | 정의 |
|---|---|
| `is_pitcher_cold_start` | `asof_pitcher_n <= 0` |
| `is_batter_cold_start` | `asof_batter_n <= 0` |
| `pitchmix_entropy` | fastball/breaking/offspeed rate의 entropy |
| `score_abs` | `abs(score_diff_pitcher_team)` |
| `is_pitcher_team_leading` | `score_diff_pitcher_team > 0` |
| `is_pitcher_team_trailing` | `score_diff_pitcher_team < 0` |

## 8. 중복 원본 컬럼 처리

중복 또는 준중복으로 지정된 원본 컬럼은 다음 5개다.

```text
asof_pitcher_pitchmix_n
run_total_before
num_runners_on
away_win_expectancy
asof_pitcher_offspeed_rate
```

모델별 처리:

| 모델 | 처리 |
|---|---|
| CatBoost | 제거 |
| LightGBM | 유지 |
| XGBoost | 제거 |

따라서 CatBoost와 XGBoost는 `47 + 49 - 5 = 91`개 feature를 사용하고, LightGBM은 `47 + 49 = 96`개 feature를 사용한다.

## 9. 추론 안전장치

`script.py`에는 다음 검증이 들어 있다.

| 단계 | 검증 |
|---|---|
| test load | `row_id` 존재 확인 |
| sample submission load | 컬럼이 정확히 `row_id`, `control_success`인지 확인 |
| input schema | `row_id` 제외 컬럼 순서가 저장 schema와 정확히 일치하는지 확인 |
| feature build | 누락/추가 컬럼 있으면 오류 |
| component feature schema | 생성된 feature 컬럼이 component별 저장 schema와 다르면 오류 |
| prediction merge | 중복 row_id, ID 집합 불일치, 행 수 불일치, NaN/inf, [0,1] 밖 값 차단 |
| output | `./output/submission.csv` 저장 |

입력 경로는 `./open/test.csv`, `./open/sample_submission.csv`를 우선 사용하고, 없으면 `./data/test.csv`, `./data/sample_submission.csv`를 fallback으로 사용한다.

## 10. 핵심 요약

최종 제출 모델은 CatBoost와 XGBoost 중심의 확률 앙상블이다. LightGBM은 bundle에 포함되어 있지만 실제 가중치가 0이므로 예측값에는 반영되지 않는다.

최종 예측 흐름은 다음과 같다.

```text
1. test.csv 47개 입력 컬럼 순서 검증
2. component별 파생피처 생성
3. CatBoost native categorical 예측
4. LightGBM category-code 예측, 단 weight=0
5. XGBoost frequency-encoded 예측
6. raw_pred = 0.65 * CatBoost + 0.35 * XGBoost
7. Platt scaling 적용
8. sample_submission row_id 순서에 맞춰 output/submission.csv 저장
```
