# 현재 시즌 성공률 피처 보고서

## 목적

기존 `asof_pitcher_n`, `asof_pitcher_success_rate`는 시즌 경계에서 리셋되지 않는 통산 누적값이다. `/5-3`에서는 투수별 이전 시즌 종료 시점 누적값을 학습 state에 저장하고, 각 row의 asof 값에서 이를 빼 현재 시즌 성과를 복원했다.

## 계산 방식

```text
asof_pitcher_success_count = asof_pitcher_n * asof_pitcher_success_rate
pitcher_current_season_n = asof_pitcher_n - prior_season_end_pitcher_n
pitcher_current_season_success_count = asof_pitcher_success_count - prior_season_end_pitcher_success_count
pitcher_current_season_success_rate = pitcher_current_season_success_count / pitcher_current_season_n
```

현재 시즌 표본이 없으면 학습 target prior로 대체하고, `pitcher_current_season_available_flag`, `pitcher_current_season_small_sample_flag`를 함께 제공한다.

## 추가 피처

- `pitcher_current_season_n`
- `log1p_pitcher_current_season_n`
- `pitcher_current_season_success_count`
- `pitcher_current_season_success_rate`
- `pitcher_current_season_success_rate_smoothed`
- `pitcher_current_season_success_minus_career`
- `pitcher_current_season_success_minus_prev1`
- `pitcher_current_season_success_minus_prev3`
- `pitcher_current_season_success_minus_prev5`
- `pitcher_current_season_n_ratio_to_career`
- `pitcher_current_season_available_flag`
- `pitcher_current_season_small_sample_flag`

## interaction 피처

- `pitcher_current_season_success_x_count_state`
- `pitcher_current_season_success_x_game_type`
- `pitcher_current_season_success_x_li_bin`
- `pitcher_current_season_success_x_base_state`
- `pitcher_current_season_success_x_hand_matchup`
- `pitcher_current_season_success_x_tm_available`
- `pitcher_current_season_success_x_tm_mapping_confidence_bucket`

## 누수 방지

- train row는 해당 row의 `asof_*` 값과 이전 시즌 종료 상수표만 사용한다.
- validation fold에서는 validation season target으로 prior table을 만들지 않는다.
- test에서는 저장된 prior table과 해당 row 자체의 asof 값만 사용한다.
- test 전체 groupby, value_counts, rolling, lag, rank, 누적 계산은 사용하지 않는다.
- test row 하나만 들어와도 같은 값이 계산된다.
