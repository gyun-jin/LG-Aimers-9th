# `/1/submit.zip` 피처 엔지니어링 상세 분석

## 1. 전체 변환 구조

`/1/submit.zip`은 공식 test 입력 47개에서 49개 파생 피처를 추가합니다.

```text
공식 원본 입력 47개
  + 범주 결합 3개
  + 경기 압박 상태 3개
  + 표본 수 로그 2개
  + 비율 평활 10개
  + 최근 경기 추세 9개
  + 결측 플래그 16개
  + cold-start 플래그 2개
  + 기타 수치 파생 4개
  = 최대 96개
```

- LightGBM: 96개 전부 사용
- CatBoost: 중복 피처 5개를 제거해 91개 사용
- XGBoost: 중복 피처 5개를 제거해 91개 사용

`row_id`는 피처가 아니며 결과 정렬에만 사용합니다. `control_success`는 학습 정답으로 test 입력에는 없습니다.

## 2. 범주 결합 피처 3개

| 파생 변수 | 계산 | 목적 |
|---|---|---|
| `hand_matchup` | `pitcher_hand + "_" + batter_hand` | 좌투수·우타자 등 투타 손 조합 효과 |
| `count_state` | `balls_before + "-" + strikes_before` | 0-0, 3-2 등 카운트별 투구 전략 차이 |
| `runner_out_state` | `base_state + "_o" + outs_before` | 동일 주자 상황에서도 아웃카운트에 따른 압박 차이 |

문자열 결측은 `__MISSING__`으로 바꿉니다. 가능한 상태 수는 다음과 같습니다.

- `hand_matchup`: 4개
- `count_state`: 12개
- `runner_out_state`: 24개

## 3. 경기 압박 피처 3개

| 파생 변수 | 계산 | 의미 |
|---|---|---|
| `close_game` | `abs(score_diff_pitcher_team) <= 2` | 2점 차 이내 접전 여부 |
| `late_inning` | `inning >= 7` | 7회 이후 여부 |
| `pressure_state` | `late_inning`, `close_game`, `li>=1.5` 문자열 결합 | 후반·접전·고중요도 상황의 상호작용 |

`pressure_state`는 최대 8개 조합입니다. 예를 들어 `late1_close1_li1`은 7회 이후, 2점 차 이내, 레버리지 지수 1.5 이상인 상황입니다.

이 방식은 비선형 상호작용을 모델이 더 쉽게 학습하게 하지만 `2점`, `7회`, `li=1.5`라는 임계값이 고정돼 있다는 한계가 있습니다.

## 4. 표본 수 로그 피처 2개

```text
log1p_asof_pitcher_n = log(1 + asof_pitcher_n)
log1p_asof_batter_n  = log(1 + asof_batter_n)
```

| 파생 변수 | 목적 |
|---|---|
| `log1p_asof_pitcher_n` | 투수 과거 투구 수의 긴 꼬리 완화 |
| `log1p_asof_batter_n` | 타자 과거 상대 투구 수의 긴 꼬리 완화 |

원본 표본 수도 유지하므로 모델은 절대 규모와 로그 규모를 함께 봅니다.

## 5. 비율 평활 피처 10개

표본이 적은 선수의 0% 또는 100% 같은 극단 비율을 그대로 믿지 않도록 경험적 베이즈 형태의 평활을 적용합니다.

```text
smoothed_rate = (n × observed_rate + alpha × global_prior) / (n + alpha)
alpha = 50
```

- `n=0`: 전체 학습 prior와 같아집니다.
- `n=50`: 관측 비율과 prior가 1:1로 섞입니다.
- `n`이 매우 큼: 원래 관측 비율에 가까워집니다.
- 관측 비율이 결측이면 prior로 대체한 뒤 계산합니다.

| 원본 비율 | 표본 수 | 파생 변수 |
|---|---|---|
| `asof_pitcher_success_rate` | `asof_pitcher_n` | `asof_pitcher_success_rate_smoothed` |
| `asof_pitcher_reverse_rate` | `asof_pitcher_n` | `asof_pitcher_reverse_rate_smoothed` |
| `asof_pitcher_middle_rate` | `asof_pitcher_n` | `asof_pitcher_middle_rate_smoothed` |
| `asof_pitcher_ball_rate` | `asof_pitcher_n` | `asof_pitcher_ball_rate_smoothed` |
| `asof_pitcher_strike_rate` | `asof_pitcher_n` | `asof_pitcher_strike_rate_smoothed` |
| `asof_batter_success_rate` | `asof_batter_n` | `asof_batter_success_rate_smoothed` |
| `asof_batter_middle_rate` | `asof_batter_n` | `asof_batter_middle_rate_smoothed` |
| `asof_pitcher_fastball_rate` | `asof_pitcher_pitchmix_n` | `asof_pitcher_fastball_rate_smoothed` |
| `asof_pitcher_breaking_rate` | `asof_pitcher_pitchmix_n` | `asof_pitcher_breaking_rate_smoothed` |
| `asof_pitcher_offspeed_rate` | `asof_pitcher_pitchmix_n` | `asof_pitcher_offspeed_rate_smoothed` |

최종 2019~2024 전체 학습에서 저장된 prior는 다음과 같습니다.

| 통계 | prior |
|---|---:|
| 투수 성공률 | 0.52376598 |
| 투수 요구 반대 방향 비율 | 0.21850677 |
| 투수 가운데 비율 | 0.14517204 |
| 투수 볼 비율 | 0.36262178 |
| 투수 스트라이크 비율 | 0.44816401 |
| 타자 상대 성공률 | 0.52376598 |
| 타자 상대 가운데 비율 | 0.14440711 |
| 패스트볼 비율 | 0.52789644 |
| 브레이킹볼 비율 | 0.29445990 |
| 오프스피드 비율 | 0.17764365 |

검증 시에는 각 fold 학습 구간에서 prior를 다시 계산하고, 최종 2025 추론용 모델에서만 2019~2024 전체 prior를 저장합니다.

## 6. 최근 경기 추세 피처 9개

### 최근 성공률 6개

원본 입력은 이전 1·3·5경기의 투수 성공률입니다.

| 파생 변수 | 계산 |
|---|---|
| `recent_success_mean_1_3_5` | prev1, prev3, prev5 성공률 평균 |
| `recent_success_range_1_3_5` | 세 값의 최댓값 - 최솟값 |
| `recent_success_std_1_3_5` | 세 값의 모집단 표준편차 (`ddof=0`) |
| `recent_gap_1` | prev1 성공률 - 누적 평활 투수 성공률 |
| `recent_gap_3` | prev3 성공률 - 누적 평활 투수 성공률 |
| `recent_gap_5` | prev5 성공률 - 누적 평활 투수 성공률 |

`recent_gap_*`가 양수면 최근 제구 성공률이 장기 평균보다 높고, 음수면 최근 성과가 장기 평균보다 낮다는 뜻입니다.

주의할 점은 prev1, prev3, prev5가 서로 독립된 세 경기가 아니라 각각 1·3·5경기 요약이라는 것입니다. 세 값을 다시 평균하면 가장 최근 경기가 여러 요약에 반복 반영됩니다. 이는 최근 경기에 자연스럽게 더 큰 비중을 주는 효과가 있지만 상관도도 높입니다.

### 최근 가운데 비율 3개

| 파생 변수 | 계산 |
|---|---|
| `recent_middle_mean_1_3_5` | prev1, prev3, prev5 가운데 비율 평균 |
| `recent_middle_range_1_3_5` | 세 값의 최댓값 - 최솟값 |
| `recent_middle_vs_cumulative` | 최근 가운데 비율 평균 - 누적 평활 가운데 비율 |

가운데 위험 코스는 제구 실패 정의 중 하나이므로 최근 가운데 비율의 상승은 실패 확률 증가 신호가 될 수 있습니다.

## 7. 결측 플래그 16개

아래 공식 비율이 결측인지 나타내는 `원본변수__missing` 피처를 각각 추가합니다.

- 투수 누적: 성공률, 요구 반대 방향, 가운데, 볼, 스트라이크
- 투수 최근 성공률: 이전 1·3·5경기
- 투수 최근 가운데 비율: 이전 1·3·5경기
- 타자 누적: 성공률, 가운데 비율
- 투수 구종 구성: 패스트볼, 브레이킹볼, 오프스피드

정확한 변수는 다음과 같습니다.

```text
asof_pitcher_success_rate__missing
asof_pitcher_reverse_rate__missing
asof_pitcher_middle_rate__missing
asof_pitcher_ball_rate__missing
asof_pitcher_strike_rate__missing
asof_pitcher_prev1_game_success_rate__missing
asof_pitcher_prev3_game_success_rate__missing
asof_pitcher_prev5_game_success_rate__missing
asof_pitcher_prev1_game_middle_rate__missing
asof_pitcher_prev3_game_middle_rate__missing
asof_pitcher_prev5_game_middle_rate__missing
asof_batter_success_rate__missing
asof_batter_middle_rate__missing
asof_pitcher_fastball_rate__missing
asof_pitcher_breaking_rate__missing
asof_pitcher_offspeed_rate__missing
```

결측 자체가 신규 선수, 기록 부족, 최근 경기 부족을 나타낼 수 있으므로 단순 대치와 별도로 정보로 사용합니다.

## 8. cold-start 피처 2개

| 파생 변수 | 계산 | 의미 |
|---|---|---|
| `is_pitcher_cold_start` | `asof_pitcher_n <= 0` | 공식 과거 투구 기록이 없는 투수 |
| `is_batter_cold_start` | `asof_batter_n <= 0` | 공식 과거 상대 기록이 없는 타자 |

신규 ID 여부가 아니라 공식 누적 표본 수를 기준으로 합니다. 따라서 기존 선수라도 제공된 이력이 0이면 cold-start로 분류됩니다.

## 9. 기타 수치 피처 4개

### `pitchmix_entropy`

```text
-(fastball_rate × log(fastball_rate)
 + breaking_rate × log(breaking_rate)
 + offspeed_rate × log(offspeed_rate))
```

구종을 고르게 섞을수록 값이 크고, 한 구종에 집중할수록 작습니다. `log(0)`을 피하려고 최소 `1e-12`로 자릅니다.

### 점수 차 파생 3개

| 파생 변수 | 계산 |
|---|---|
| `score_abs` | `abs(score_diff_pitcher_team)` |
| `is_pitcher_team_leading` | `score_diff_pitcher_team > 0` |
| `is_pitcher_team_trailing` | `score_diff_pitcher_team < 0` |

같은 3점 차라도 리드와 열세에서 투구 전략이 다를 수 있어 방향 플래그를 별도로 둡니다.

## 10. 모델별 최종 피처 차이

### CatBoost: 91개

- 47개 원본 + 49개 파생 - 중복 5개
- 14개 범주형을 문자열로 직접 전달
- 투수·타자 ID를 범주형으로 사용
- 학습에서 없던 범주는 CatBoost의 미관측 범주로 처리

### LightGBM: 96개

- 47개 원본 + 49개 파생
- 중복 5개도 유지
- 14개 범주형을 학습 mapping의 category code로 변환
- 미관측 범주는 `-1`
- `categorical_feature`로 지정해 범주 분할 사용

### XGBoost: 91개

- 47개 원본 + 49개 파생 - 중복 5개
- 14개 범주형을 학습 fold 출현 빈도로 변환
- 미관측 범주는 `0`
- 빈도가 같은 서로 다른 범주는 같은 값으로 표현될 수 있음

## 11. 제거되는 중복 피처 5개

CatBoost와 XGBoost에서는 다음을 제거합니다.

| 제거 피처 | 중복 이유 |
|---|---|
| `asof_pitcher_pitchmix_n` | `asof_pitcher_n`과 100% 동일 |
| `run_total_before` | `run_top_before + run_bot_before`로 복원 가능 |
| `num_runners_on` | 1·2·3루 주자 플래그 합으로 복원 가능 |
| `away_win_expectancy` | `home_win_expectancy`와 거의 완전 반대 |
| `asof_pitcher_offspeed_rate` | 세 구종 비율 합이 1이라 나머지 두 비율로 복원 가능 |

다만 `asof_pitcher_offspeed_rate_smoothed`와 결측 플래그는 별도 파생 피처로 남아 있습니다. 즉 원본 비율만 제거하고 평활된 정보는 유지합니다.

## 12. 피처 생성의 누수 방지 방식

- 현재 투구 위치, 판정, 결과, 실제 구종, 현재 Trackman 측정값은 사용하지 않습니다.
- 공식 `asof_*`는 해당 투구 직전까지 운영 측이 계산한 과거 정보입니다.
- 피처 생성은 현재 test 행과 학습에서 저장한 prior·mapping만 사용합니다.
- test 전체 `groupby`, `value_counts`, rolling, expanding, target encoding을 하지 않습니다.
- test 행 순서가 바뀌어도 같은 행의 피처와 예측은 동일합니다.
- 범주 mapping과 frequency encoding은 test가 아니라 학습 데이터에서 생성합니다.

## 13. 주요 피처와 모델별 의존성

### CatBoost 상위 신호

- `game_type`
- `season`
- `hand_matchup`
- `asof_pitcher_success_rate_smoothed`
- 양 팀 ID와 투수 ID

### LightGBM 상위 신호

- `pitcher_id`
- `batter_id`
- `season`
- `game_type`
- `hand_matchup`, `count_state`

### XGBoost 상위 신호

- 원본·평활 투수 누적 성공률
- `game_type`
- `season`
- 최근 성공률 평균

따라서 선수 ID, `season`, `game_type`, 누적 성공률과 평활 성공률이 `/1`의 핵심 축입니다.

## 14. 장점

1. 표본이 적은 선수의 비율을 평활해 극단값을 완화합니다.
2. 원본 비율과 평활 비율을 함께 제공해 트리 모델이 경험량에 따라 선택할 수 있습니다.
3. 최근 성과와 장기 누적 성과의 차이를 직접 표현합니다.
4. 결측과 cold-start를 값 대치에 묻지 않고 별도 신호로 남깁니다.
5. 카운트·주자·압박 상태의 상호작용을 범주로 제공합니다.
6. 모델마다 다른 범주 처리 방식으로 앙상블 다양성을 만듭니다.

## 15. 개선할 수 있는 부분

1. `alpha=50` 하나만 사용하므로 투수·타자·구종별 최적 평활 강도를 따로 탐색할 수 있습니다.
2. 원본 비율과 평활 비율을 동시에 넣어 중복성이 커질 수 있으므로 모델별 ablation이 필요합니다.
3. 최근 1·3·5경기 요약은 강하게 상관돼 추세 기울기나 최근성 가중 평균과 비교할 수 있습니다.
4. `close_game`, `late_inning`, `li` 임계값을 여러 값으로 검증할 수 있습니다.
5. `season`과 `game_type`의 영향이 지나치게 커 분포 이동에 민감할 수 있습니다.
6. LightGBM·XGBoost의 ID 인코딩은 신규 선수 대응이 약하므로 ID 유지·제거·빈도·평활 target encoding을 같은 fold에서 비교해야 합니다.
7. 팀·선수·손 조합의 교차 피처를 무작정 늘리지 말고 시간 검증으로 제한해야 합니다.
8. Trackman을 사용하지 않으므로 허용 범위가 확인되면 cutoff-safe 과거 요약을 별도 후보로 비교할 수 있습니다.

## 16. `/2` 개선에 주는 시사점

`/2`는 `/1`에서 중요했던 선수 ID, `season`, `game_type`, Platt 보정을 동시에 제거했습니다. 피처 효과를 정확히 확인하려면 `/1`을 기준으로 한 번에 하나만 바꾸는 ablation이 필요합니다.

권장 순서는 다음과 같습니다.

1. `/1` 피처와 모델을 같은 rolling fold에서 재현
2. ID만 제거·변경
3. `season`만 제거
4. `game_type`만 제거
5. 원본 비율과 평활 비율 조합 변경
6. alpha를 투수·타자·구종별로 탐색
7. 최근 추세 피처 조합 탐색
8. 보정과 앙상블을 마지막에 재선택

각 단계에서 `/1`보다 악화되면 해당 변경을 유지하지 않는 것이 핵심입니다.
