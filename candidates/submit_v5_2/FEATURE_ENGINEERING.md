# 피처 엔지니어링

## 원본 피처

공식 test 입력 구조의 컬럼을 기준으로 피처를 만든다. 주요 원본 피처군은 다음과 같다.

- 시즌, 월, 요일, 이닝
- 초/말, 경기 타입
- 볼카운트, 아웃카운트
- 점수, 점수 차, 승리 기대값, LI
- 주자/베이스 상태
- 투수/타자 ID, 팀 ID, 손잡이
- `asof_*` 누적 이력

## `asof_*` 피처

`asof_*` 컬럼은 현재 투구 이전에 이미 관측된 누적/최근 이력이다.

사용한 주요 정보:

- 투수 누적 제구 성공률, reverse/middle/ball/strike 비율
- 투수 최근 1/3/5경기 제구 성공률과 middle 비율
- 타자 누적 상대 성공률과 middle 비율
- 투수 구종 비율

비율 계열은 학습 데이터에서 계산한 사전 평균으로 완화한다.

```text
smoothed_rate = (count * observed_rate + alpha * prior) / (count + alpha)
```

## count 관련 피처

- `count_state`
- `balls_before`
- `strikes_before`
- 최근 성공률과 누적 성공률의 차이

볼카운트는 제구 성공 확률에 큰 영향을 줄 수 있다. 타자에게 유리한 카운트에서는 존 안에 넣어야 하는 압박이 커지고, 투수에게 유리한 카운트에서는 유인구나 코스 선택이 달라질 수 있다.

## runner/base 상황 피처

- `base_state`
- `runner_out_state`
- 주자 존재 여부
- 득점권 상황
- 점수 차와 LI

주자 상황과 경기 압박은 투수의 투구 선택과 제구 안정성에 영향을 줄 수 있다.

## hand match 피처

- `pitcher_hand`
- `batter_hand`
- `hand_matchup`

투수/타자 손잡이 조합을 통해 matchup 효과를 반영한다.

## `game_type` 피처

`game_type`은 범주형 입력으로 유지했다. 경기 타입에 따라 투수 운용, 압박 상황, 투구 선택이 달라질 가능성을 반영하기 위한 피처다.

## pitcher/batter/team 관련 피처

최종 v5-2 피처 행렬에서는 원본 `pitcher_id`, `batter_id`를 제거했다. 대신 다음 정보로 투수/타자 맥락을 반영한다.

- 공식 `asof_*` 투수/타자 이력
- 팀 ID
- Trackman 이전 시즌 투수 요약

즉, 원본 ID를 직접 외우는 구조는 줄이고, 과거 성과와 투구 특성 중심으로 예측하게 했다.

## Trackman 5-2 피처 29개

최종 사용한 Trackman 피처 수는 `29`개다.

### 매핑/이력 메타데이터

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

### 물리 특성 핵심 피처

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

### 구종 구성

- `tm_hist_pitch_group_fastball_shrunk`
- `tm_hist_pitch_group_breaking_shrunk`
- `tm_hist_pitch_group_offspeed_shrunk`
- `tm_hist_pitch_type_entropy`

## `tm_mapping_confidence_bucket`

`tm_mapping_confidence_bucket`은 Trackman 매핑 신뢰도를 구간화한 범주형 피처다. 모델은 이 피처를 통해 Trackman 요약값을 강하게 믿어도 되는지, 아니면 사전 평균에 가까운 약한 정보로 봐야 하는지 구분할 수 있다.

## 예측 관점의 의미

Trackman 물리 피처는 현재 투구의 결과를 직접 알려주지 않는다. 대신 투수의 과거 릴리스, 구속, 회전수, 익스텐션, 구종 구성 같은 투구 특성을 제공한다. 같은 경기 상황과 같은 카운트라도 투수의 과거 물리 특성이 다르면 제구 성공 확률이 달라질 수 있으므로, 이 정보가 v5-2의 서버 점수 개선에 기여한 것으로 본다.
