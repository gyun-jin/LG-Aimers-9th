# 2023/2024 Brier 개선 분석

## 현재 문제 요약

`/5-3`은 `/5-2`보다 전체 mean Brier는 개선됐지만, 연도별로 보면 2023과 2024가 여전히 어렵다.

| validation season | Brier | AUC | mean_pred | mean_target |
|---:|---:|---:|---:|---:|
| 2022 | 0.24309218 | 0.581083 | 0.525743 | 0.528921 |
| 2023 | 0.24982374 | 0.533098 | 0.499517 | 0.499937 |
| 2024 | 0.24760468 | 0.552749 | 0.489524 | 0.486112 |

가장 큰 문제는 2023이다. 2023은 Brier가 `0.24982374`로 거의 baseline에 가깝고, AUC도 `0.533098`이라 성공/실패 구분력이 약하다.

## 연도별 target 분포 변화

정제 train 기준 실제 제구 성공률은 다음처럼 내려간다.

| season | rows | control_success mean | baseline Brier |
|---:|---:|---:|---:|
| 2019 | 237356 | 0.564633 | 0.24582259 |
| 2020 | 244070 | 0.532704 | 0.24893047 |
| 2021 | 247082 | 0.532762 | 0.24892662 |
| 2022 | 247470 | 0.528921 | 0.24916359 |
| 2023 | 245505 | 0.499937 | 0.25000000 |
| 2024 | 253493 | 0.486112 | 0.24980712 |

2023은 실제 성공률이 거의 `0.5`라 baseline Brier가 이론상 최대에 가까운 `0.25`다. 즉, 평균만 찍는 모델도 Brier가 높게 나오고, 모델이 강한 구분 신호를 잡지 못하면 Brier를 낮추기 어렵다.

## baseline 대비 개선폭

| season | baseline Brier | model Brier | 개선폭 |
|---:|---:|---:|---:|
| 2022 | 0.24916359 | 0.24309218 | -0.00607141 |
| 2023 | 0.25000000 | 0.24982374 | -0.00017626 |
| 2024 | 0.24980712 | 0.24760468 | -0.00220244 |

2022는 baseline 대비 많이 개선된다. 반면 2023은 거의 개선되지 않는다. 따라서 단순히 calibration 문제라기보다, 2023에서 모델이 성공/실패를 구분할 피처 신호를 충분히 못 잡고 있다고 보는 게 맞다.

## 왜 2023/2024가 어려운가

### 1. 시즌 drift

2019~2022까지는 제구 성공률이 대체로 `0.53~0.56` 수준이었는데, 2023은 `0.4999`, 2024는 `0.4861`까지 내려간다.

기존 누적 피처인 `asof_pitcher_success_rate`는 과거 시즌 성과가 많이 섞인다. 그래서 특정 시즌에 리그 전체 성공률이나 판정/운용 패턴이 바뀌면 모델이 늦게 따라갈 수 있다.

### 2. 2023의 구분력 저하

2023 AUC는 `0.533098`이다. 2022의 `0.581083`보다 크게 낮다.

이는 2023에서 기존 피처들이 성공/실패를 잘 가르지 못했다는 의미다. 즉, 예측 평균은 실제 평균에 거의 맞췄지만, 개별 row별 ranking이 약해서 Brier가 거의 baseline에 머물렀다.

### 3. 전체 평균 중심 학습

현재 모델은 2019~2024 전체 데이터를 사용한다. 오래된 시즌도 많이 들어가므로, 2023/2024 drift를 강하게 반영하기 어렵다.

2023/2024를 더 잘 맞추려면 최근 시즌에 더 큰 학습 가중치를 주거나, 모델 선택 기준을 2023/2024 중심으로 바꿔야 한다.

### 4. Trackman 효과가 연도별로 다를 가능성

quick ablation에서는 현재 시즌 피처를 포함한 no-Trackman 후보가 가장 좋았다.

| 후보 | quick mean Brier | quick worst Brier |
|---|---:|---:|
| no-Trackman | 0.24696972 | 0.24973173 |
| Trackman metadata only | 0.24700277 | 0.24982940 |
| Trackman pitchmix | 0.24698750 | 0.24981527 |
| Trackman physical core | 0.24698188 | 0.24980606 |
| Trackman full | 0.24698039 | 0.24985762 |

Trackman full은 서버에서 좋았던 `/5-2` 구조를 유지하기 위해 최종 모델에 넣었지만, 2023/2024 local 기준에서는 항상 유리하지 않을 수 있다.

## 개선 방향

## 1. 최근 시즌 sample weight 적용

2023/2024를 더 잘 맞추려면 최근 시즌 row에 더 큰 weight를 주는 실험이 가장 우선이다.

예시 weight:

| season | sample weight |
|---:|---:|
| 2019 | 0.7 |
| 2020 | 0.8 |
| 2021 | 0.9 |
| 2022 | 1.0 |
| 2023 | 1.3 |
| 2024 | 1.5 |

기대 효과:

- 2023/2024 target drift를 더 빠르게 반영
- 2023/2024 Brier 개선 가능
- 2022 Brier는 약간 나빠질 수 있음

판단 기준은 단순 mean Brier가 아니라 2023/2024 Brier와 worst Brier를 같이 봐야 한다.

## 2. validation 선택 기준 변경

현재처럼 전체 mean Brier만 보면 2022를 잘 맞추는 모델이 유리할 수 있다.

추천 선택 기준:

```text
score = mean_brier + 0.5 * worst_brier
```

또는:

```text
score = 0.25 * brier_2022 + 0.35 * brier_2023 + 0.40 * brier_2024
```

목표가 서버 점수 개선이라면 최근 시즌과 worst fold를 더 강하게 보는 쪽이 낫다.

## 3. CatBoost 비중 증가

quick component 비교에서 LightGBM 단독은 약했다.

| model | quick mean Brier | quick worst Brier |
|---|---:|---:|
| CatBoost | 0.24708633 | 0.24977960 |
| LightGBM | 0.24855746 | 0.25267830 |
| Ensemble 0.8/0.2 | 0.24708110 | 0.24977163 |

LightGBM은 보조 역할만 하는 게 맞아 보인다. 다음 실험에서는 다음 weight를 비교하는 게 좋다.

```text
CatBoost 1.0
CatBoost 0.95 + LightGBM 0.05
CatBoost 0.9 + LightGBM 0.1
CatBoost 0.8 + LightGBM 0.2
```

특히 2023/2024 기준으로 LightGBM 비중이 낮을수록 좋은지 확인해야 한다.

## 4. Trackman 피처셋 재선택

Trackman full이 항상 좋은 것은 아니다. 다음 조합을 full validation으로 다시 비교할 필요가 있다.

```text
no-Trackman
Trackman metadata only
Trackman metadata + pitchmix
Trackman metadata + physical core
Trackman full
```

추가로 모델별로 다르게 적용할 수도 있다.

```text
CatBoost: Trackman 사용
LightGBM: Trackman 미사용
```

또는:

```text
CatBoost: metadata + physical
LightGBM: metadata only
```

## 5. season-aware calibration

현재는 전체 OOF 기반 Platt 보정 하나를 사용한다. 하지만 연도별 target prior가 크게 다르다.

가능한 개선:

- season별 prior correction
- 최근 시즌 OOF만 이용한 calibration
- 2023/2024 validation 성능을 기준으로 calibration 선택

주의할 점:

- test의 `season`은 row 자체 값이므로 사용 가능하다.
- test 전체 분포를 보고 보정하면 안 된다.
- train에서 저장한 season별 상수표만 사용해야 한다.

## 6. 현재 시즌 피처 강화

`/5-3`에서 현재 시즌 성공률 복원 피처를 추가했지만, 더 강화할 수 있다.

추가 후보:

- `pitcher_current_season_success_rate_smoothed - league_season_prior`
- `pitcher_current_season_success_rate_smoothed / asof_pitcher_success_rate_smoothed`
- `pitcher_current_season_n` 구간화
- `current_season_rate_bin × pitcher_hand`
- `current_season_rate_bin × game_type × count_state`
- `current_season_rate_bin × pressure_state`

이 피처들은 2023/2024처럼 현재 시즌 성향이 과거 누적과 달라지는 상황에서 도움될 수 있다.

## 추천 실험 순서

1. `5-4`: 최근 시즌 sample weight 적용
2. CatBoost/LightGBM ensemble weight 재탐색
3. no-Trackman vs Trackman 피처셋 full validation 비교
4. 2023/2024 가중 validation score로 최종 후보 선택
5. season-aware calibration 적용
6. 현재 시즌 피처 interaction 확장

## 가장 유력한 해결책

가장 먼저 시도할 조합은 다음이다.

```text
hand clean train
+ current season features
+ recent season sample weight
+ CatBoost 비중 0.9 이상
+ Trackman 피처셋 재선택
+ 2023/2024 weighted validation 기준
```

이 방향이 2023/2024 Brier를 낮출 가능성이 가장 높다.
