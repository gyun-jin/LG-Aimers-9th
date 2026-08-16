# TabM Feature Set v1 — Train + Test

최종 후보: **B_Lean**

## 최종 입력 구조

- Numeric: 32
- Binary: 8
- Categorical: 13
- Total: **53**

## 지원 범위

- 공식 `train.csv`
- 배포용 5행 `test.csv`
- 평가 서버에서 동일 경로/동일 스키마로 교체되는 비공개 2025 test

## 핵심 추가

Current Season의 **Level + Reliability** 8개를 추가한다.

Level:
- `pitcher_current_season_success_rate`
- `pitcher_current_season_success_rate_smoothed`

Reliability:
- `pitcher_current_season_n`
- `pitcher_current_season_success_count`
- `log1p_pitcher_current_season_n`
- `pitcher_current_season_n_ratio_to_career`
- `pitcher_current_season_available_flag`
- `pitcher_current_season_small_sample_flag`

## 최종 제거

Recent Middle raw 3개:
- `asof_pitcher_prev1_game_middle_rate`
- `asof_pitcher_prev3_game_middle_rate`
- `asof_pitcher_prev5_game_middle_rate`

Recent Delta / Current Relative 계열은 최종 v1에 포함하지 않는다.

## 주요 검증 결과

### Current Season block ablation — seed 42

- Baseline: 0.249097
- Level: 0.248635
- Reliability: 0.248874
- Relative: 0.248717
- Level + Relative: 0.248579
- Reliability + Relative: 0.248533
- Full: 0.248537
- **Level + Reliability: 0.248446**

### Multi-seed final comparison — seeds 42 / 52 / 62

- **B_Lean: Mean Brier 0.248562 / Std 0.000112**
- A_Champion: Mean Brier 0.248593 / Std 0.000135
- C_LeanPlus: Mean Brier 0.248594 / Std 0.000096

## Test current-season 처리

배포용 `test.csv`에는 형식 확인용 5행만 포함되고, 실제 평가는 서버에서
245,789행의 2025 test로 교체된다.

따라서 2025 current-season 기준점을 test 내부 최소값으로 추정하지 않는다.

- 공개 train에 존재하는 투수:
  - 공개 train의 마지막 투구 직후 누적 상태를 2025 시즌 시작 anchor로 사용
- 공개 train에 없는 투수:
  - 2025 신규 투수로 처리하여 season-start anchor를 0으로 설정
  - test의 `asof_pitcher_n`을 2025 current-season 누적 투구 수로 사용

5행 샘플 검증 결과:
- train final feature matrix: `(1475092, 53)`
- test final feature matrix: `(5, 53)`
- 신규 2025 투수의 current-season 누적값 처리 확인 완료

## 사용

```python
import pandas as pd

from features.tabm.tabm_features_v1 import (
    build_tabm_train_v1,
    build_tabm_test_v1,
    TABM_V1_NUMERIC,
    TABM_V1_BINARY,
    TABM_V1_CATEGORICAL,
    TABM_V1_FEATURES,
)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

tabm_train = build_tabm_train_v1(train)
tabm_test = build_tabm_test_v1(train, test)

X_train = tabm_train[TABM_V1_FEATURES]
X_test = tabm_test[TABM_V1_FEATURES]
```

이후 TabM preprocessing에서 numeric / binary / categorical 리스트를 각각 사용한다.

## 주의
- 이 v1에는 TrackMan feature가 포함되지 않았다.
- validation은 train에서 `season < 2024` 학습, `season == 2024` 검증 기준으로 선정했다.
