# TabM Train Feature Set v1

최종 후보: **B_Lean**

## 최종 입력 구조

- Numeric: 32
- Binary: 8
- Categorical: 13
- Total: **53**

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

## 제거

Recent Middle raw 3개:
- `asof_pitcher_prev1_game_middle_rate`
- `asof_pitcher_prev3_game_middle_rate`
- `asof_pitcher_prev5_game_middle_rate`

Recent Delta / Current Relative 계열도 최종 v1에는 포함하지 않는다.

## 주요 실험 결과

### Current Season block ablation, seed 42

- Baseline: 0.249097
- Level: 0.248635
- Reliability: 0.248874
- Relative: 0.248717
- Level + Relative: 0.248579
- Reliability + Relative: 0.248533
- Full: 0.248537
- **Level + Reliability: 0.248446**

### Drop ablation, seed 42

- Drop Recent Middle: 0.248449
- Drop Career Reliability: 0.248465
- Drop Pitcher Career Detail: 0.248497
- Drop WinExp + LI: 0.248509
- Drop Pitch Mix: 0.248518
- Drop Recent Success: 0.248540
- Drop Batter History: 0.249017

### Typing ablation, seed 42

- **Current typing: 0.248446**
- Count -> Numeric: 0.248451
- Game Progress -> Numeric: 0.248468
- Count + Game -> Numeric: 0.248517

### Multi-seed final comparison, seeds 42/52/62

- **B_Lean: Mean 0.248562 / Std 0.000112 / Max 0.248674**
- A_Champion: Mean 0.248593 / Std 0.000135 / Max 0.248711
- C_LeanPlus: Mean 0.248594 / Std 0.000096 / Max 0.248681

## 해석

현재 실험에서는 TabM에 **level + reliability** 정보를 직접 주는 것이 효과적이었고,
명시적인 relative/delta 파생은 추가 가치가 제한적이었다.

Batter history와 recent success raw level은 유지 가치가 높았고,
low-cardinality 상황 변수는 기존 categorical typing을 유지하는 편이 좋았다.

## 사용

```python
import pandas as pd
from tabm_train_features_v1 import (
    build_tabm_train_v1,
    TABM_TRAIN_V1_NUMERIC,
    TABM_TRAIN_V1_BINARY,
    TABM_TRAIN_V1_CATEGORICAL,
)

train = pd.read_csv("train.csv")
tabm_train = build_tabm_train_v1(train)
```

이후 기존 TabM preprocessing에 위 numeric / binary / categorical 리스트를 그대로 사용한다.

## 주의

이 v1은 **공식 train.csv 기반 train-only 피처셋 선정 결과**다.
`test.csv` 적용 시 current-season 집계가 동일하게 시점 안전한지 별도 검증이 필요하다.
TrackMan은 v1에 포함하지 않는다.
