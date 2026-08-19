# TabM Feature Set v1

최종 TabM 전용 피처셋은 **Current-season Core + TrackMan exact Prev1**입니다.

- Numeric 42 / Binary 9 / Categorical 13 / Total 64
- Prev1: target season S에 대해 정확히 S-1 TrackMan만 사용
- exact S-1이 없으면 더 오래된 시즌으로 대체하지 않음
- calibration은 피처가 아니라 별도 모델 후처리

## 생성

```python
import pandas as pd
from features.tabm.tabm_features_final_v1 import save_tabm_final_v1

train = pd.read_csv("./open/data/train.csv")
test = pd.read_csv("./open/data/test.csv")
trackman = pd.read_csv("./open/data/trackman_history_with_train_id_clean.csv", low_memory=False)

save_tabm_final_v1(train, test, trackman)
```

## Validation

| Seed | Current Core | Prev1 | Delta |
|---:|---:|---:|---:|
| 42 | 0.248531 | 0.248389 | -0.000142 |
| 43 | 0.248852 | 0.248499 | -0.000353 |
| 44 | 0.248742 | 0.248588 | -0.000154 |

Prev1은 3/3 seed에서 개선, 평균 Delta Brier는 약 -0.000216.

P1 preprocessing은 3-seed 평균 효과가 사실상 0이었고 mixed BCE+Brier도 동률권이라 v1에서 제외했습니다.
F drift, bucket/interaction, ID, TrackMan change/stability/arsenal은 현재 구현에서 우선순위를 낮춘 후보이며 가설 자체를 폐기한 것은 아닙니다.

대용량 final train/test parquet은 GitHub 대신 팀 공유 Drive 사용을 권장합니다.
