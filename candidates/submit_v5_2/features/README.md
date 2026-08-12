# train_trackman

새 모델 학습에서 5-2 방식의 Trackman 피처를 붙일 때 필요한 파일 묶음이다.

## 포함 파일

- `pitcher_id_mapping_clean.csv`
  - `train.csv`의 `pitcher_id`와 `trackman_history.csv`의 `pitcher_trackman_id`를 연결하는 매핑 파일.
- `trackman_history.csv`
  - 2019-2024 Trackman 과거 투구 기록.
- `trackman_features.py`
  - 매핑과 Trackman history를 읽어 prior-season 투수 요약 피처를 만드는 공통 모듈.
- `trackman_reuse_guide.md`
  - 다른 모델에 Trackman 피처를 붙이는 사용 가이드.

## 사용 방식

학습 코드에서 루트의 `trackman_features.py`를 import하거나, 이 폴더의 복사본을 import해서 쓴다.

```python
import pandas as pd
import trackman_features as tm

train = pd.read_csv("data/train.csv")
tm_state = tm.build_trackman_state(train, data_dir="train_trackman")

base_features = build_my_base_features(train)
features = tm.append_trackman_features(
    base_features,
    train,
    tm_state,
    strategy="mapping_all_shrink",
    feature_set="5-2_server_957",
)
```

## 5-2 서버 점수 설정

- 서버 점수: `957.6800554874`
- 전략: `mapping_all_shrink`
- feature_set: `5-2_server_957`
- Trackman 피처 수: `29`
- categorical 추가 컬럼: `tm_mapping_confidence_bucket`

## 주의

- 학습 때는 `pitcher_id_mapping_clean.csv`와 `trackman_history.csv`가 필요하다.
- 제출 때는 원본 CSV를 다시 읽지 말고, 학습 때 만든 `tm_state`를 model bundle에 저장하는 방식이 안전하다.
- Trackman 피처는 각 row의 `season`보다 이전 시즌 기록만 요약해서 붙인다.
