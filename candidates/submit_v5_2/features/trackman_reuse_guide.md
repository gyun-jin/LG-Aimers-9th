# Trackman 피처 재사용 가이드

## 결론

5-2 서버 점수는 `957.6800554874`로 확인됐다. 따라서 5-2에서 쓴 Trackman 피처는 다른 모델에도 붙여볼 가치가 있다.

공통 재사용 모듈은 루트의 `trackman_features.py`다. 다른 버전 모델에서는 이 파일을 import해서 Trackman state를 만들고, 기존 feature matrix에 concat하면 된다.

## 5-2에서 실제로 쓴 Trackman 설정

- 전략: `mapping_all_shrink`
- 피처셋: `5-2_server_957`
- 구성: metadata 15개 + physical core 10개 + pitch mix 4개 = 총 29개
- 모델 반영 방식: `/5` no-id 기본 피처 뒤에 Trackman 29개를 붙임
- categorical 추가: `tm_mapping_confidence_bucket`

## 사용 코드

학습 코드에서 한 번 state를 만들고, train/valid/test row에 같은 state를 적용한다.

```python
import pandas as pd
import trackman_features as tm

train = pd.read_csv("data/train.csv")
tm_state = tm.build_trackman_state(train)

# base_features는 기존 모델이 만들던 feature matrix
base_features = build_my_base_features(train)
features = tm.append_trackman_features(
    base_features,
    train,
    tm_state,
    strategy="mapping_all_shrink",
    feature_set="5-2_server_957",
)

cat_cols = existing_cat_cols + tm.categorical_columns(list(features.columns))
```

검증 fold에서도 `tm_state`는 전체 train으로 만들 수 있다. Trackman summary 자체는 target을 쓰지 않고, 각 행의 `season`보다 이전 Trackman history만 사용하도록 내부에서 prior-season lookup을 만든다. 모델의 target smoothing, calibration, ID frequency 같은 target/학습통계는 기존처럼 fold train split에서만 fit해야 한다.

## 다른 모델에 붙이는 방법

1. 기존 모델의 `build_features()`는 그대로 둔다.
2. `tm_state = trackman_features.build_trackman_state(train)`를 학습 시작부에 만든다.
3. 각 fold에서 base feature를 만든 뒤 `append_trackman_features(base, raw_rows, tm_state)`를 호출한다.
4. CatBoost/Transformer 계열은 `tm_mapping_confidence_bucket`을 categorical로 추가한다.
5. LightGBM/XGBoost/TabM 수치 모델은 `tm_mapping_confidence_bucket`을 label encoding하거나 one-hot 처리한다.
6. 최종 제출 bundle에는 `tm_state`, `trackman_strategy`, `trackman_feature_set`을 같이 저장한다.
7. `script.py`에서도 같은 `trackman_features.py` 로직 또는 동등한 함수를 포함해서 test feature를 만들어야 한다.

## 모델별 적용 우선순위

| 모델 | 적용 방식 | 우선순위 |
|---|---|---:|
| CatBoost | base 피처 + 29개 Trackman, bucket categorical | 높음 |
| LightGBM | base 피처 + 29개 Trackman, bucket label encoding | 높음 |
| XGBoost | 수치형 Trackman 위주, bucket label encoding | 중간 |
| Transformer | lowfreq ID embedding 피처 + Trackman 수치/metadata | 중간 |
| TabM | 수치형 Trackman 위주, categorical은 인코딩 후 사용 | 중간 |

## 피처셋 후보

| feature_set | 구성 | 용도 |
|---|---|---|
| `metadata_only` | mapping/history 여부와 신뢰도 15개 | 가장 안전한 후보 |
| `metadata_pitchmix` | metadata + 구종 mix 4개 | 구종 성향 반영 |
| `metadata_physical_core` | metadata + 구속/회전/릴리스/extension 10개 | 물리 특성 반영 |
| `metadata_physical_pitchmix` | metadata + physical + pitchmix 29개 | 5-2 검증/서버 제출 후보 |
| `5-2_server_957` | `metadata_physical_pitchmix`와 동일 | 서버 점수 957.6800554874 기준 재사용 |

## 주의

- `trackman_history.csv`의 현재 투구 단위 측정값을 test row에 직접 붙이는 방식이 아니다.
- 각 행의 `season` 기준 이전 시즌 Trackman history만 summary로 사용한다.
- `pitcher_id_mapping_clean.csv`는 pitcher_id와 Trackman pitcher id를 연결하는 매핑으로 사용한다.
- 새로운 투수가 들어오면 mapping/history가 없을 수 있으므로 hand/season prior와 `tm_is_rookie_or_no_history` 플래그로 처리한다.
- 서버 제출용 `script.py`에는 `trackman_features.py`를 별도 파일로 넣을 수 없으면, 해당 함수 내용을 script 안에 포함하거나 model bundle에 필요한 state를 저장해야 한다.
