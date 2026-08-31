# Trackman 피처 재사용 가이드

## 결론

5-2 서버 점수는 `957.6800554874`로 확인됐다. 따라서 5-2에서 쓴 Trackman 피처는 다른 모델에도 붙여볼 가치가 있다.

공통 재사용 모듈은 루트의 `trackman_features.py`다. 다른 버전 모델에서는 이 파일을 가져와서 Trackman 상태값을 만들고, 기존 피처 행렬에 붙이면 된다.

## 5-2에서 실제로 쓴 Trackman 설정

- 전략: `mapping_all_shrink`
- 피처셋: `5-2_server_957`
- 구성: 메타데이터 15개 + 물리 핵심 피처 10개 + 구종 구성 4개 = 총 29개
- 모델 반영 방식: `/5` no-id 기본 피처 뒤에 Trackman 29개를 붙임
- 범주형 추가 컬럼: `tm_mapping_confidence_bucket`

## 사용 코드

학습 코드에서 한 번 상태값을 만들고, train/valid/test 행에 같은 상태값을 적용한다.

```python
import pandas as pd
import trackman_features as tm

train = pd.read_csv("data/train.csv")
tm_state = tm.build_trackman_state(train)

# base_features는 기존 모델이 만들던 피처 행렬
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

검증 fold에서도 `tm_state`는 전체 train으로 만들 수 있다. Trackman 요약값 자체는 정답값을 쓰지 않고, 각 행의 `season`보다 이전 Trackman 과거 기록만 사용하도록 내부에서 이전 시즌 조회를 만든다. 모델의 정답값 완화, 보정, ID 빈도 같은 정답값/학습 통계는 기존처럼 fold 학습 분할에서만 학습해야 한다.

## 다른 모델에 붙이는 방법

1. 기존 모델의 `build_features()`는 그대로 둔다.
2. `tm_state = trackman_features.build_trackman_state(train)`를 학습 시작부에 만든다.
3. 각 fold에서 기본 피처를 만든 뒤 `append_trackman_features(base, raw_rows, tm_state)`를 호출한다.
4. CatBoost/Transformer 계열은 `tm_mapping_confidence_bucket`을 범주형 피처로 추가한다.
5. LightGBM/XGBoost/TabM 수치 모델은 `tm_mapping_confidence_bucket`을 정수 인코딩하거나 원-핫 인코딩한다.
6. 최종 제출 번들에는 `tm_state`, `trackman_strategy`, `trackman_feature_set`을 같이 저장한다.
7. `script.py`에서도 같은 `trackman_features.py` 로직 또는 동등한 함수를 포함해서 test 피처를 만들어야 한다.

## 모델별 적용 우선순위

| 모델 | 적용 방식 | 우선순위 |
|---|---|---:|
| CatBoost | 기본 피처 + 29개 Trackman, bucket 범주형 처리 | 높음 |
| LightGBM | 기본 피처 + 29개 Trackman, bucket 정수 인코딩 | 높음 |
| XGBoost | 수치형 Trackman 위주, bucket 정수 인코딩 | 중간 |
| Transformer | lowfreq ID 임베딩 피처 + Trackman 수치/메타데이터 | 중간 |
| TabM | 수치형 Trackman 위주, 범주형은 인코딩 후 사용 | 중간 |

## 피처셋 후보

| 피처셋 | 구성 | 용도 |
|---|---|---|
| `metadata_only` | 매핑/이력 여부와 신뢰도 15개 | 가장 안전한 후보 |
| `metadata_pitchmix` | 메타데이터 + 구종 구성 4개 | 구종 성향 반영 |
| `metadata_physical_core` | 메타데이터 + 구속/회전/릴리스/익스텐션 10개 | 물리 특성 반영 |
| `metadata_physical_pitchmix` | 메타데이터 + 물리 특성 + 구종 구성 29개 | 5-2 검증/서버 제출 후보 |
| `5-2_server_957` | `metadata_physical_pitchmix`와 동일 | 서버 점수 957.6800554874 기준 재사용 |

## 주의

- `trackman_history.csv`의 현재 투구 단위 측정값을 test 행에 직접 붙이는 방식이 아니다.
- 각 행의 `season` 기준 이전 시즌 Trackman 과거 기록만 요약값으로 사용한다.
- `pitcher_id_mapping_clean.csv`는 pitcher_id와 Trackman pitcher id를 연결하는 매핑으로 사용한다.
- 새로운 투수가 들어오면 매핑/이력이 없을 수 있으므로 투수 손잡이/시즌 사전 평균과 `tm_is_rookie_or_no_history` 플래그로 처리한다.
- 서버 제출용 `script.py`에는 `trackman_features.py`를 별도 파일로 넣을 수 없으면, 해당 함수 내용을 script 안에 포함하거나 모델 번들에 필요한 상태값을 저장해야 한다.
