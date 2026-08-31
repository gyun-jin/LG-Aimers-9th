# Trackman 사용 방식

## 사용 파일

- `pitcher_id_mapping_clean.csv`: 공식 `pitcher_id`와 `pitcher_trackman_id`를 연결하는 매핑 파일
- `trackman_history.csv`: 공식 과거 Trackman 데이터. 대용량 원본 파일이라 이 Git 패키지에는 포함하지 않음
- `trackman_features.py`: Trackman 피처 생성 공통 wrapper

## 매핑 방식

Trackman 피처 생성기는 다음 순서로 투수를 연결한다.

```text
train/test pitcher_id -> pitcher_id_mapping_clean.csv -> pitcher_trackman_id
```

이후 `pitcher_trackman_id`로 과거 Trackman 기록을 조회한다.

## 전략

- strategy: `mapping_all_shrink`
- feature_set: `5-2_server_957`
- Trackman 피처 수: `29`

`mapping_all_shrink`는 가능한 매핑을 모두 사용하되, 이력이 적은 투수의 요약값은 손잡이/시즌 prior 쪽으로 shrink한다. 또한 모델이 Trackman 정보의 신뢰도를 판단할 수 있도록 매핑 신뢰도와 이력 여부 metadata를 함께 제공한다.

## 시간 기준

각 row에 대해 해당 row의 `season`보다 이전 시즌 Trackman 기록만 요약한다.

예시:

- 2023 row는 2019-2022 Trackman 이력만 사용
- 2024 row는 2019-2023 Trackman 이력만 사용
- 2025 test row는 2019-2024 Trackman 이력만 사용

## 사용하지 않은 정보

- 현재 투구의 Trackman 측정값
- 2025년 Trackman 데이터
- 현재 투구 실제 위치
- 현재 투구 판정 결과
- 현재 투구 실제 구종
- 현재 train/test pitch row와 Trackman row의 현재 투구 단위 1:1 직접 결합

## 제출 시 동작

제출 스크립트는 원본 `trackman_history.csv`를 다시 읽지 않는다. 학습 과정에서 `tm_state`를 만들고, 이 상태를 `model/final_model.joblib` 안에 저장한다. 추론 시 `script.py`는 저장된 `tm_state`를 사용해 test row의 Trackman 피처를 재생성한다.
