# 규칙 준수 체크리스트

## 체크리스트

- [x] 공식 제공 데이터만 사용했다.
- [x] 외부 데이터를 사용하지 않았다.
- [x] 원격 API를 호출하지 않는다.
- [x] 추론 중 외부 다운로드를 수행하지 않는다.
- [x] 외부 사전학습 모델이나 가중치를 사용하지 않았다.
- [x] `test.csv`의 다른 행을 이용해 test 집계를 만들지 않는다.
- [x] 현재 투구 이후 정보를 사용하지 않는다.
- [x] 현재 투구의 실제 위치나 코스를 사용하지 않는다.
- [x] 현재 투구의 실제 판정이나 결과를 사용하지 않는다.
- [x] 현재 투구의 `control_success`를 사용하지 않는다.
- [x] 현재 투구의 실제 구종을 사용하지 않는다.
- [x] 현재 투구 단위 Trackman 측정값을 사용하지 않는다.
- [x] 2025년 Trackman 데이터를 사용하지 않는다.
- [x] Trackman은 이전 시즌 투수 이력 요약으로만 사용한다.
- [x] 추론용 Trackman 상태는 모델 번들 안에 저장한다.
- [x] `submit.zip`은 코드 제출 형식의 구조를 지킨다.
- [x] 원본 `train.csv`, `test.csv`, `trackman_history.csv`를 Git에 커밋하지 않았다.

## `submit.zip` 구조

```text
model/
model/final_model.joblib
model/calibration_model.joblib
model/feature_schema.json
model/ensemble_config.json
script.py
requirements.txt
```

## 추가 설명

`pitcher_id_mapping_clean.csv`는 작은 파생 매핑 파일로 포함했다. 반면 원본 공식 대용량 파일인 `trackman_history.csv`는 Git에서 제외했으며, 재학습할 때 별도로 공식 데이터 위치에 배치해야 한다.
