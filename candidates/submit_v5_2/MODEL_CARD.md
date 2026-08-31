# 모델 카드

## 모델 식별 정보

- 모델명: `submit v5-2`
- 서버 점수: `957.6800554874`
- 예측 대상: `control_success`
- 출력값: `[0, 1]` 범위의 제구 성공 확률

## 사용 모델

| 모델 | 사용 여부 | 역할 |
|---|---|---|
| CatBoost | 사용 | 주 모델. 최종 가중치 `0.8` |
| LightGBM | 사용 | 보조 모델. 최종 가중치 `0.2` |
| RandomForest | 미사용 | v5-2에는 포함하지 않음 |
| Transformer | 미사용 | v5-2에는 포함하지 않음 |
| TabM | 미사용 | v5-2에는 포함하지 않음 |

## 앙상블과 보정

최종 예측은 CatBoost와 LightGBM의 확률 예측을 고정 가중 평균한 뒤 Platt 보정을 적용한다.

```text
raw_pred = 0.8 * CatBoost + 0.2 * LightGBM
final_pred = Platt(raw_pred)
```

Platt 보정은 OOF 예측으로 학습했다.

## 저장 파일 설명

| 파일 | 설명 |
|---|---|
| `model/final_model.joblib` | CatBoost/LightGBM 모델, 기본 피처 상태, Trackman 상태를 포함한 최종 번들 |
| `model/calibration_model.joblib` | Platt 보정 모델 |
| `model/feature_schema.json` | 입력 컬럼, 최종 피처 컬럼, 선택한 Trackman 피처셋 정보 |
| `model/ensemble_config.json` | 앙상블 가중치, 보정 방식, Trackman 전략 |
| `script.py` | 평가 서버에서 실행되는 추론 코드 |
| `requirements.txt` | 추론에 필요한 패키지 목록 |

## 추론 시 동작

`script.py`는 모델 번들을 로드한 뒤 다음 순서로 예측한다.

1. test 행에서 v5 기본 파생피처를 만든다.
2. `model/final_model.joblib`에 저장된 `trackman_state`로 Trackman 피처를 만든다.
3. 각 모델이 요구하는 컬럼 순서대로 피처 행렬을 맞춘다.
4. CatBoost와 LightGBM의 확률 예측을 구한다.
5. `0.8`, `0.2` 가중 평균을 계산한다.
6. Platt 보정을 적용한다.
7. `output/submission.csv`를 저장한다.
