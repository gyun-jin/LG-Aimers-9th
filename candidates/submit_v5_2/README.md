# Submit v5-2 Trackman 모델

## 요약

- 버전명: `submit v5-2`
- 실제 서버 리더보드 점수: `957.6800554874`
- 대회 목표: 각 투구의 `control_success = 1` 확률 예측
- 최종 모델: CatBoost + LightGBM 가중 앙상블
- 최종 가중치: CatBoost `0.8`, LightGBM `0.2`
- 보정 방식: Platt 보정
- 주요 개선점: `mapping_all_shrink` 방식의 이전 시즌 Trackman 투수 요약 피처 추가

## 폴더 구조

```text
candidates/submit_v5_2/
├── model/
├── script.py
├── requirements.txt
├── submit.zip
├── train.py
├── full_validation.py
├── build_submit.py
├── features/
├── notebooks/
├── output/
├── reports/
├── README.md
├── MODEL_CARD.md
├── FEATURE_ENGINEERING.md
├── TRACKMAN_USAGE.md
├── PREPROCESSING.md
├── TRAINING_AND_INFERENCE.md
├── RULE_COMPLIANCE.md
└── SCORE_REPORT.md
```

## 모델 구조

v5-2는 v5의 no-id tabular 피처 파이프라인을 유지하고, 여기에 Trackman 파생 피처 29개를 추가한 모델이다.

최종 예측 확률 계산 방식은 다음과 같다.

```text
raw_pred = 0.8 * CatBoost_proba + 0.2 * LightGBM_proba
final_pred = Platt(raw_pred)
```

최종 피처 행렬은 총 `118`개 컬럼을 사용한다.

## 포함 데이터와 제외 데이터

Git에 포함한 파일:

- `features/pitcher_id_mapping_clean.csv`
- `features/trackman_features.py`
- `features/README.md`
- `features/trackman_reuse_guide.md`

Git에 포함하지 않은 파일:

- `train.csv`
- `test.csv`
- `trackman_history.csv`

`trackman_history.csv`는 공식 제공 대용량 원본 데이터이므로 Git에 포함하지 않았다. v5-2를 재학습하려면 공식 `trackman_history.csv`를 `data/` 또는 `train_trackman/` 아래에 배치해야 한다.

제출용 `submit.zip`은 추론 시 원본 Trackman CSV를 다시 읽지 않는다. 학습 시 생성한 Trackman 요약 상태인 `tm_state`가 `model/final_model.joblib` 안에 저장되어 있고, `script.py`는 이 저장된 상태를 사용해 test row의 Trackman 피처를 재구성한다.

## Trackman 사용 방식

Trackman 데이터는 현재 투구 단위 측정값으로 직접 붙인 것이 아니라, 이전 시즌 투수 이력 요약으로 사용했다.

처리 흐름:

1. `train.csv` 또는 `test.csv`의 `pitcher_id`를 확인한다.
2. `pitcher_id_mapping_clean.csv`로 `pitcher_trackman_id`를 찾는다.
3. `trackman_history.csv`에서 해당 `pitcher_trackman_id`의 과거 기록을 찾는다.
4. 각 행의 `season`보다 이전 시즌 Trackman 기록만 요약한다.
5. 구속, 회전수, 릴리스 위치, 익스텐션, 구종 비율, 매핑 신뢰도 등을 피처로 붙인다.

최종 Trackman 설정:

- 전략: `mapping_all_shrink`
- 피처셋: `5-2_server_957`
- Trackman 피처 수: `29`
- 범주형 Trackman 피처: `tm_mapping_confidence_bucket`

사용하지 않은 정보:

- 현재 투구의 Trackman 측정값
- 2025년 Trackman 데이터
- 현재 투구 실제 위치
- 현재 투구 판정/결과
- 현재 투구 실제 구종
- test.csv 내부 다른 행을 이용한 집계

## 전처리 방식

- 범주형 피처는 문자열로 변환하고 결측값은 `__MISSING__`로 채운다.
- CatBoost는 범주형 컬럼을 자체 범주형 입력으로 받는다.
- LightGBM은 학습 데이터에서 만든 범주 매핑으로 범주형 컬럼을 정수 인코딩한다.
- 수치형 값은 `pd.to_numeric`으로 변환한다.
- 비율 계열 피처는 학습 데이터 사전 평균으로 완화한다.
- train/test 입력 구조는 저장된 `input_columns`와 일치하는지 검사한다.
- 최종 v5-2 피처 행렬에서는 원본 `pitcher_id`, `batter_id`를 제거한다.
- 투수 정보는 공식 `asof_*` 이력과 Trackman 이전 시즌 요약으로 반영한다.

## 학습 방법

필요한 원본 데이터 배치:

```text
data/train.csv
data/trackman_history.csv
features/pitcher_id_mapping_clean.csv
```

필요 패키지 설치:

```bash
pip install -r requirements.txt
```

실험 실행:

```bash
python train.py
```

전체 fold 검증:

```bash
python full_validation.py
```

최종 제출 패키지 생성:

```bash
python build_submit.py
```

참고: 여기 포함된 학습 코드는 원래 작업 루트의 `/5`, `/5-1`, `/5-2` 구조를 기준으로 작성된 재현용 코드다. 공유 레포 단독 실행 시에는 데이터 경로와 원본 보조 코드 경로를 맞춰야 한다.

## 추론 방법

평가 서버는 `submit.zip`을 풀고 다음을 실행한다.

```bash
python script.py
```

`script.py`는 먼저 아래 경로를 확인한다.

```text
./open/test.csv
./open/sample_submission.csv
```

없으면 다음 경로를 대체 경로로 사용한다.

```text
./data/test.csv
./data/sample_submission.csv
```

최종 제출 파일은 다음 경로에 생성된다.

```text
output/submission.csv
```

## submit.zip 구조

`submit.zip`에는 다음 파일만 포함된다.

```text
model/
model/final_model.joblib
model/calibration_model.joblib
model/feature_schema.json
model/ensemble_config.json
script.py
requirements.txt
```

압축 무결성 검사는 통과했으며, `submit.zip` SHA256은 다음과 같다.

```text
284e2a00908ae0a5bd81ef40d57c11b3df7210dfd977e5f353b852e50b449ef2
```

## 규칙 준수 요약

- 공식 제공 데이터만 사용했다.
- 외부 API를 사용하지 않았다.
- 외부 사전학습 모델이나 가중치를 사용하지 않았다.
- test 행 사이의 집계를 만들지 않았다.
- 현재 투구 이후 정보와 실제 결과를 사용하지 않았다.
- 현재 투구의 실제 위치, 실제 구종, 판정 결과를 사용하지 않았다.
- 현재 투구 Trackman 측정값을 사용하지 않았다.
- Trackman은 이전 시즌 투수 요약 피처로만 사용했다.
- 원본 `train.csv`, `test.csv`, `trackman_history.csv`는 Git 패키지에 포함하지 않았다.

## 관련 문서

- `MODEL_CARD.md`: 모델 구성과 저장 파일 설명
- `FEATURE_ENGINEERING.md`: 전체 피처 엔지니어링 설명
- `TRACKMAN_USAGE.md`: Trackman 매핑과 사용 방식
- `PREPROCESSING.md`: 결측치, 범주형/수치형 처리 방식
- `TRAINING_AND_INFERENCE.md`: 학습/추론 실행 방법
- `RULE_COMPLIANCE.md`: 대회 규칙 준수 체크리스트
- `SCORE_REPORT.md`: 서버 점수와 개선 요약
