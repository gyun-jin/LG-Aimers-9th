# 학습과 추론

## requirements 설치

```bash
pip install -r requirements.txt
```

## 학습에 필요한 데이터

공식 원본 데이터는 Git 패키지 밖에 배치한다.

```text
data/train.csv
data/trackman_history.csv
```

작은 매핑 파일은 이 패키지에 포함했다.

```text
features/pitcher_id_mapping_clean.csv
```

## 학습 명령

후보 실험:

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

## 추론 명령

`submit.zip`을 푼 환경에서 다음을 실행한다.

```bash
python script.py
```

스크립트는 다음 경로에서 입력 파일을 찾는다.

```text
./open/test.csv
./open/sample_submission.csv
./data/test.csv
./data/sample_submission.csv
```

예측 결과는 다음 파일로 저장된다.

```text
output/submission.csv
```

## 평가 서버 동작

대회 평가 서버는 `requirements.txt`를 설치하고, `submit.zip`을 푼 뒤, test 데이터를 `open/` 또는 `data/`에 추가하고 `script.py`를 실행한다.

## `submit.zip` 재생성

`build_submit.py`는 선택된 `tm_metadata_physical_pitchmix` 모델을 학습하고, 모델 artifact를 저장하고, 추론 스크립트를 복사한 뒤 `submit.zip`을 생성한다.

예상 제출 파일 구성:

```text
model/
model/final_model.joblib
model/calibration_model.joblib
model/feature_schema.json
model/ensemble_config.json
script.py
requirements.txt
```
