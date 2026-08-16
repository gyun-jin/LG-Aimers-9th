# 5-3 제출 패키지 검증

## submit.zip 구조

`submit.zip` 최상위 구성:

- `model/`
- `model/final_model.joblib`
- `model/calibration_model.joblib`
- `model/feature_schema.json`
- `model/ensemble_config.json`
- `script.py`
- `requirements.txt`

## 검증 결과

- `py_compile`: 통과
- `unzip -t submit.zip`: 통과
- 임시 폴더 압축 해제: 통과
- `script.py` 실행: 통과
- `output/submission.csv` 생성: 통과
- 컬럼 `row_id`, `control_success`: 통과
- 예측값 finite: 통과
- 예측값 0~1 범위: 통과
- test row 독립성: 통과

## test 독립성

- subset max abs diff: `0.0`
- single row max abs diff: `0.0`
- 허용 오차: `1e-12`
