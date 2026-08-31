# test row 독립성 검증 보고서

## 검증 방식

1. `submit.zip`을 임시 폴더에 풀었다.
2. `open/test.csv`, `open/sample_submission.csv`를 넣고 `script.py`를 실행했다.
3. 전체 test 예측값을 계산했다.
4. 같은 feature/predict 함수로 일부 row subset과 단일 row만 따로 예측했다.
5. 같은 `row_id`의 예측값 차이가 `1e-12` 이하인지 확인했다.

## 결과

- `script.py` 실행: 통과
- `output/submission.csv` 생성: 통과
- 제출 컬럼: `row_id`, `control_success`
- 예측값 finite 검사: 통과
- 예측값 `[0, 1]` 범위 검사: 통과
- subset max abs diff: `0.0`
- single row max abs diff: `0.0`
- 허용 오차: `1e-12`
- test row 독립성: 통과

## 수정 사항

초기 검증에서 subset 예측 시 원본 index가 유지되면 Trackman feature concat이 index 기준으로 어긋나는 문제가 발견됐다. `script.py`의 `component_frame()`에서 feature frame과 Trackman frame을 position 기준으로 `reset_index(drop=True)` 처리하도록 수정했고, 수정 후 독립성 검증을 통과했다.
