# 규칙 준수 체크

## 사용 데이터

- `hand_cleaning_analysis/train_hand_trackman_clean.csv`
- `data/test.csv`
- `data/sample_submission.csv`
- `data/trackman_history.csv`
- `train_trackman/pitcher_id_mapping_clean.csv`

외부 데이터, 외부 API, 외부 사전학습 모델은 사용하지 않았다.

## Trackman 사용

- 현재 투구의 Trackman 측정값을 test row에 직접 붙이지 않았다.
- 2025년 Trackman 데이터는 사용하지 않았다.
- Trackman은 과거 시즌 투수 단위 요약 피처로만 사용했다.
- `mapping_all_shrink` 방식으로 mapping confidence와 prior fallback을 적용했다.

## test 독립성

`script.py`는 각 test row 자체 값과 학습 때 저장된 model state만 사용한다.

사용하지 않은 처리:

- test 전체 groupby
- test 전체 value_counts
- test 전체 평균/분포/빈도 기반 보정
- rolling, expanding, shift, lag
- sort 기반 누적
- test 내부 target encoding
- test row 사이의 선수/팀/월/경기 단위 집계

## 금지 정보

사용하지 않은 정보:

- 현재 투구 이후 정보
- 현재 투구 실제 위치/코스
- 현재 투구 판정/결과
- 현재 투구 `control_success`
- 현재 투구 실제 구종
- 현재 투구 Trackman 측정값

## submit.zip 검증

- `py_compile`: 통과
- `unzip -t submit.zip`: 통과
- 최상위 구조:
  - `model/`
  - `model/final_model.joblib`
  - `model/calibration_model.joblib`
  - `model/feature_schema.json`
  - `model/ensemble_config.json`
  - `script.py`
  - `requirements.txt`
- 임시 폴더 실행: 통과
- `output/submission.csv` 생성: 통과
- test row 독립성 검증: 통과

## 제출 판단

`submission_recommended=true`

full validation에서 `/5-2` 대비 mean Brier와 worst Brier가 모두 개선됐고, 제출 패키지 검증과 test row 독립성 검증을 통과했다.
