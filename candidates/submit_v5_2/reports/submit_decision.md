# 5-2 제출 파일 생성 결과

## 결론

`5-2/submit.zip`을 생성했다. 형식 검증, 압축 무결성 검사, 임시 제출 환경 실행 검사를 모두 통과했다.

다만 `/5` 서버 제출 점수 `938.2526739958`을 이미 확인한 상태라면, `5-2`는 평균 Brier 개선을 노린 대체 후보이지 확실한 상위 제출본은 아니다. full fold 기준으로 평균 Brier는 좋아졌지만 worst fold는 거의 동률이다.

## 최종 패키지 구성

- 선택 후보: `tm_metadata_physical_pitchmix`
- Trackman 전략: `mapping_all_shrink`
- 모델: CatBoost `0.8` + LightGBM `0.2`
- calibration: `platt`
- 최종 학습 rows: `1,475,092`
- 최종 feature 수: `118`
- zip SHA256: `284e2a00908ae0a5bd81ef40d57c11b3df7210dfd977e5f353b852e50b449ef2`

## OOF 검증 결과

| valid season | raw Brier | calibrated Brier | AUC |
|---:|---:|---:|---:|
| 2022 | 0.24329609 | 0.24325906 | 0.579127 |
| 2023 | 0.24987258 | 0.24986192 | 0.526462 |
| 2024 | 0.24783842 | 0.24776319 | 0.550264 |

- calibrated mean Brier: `0.24696139`
- calibrated worst Brier: `0.24986192`

## 제출 형식 검사

- zip members:
  - `model/`
  - `model/final_model.joblib`
  - `model/calibration_model.joblib`
  - `model/feature_schema.json`
  - `model/ensemble_config.json`
  - `script.py`
  - `requirements.txt`
- `unzip -t 5-2/submit.zip`: 통과
- 임시 폴더에서 `script.py` 실행: 통과
- 생성 파일: `output/submission.csv`
- 테스트 실행 결과: rows `5`, columns `row_id`, `control_success`, NaN `0`, 예측 범위 `[0.40797738, 0.49590037]`

## 주의

이 패키지는 제출 오류가 나지 않도록 형식은 맞췄지만, `/5`보다 서버 점수가 높다는 보장은 없다. `/5` 대비 확실한 개선 후보라기보다는 Trackman 추가가 평균 Brier를 아주 작게 낮춘 실험 제출본으로 보는 것이 맞다.
