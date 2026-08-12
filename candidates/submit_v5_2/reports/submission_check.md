# 5-2 제출 가능성 점검

## 결론

제출 가능하다. `submit.zip` 구조, 용량, 압축 무결성, 샘플 추론 실행을 모두 통과했다. 서버 제출 점수는 `957.6800554874`로 확인됐다.

## 규칙 체크

| 항목 | 결과 | 근거 |
|---|---|---|
| zip 용량 10GB 이하 | 통과 | 약 `3.7M` |
| 압축 해제 32GB 이하 | 통과 | 압축 내부 총 약 `3.9M` |
| 최상위 구조 | 통과 | `model/`, `script.py`, `requirements.txt` |
| `output/submission.csv` 생성 | 통과 | 임시 폴더 실행 확인 |
| 제출 컬럼 | 통과 | `row_id`, `control_success` |
| 예측 범위 | 통과 | `[0.40797738, 0.49590037]` on sample |
| NaN 없음 | 통과 | sample 실행 NaN `0` |
| 외부 다운로드/인터넷 | 통과 | 없음 |
| test 내부 집계/target leakage | 통과 | 없음 |
| Trackman 사용 | 조건부 통과 | 제공 보조 데이터 기반 prior-season summary를 학습 bundle에 저장해서 사용 |
| 서버 점수 | 확인 | `957.6800554874` |

## 패키지

- `5-2/submit.zip`
- SHA256: `284e2a00908ae0a5bd81ef40d57c11b3df7210dfd977e5f353b852e50b449ef2`
- 구성:
  - `model/final_model.joblib`
  - `model/calibration_model.joblib`
  - `model/feature_schema.json`
  - `model/ensemble_config.json`
  - `script.py`
  - `requirements.txt`

## 제출 리스크

형식 오류 가능성은 낮다. 다만 서버 점수 관점에서는 `/5`보다 확실히 좋다고 보기 어렵다. 평균 OOF Brier는 좋아졌지만 worst fold는 거의 동률이다.
