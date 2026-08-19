# Submit v6 candidate

submit v3 파이프라인에 파생 피처 10개를 추가한 제출 후보입니다. `submit.zip`이 실제
제출물이고, 나머지는 재현용 학습 코드와 규칙 검사 스크립트입니다. 원본 데이터 CSV는
포함하지 않았습니다.

## 구성

```text
candidates/submit_v6/
├── submit.zip               # 제출물 (script.py + requirements.txt + model/)
├── script.py                # 평가 서버 추론 코드 (zip 내용과 동일)
├── requirements.txt         # 추론 환경 버전 고정
├── model/                   # final_model.joblib, calibration_model.joblib, schema, config
├── train_fast.py            # 재현용 최종 학습 (v4 확정 구성으로 바로 학습)
├── feature_engineering.py   # 피처 생성 (v3 + v4 6개 + v6 4개)
├── calibration.py / validation.py / metrics.py
├── requirements-train.txt
├── audit_submission.py      # 대회 규칙 자동 검사 (제출 전 필수)
└── output/validation_report.json
```

## 모델 요약

```text
공식 투구 직전 입력
  -> 행 단위 피처 엔지니어링 (v3 피처 + v4/v6 파생 10개)
  -> CatBoost 0.65 + XGBoost 0.30 + LightGBM 0.05  (v3/v4에서 확정한 구성)
  -> Platt 확률 보정
  -> output/submission.csv
```

## v3 대비 추가 피처

v4 (제구 실패 정의 직결, smoothed rate 사용)

- `v4_fail_prone` = middle + reverse + ball (smoothed) 합
- `v4_middle_matchup` = 투수 middle x 타자 middle
- `v4_succ_gap` = 투수 성공률 - 타자 성공률
- `v4_reverse_x_2strike`, `v4_ball_x_3ball`, `v4_middle_x_risp`

v6 (구종 운영 x 상황. 근거: 조선미(2023), 상황별 구종 예측 모델, 한국융합과학회지)

- `v6_offspeed_x_2strike`, `v6_breaking_x_2strike`
- `v6_nonfastball_x_risp` = (breaking + offspeed) x 득점권
- `v6_offspeed_x_outs`

모든 피처는 한 행 내 값만 사용합니다 (test 내 다른 행 집계 없음).

## 검증 (2024 시간 holdout, 2019~2023 학습)

| 항목 | v4 | v6 |
|---|---:|---:|
| 2024 raw Brier | 0.248315 | 0.248069 |
| 2024 raw BSS | 597.1 | 695.5 |

로컬 수치는 참고용입니다. v3 리더보드 914.15 대비 실제 성능은 제출로 확인해야 합니다.

## 규칙 검사

제출 전 `python audit_submission.py <이 폴더>` 로 14개 항목을 검사했고 ALL PASS입니다.

- 행 독립성: 단독 행 예측 == 전체 배치 예측 (max diff 0.0), 순서 뒤집기/부분집합 불변
- 원격 API / 네트워크 접근 없음
- 외부 데이터 없음 (test.csv, sample_submission.csv, model/ 만 읽음)
- 사전학습 가중치 패키지 없음
- numpy / scipy / scikit-learn / joblib / pandas 버전 고정
- 클린 추출 후 script.py 정상 실행, 출력 형식/NaN/범위 정상

## 주의: requirements 버전 고정

`requirements.txt`는 학습 환경 버전으로 고정했습니다
(numpy 2.0.2, scipy 1.13.1, scikit-learn 1.6.1, joblib 1.5.3, pandas 2.3.3,
catboost 1.2.10, lightgbm 4.6.0, xgboost 2.1.4). 보정 모델이 sklearn 객체라서
sklearn/numpy 버전이 다르면 unpickle에 실패할 수 있습니다. 참고로 submit v3의
`calibration_model.joblib`은 scikit-learn 1.6.1 환경에서 로드에 실패했으므로
(`LogisticRegression has no attribute multi_class`), v3 requirements에도
sklearn/numpy/scipy/joblib 버전을 추가하는 것을 권장합니다.

## 학습 재현

```bash
cd candidates/submit_v6
pip install -r requirements-train.txt
python train_fast.py      # data/ 에 train.csv, test.csv 필요. 약 20분
python audit_submission.py .
```

rolling 5모델 x 3폴드 검증은 시간 관계상 생략하고 v4가 확정한 앙상블 구성과
파라미터(iterations 100)를 그대로 사용했습니다.
