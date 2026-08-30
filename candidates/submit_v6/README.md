# Submit v6 candidate

submit v3 파이프라인에 파생 피처를 추가한 제출 후보입니다. 현재 `model/`과
`submit.zip`에는 strict temporal OOF 검증 후 전체 데이터로 다시 학습한 hybrid 모델이
반영되어 있습니다. 원본 데이터 CSV는 포함하지 않았습니다.

## 구성

```text
candidates/submit_v6/
├── submit.zip               # 제출물 (script.py + requirements.txt + model/)
├── script.py                # legacy v1과 신규 v2를 모두 지원하는 평가 서버 추론 코드
├── requirements.txt         # 추론 환경 버전 고정
├── model/                   # final_model.joblib, calibration_model.joblib, schema, config
├── train_fast.py            # strict temporal OOF 검증 + 최종 학습
├── feature_engineering.py   # 버전 호환 피처 생성 (legacy v1 / 정리된 v2)
├── calibration.py / validation.py / metrics.py
├── requirements-train.txt
├── audit_submission.py      # 대회 규칙 자동 검사 (제출 전 필수)
└── output/validation_report.json
```

## 모델 요약

```text
공식 투구 직전 입력
  -> 행 단위 피처 엔지니어링 (v3 피처 + v4/v6 파생 10개)
  -> CatBoost + XGBoost + LightGBM
  -> 검증된 CatBoost 0.65 + XGBoost 0.30 + LightGBM 0.05 고정 가중치
  -> 2021~2023 strict temporal OOF에서 Platt fit, 2024에서 채택 여부 평가
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

XGBoost/LightGBM은 feature v2를 사용해 v6 상호작용에 fold-fit prior로 평활한 구종
비율을 적용하고 reliability/posterior SD를 추가합니다. CatBoost는 2024 검증에서 더
안정적이었던 legacy v1 레이아웃을 유지합니다. XGBoost/LightGBM의 `season`도 보존해
연도별 성공률 하락을 반영합니다.

모든 피처는 한 행 내 값만 사용합니다 (test 내 다른 행 집계 없음).

## 저장 모델 검증

| 항목 | 기존 v6 | 현재 hybrid v6 |
|---|---:|---:|
| 2024 raw Brier | 0.248069 | 0.247998 |
| 2024 raw BSS | 695.5 | 724.1 |

로컬 수치는 참고용입니다. v3 리더보드 914.15 대비 실제 성능은 제출로 확인해야 합니다.
Platt는 독립 2024 구간에서 raw보다 나빠 채택하지 않았습니다. 극단적인 pooled OOF
가중치는 진단값으로만 기록하고 실제 모델에는 사용하지 않습니다.

## 규칙 검사

제출 전 `python audit_submission.py <이 폴더>` 로 14개 항목을 검사했고 ALL PASS입니다.

- 행 독립성: 단독 행 예측 == 전체 배치 예측 (max diff 0.0), 순서 뒤집기/부분집합 불변
- 원격 API / 네트워크 접근 없음
- 외부 데이터 없음 (test.csv, sample_submission.csv, model/ 만 읽음)
- 사전학습 가중치 패키지 없음
- numpy / scipy / scikit-learn / joblib / pandas 버전 고정
- 클린 추출 후 script.py 정상 실행, 출력 형식/NaN/범위 정상

## 주의: requirements 버전 고정

`requirements.txt`는 실제 학습 환경 버전으로 고정했습니다
(numpy 1.26.4, scipy 1.15.3, scikit-learn 1.8.0, joblib 1.5.3, pandas 2.0.3,
catboost 1.2.10, lightgbm 4.7.0, xgboost 3.2.0). 보정 모델이 sklearn 객체라서
sklearn/numpy 버전이 다르면 unpickle에 실패할 수 있습니다. 참고로 submit v3의
`calibration_model.joblib`은 scikit-learn 1.6.1 환경에서 로드에 실패했으므로
(`LogisticRegression has no attribute multi_class`), v3 requirements에도
sklearn/numpy/scipy/joblib 버전을 추가하는 것을 권장합니다.

## 학습 재현

```bash
cd candidates/submit_v6
pip install -r requirements-train.txt
python train_fast.py      # 저장소 data/ 에 train.csv, test.csv 필요
python audit_submission.py .
```

시간 fold는 2019~2020→2021, 2019~2021→2022, 2019~2022→2023,
2019~2023→2024입니다. 고정 가중치는 유지하고 2021~2023 strict OOF로 학습한 보정기를
2024에서 평가합니다. 실제 개선된 경우에만 같은 종류의 보정기를 2021~2024 전체 strict
OOF로 다시 fit합니다.
모델 파라미터(iterations 100)는 기존 v4 설정을 유지합니다.

Windows wheel에서는 pandas/CatBoost가 LightGBM보다 먼저 로드되거나 CatBoost fit 뒤에
LightGBM fit을 수행하면 OpenMP DLL 충돌이 발생할 수 있습니다. 학습 코드는 LightGBM을
가장 먼저 import하고 모든 LightGBM fold/final fit을 CatBoost보다 먼저 끝내도록 고정했습니다.
추론 코드도 모델 역직렬화와 pandas import 전에 LightGBM 런타임을 먼저 초기화합니다.
