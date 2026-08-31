# Submit v1 candidate

개인 제출 v1의 실행 가능한 `submit.zip` 내용을 팀 검토용 후보로 정리한 폴더입니다.
학습 데이터나 실험 산출물은 포함하지 않고, 제출 당시의 추론 코드·고정 패키지·모델
artifact를 그대로 보존했습니다.

## 구성

```text
candidates/submit_v1/
├── model/
│   ├── calibration_model.joblib
│   ├── ensemble_config.json
│   ├── feature_schema.json
│   └── final_model.joblib
├── script.py
├── requirements.txt
├── README.md
└── docs/
    ├── 1_submit_zip_입력_모델_아키텍처_분석.md
    ├── 1_submit_zip_피처_엔지니어링_상세.md
    └── 팀_공통_코드작성_및_제출전_체크리스트.md
```

이 PR은 제출 후보 공유만을 목적으로 하므로 `train.py`는 포함하지 않습니다. 모델은
공식 2019~2024 `train.csv`로 이미 학습된 제출 artifact이며, 원본 CSV는 저장소에
추가하지 않았습니다.

## 모델 아키텍처

```text
47개 공식 투구 직전 입력
  → 행 단위 피처 엔지니어링
  → CatBoost 0.60 + LightGBM 0.05 + XGBoost 0.35
  → Platt 확률 보정
  → output/submission.csv
```

| 항목 | 내용 |
|---|---|
| 타깃 | `control_success=1` 확률 |
| 학습 기간 | 2019~2024 공식 학습 데이터 |
| 최종 학습 행 | 1,475,092 |
| seed | 42 |
| CatBoost | 108 trees, depth 9, lr 0.03, weight 0.60 |
| LightGBM | 100 trees, 31 leaves, lr 0.04, weight 0.05 |
| XGBoost | 100 trees, depth 7, lr 0.04, weight 0.35 |
| 보정 | Platt calibration |
| Trackman | 미사용 |
| 외부 데이터·API·가중치 | 미사용 |

CatBoost와 XGBoost는 중복 5열을 제외한 91개 피처, LightGBM은
96개 피처를 사용합니다. CatBoost는 범주형을 native categorical로,
LightGBM은 학습 mapping의 category code로, XGBoost는 학습 빈도로
처리합니다.

## 피처 요약

- 손 조합, 볼·스트라이크, 주자·아웃 조합 범주
- 접전·후반·레버리지 압박 상태
- 투수·타자 과거 표본 수의 `log1p`
- alpha 50 경험적 베이즈 비율 평활
- 최근 1·3·5경기 성공률·가운데 비율 요약과 누적 대비 gap
- 결측 16개 및 신규 투수·타자 cold-start 표시
- 과거 구종 구성 entropy와 점수 차 파생값

피처는 현재 행과 학습에서 저장한 prior·mapping만 사용합니다. test 내부
`groupby`, 빈도, rolling, target encoding을 사용하지 않아 각 행이
다른 test 행에 독립적입니다.

## 검증 기록

| 항목 | 값 |
|---|---:|
| 2024 개발 Brier | 0.2479843913 |
| 2024 개발 환산 BSS | 729.5776 |
| 실제 DACON 점수 | 907.58227 |
| 245,789행 추론 | 약 3.29초 |
| 245,789행 RAM | 약 1.43 GB |
| GPU | 불필요 |

2024 결과는 가중치 선택에 사용됐으므로 완전한 sealed holdout은
아닙니다. 또한 같은 OOF로 보정기를 학습하고 재평가한 수치는
독립 검증 점수로 사용하면 안 됩니다.

## 실행

Python 3.11 환경에서 후보 전용 패키지를 설치합니다.

```bash
pip install -r candidates/submit_v1/requirements.txt
```

`script.py`는 평가 서버의 `./open/test.csv`, `./open/sample_submission.csv`를
우선 사용하고 로컬 호환을 위해 `./data/` 경로도 지원합니다. 출력은
`./output/submission.csv`입니다. 실행 시 현재 디렉터리에 `model/`이
있어야 하므로 후보 디렉터리에서 실행합니다.

```bash
cd candidates/submit_v1
python script.py
```

실제 데이터 CSV는 저장소에 commit하지 마세요.

## 제출 아티팩트 출처

- 원본 `submit.zip` SHA-256:
  `7cb305fc8778dcc350eb20f76d4db50bd31c3c2fb20c72e7759d6a59aa9089f0`
- 이 폴더의 `script.py`, `requirements.txt`, `model/` 내용은 해당 ZIP에서
  그대로 추출했습니다.
- 추가 학습이나 모델 변환을 하지 않았습니다.

## 참고 문서

- [입력·모델·아키텍처 분석](docs/1_submit_zip_입력_모델_아키텍처_분석.md)
- [피처 엔지니어링 상세](docs/1_submit_zip_피처_엔지니어링_상세.md)
- [팀 공통 코드 작성 및 제출 전 체크리스트](docs/팀_공통_코드작성_및_제출전_체크리스트.md)

PR 검토 시에는 특히 금지 정보 미사용, test 행 독립성, 학습·추론
피처 동등성, 출력 형식, 새 Python 3.11 환경 실행 항목을 함께
확인해 주세요.
