# Submit v3 candidate

개인 제출 v3의 실행 가능한 `/3/submit.zip` 내용을 팀 검토용 후보로 정리한 폴더입니다.
학습 데이터나 실험 산출물은 포함하지 않고, 제출 당시의 추론 코드, 고정 패키지, 모델
artifact를 그대로 보존했습니다.

## 구성

```text
candidates/submit_v3/
├── model/
│   ├── calibration_model.joblib
│   ├── ensemble_config.json
│   ├── feature_schema.json
│   └── final_model.joblib
├── script.py
├── requirements.txt
├── README.md
├── docs/
│   └── submit3_model_analysis.md
└── submit.zip
```

이 PR은 제출 후보 공유만을 목적으로 하므로 `train.py`는 포함하지 않습니다. 원본
제출 ZIP도 팀 공유와 재현 확인을 위해 `candidates/submit_v3/submit.zip`으로 함께
보관했습니다.

## 제출 결과

| 항목 | 값 |
|---|---:|
| 실제 리더보드 점수 | 914점 |
| 로컬 2024 Brier reference | 0.24803100 |
| Trackman | 미사용 |

## 모델 요약

```text
공식 투구 직전 입력
  → 행 단위 피처 엔지니어링
  → CatBoost 0.65 + XGBoost 0.35 + LightGBM 0.00
  → Platt 확률 보정
  → output/submission.csv
```

| 항목 | 내용 |
|---|---|
| 타깃 | `control_success=1` 확률 |
| 앙상블 | CatBoost + XGBoost 중심 |
| LightGBM | 포함되어 있으나 최종 가중치 0 |
| 보정 | Platt calibration |
| Trackman | 미사용 |
| 제출 구조 | `model/`, `script.py`, `requirements.txt` |

`script.py`는 평가 서버 입력을 읽어 `output/submission.csv`를 생성합니다. 추론 시
test 내부 통계나 rolling feature를 새로 만들지 않고, 저장된 모델 artifact와 schema를
사용합니다.


## 실행

후보 디렉터리에서 패키지를 설치하고 실행합니다.

```bash
pip install -r candidates/submit_v3/requirements.txt
cd candidates/submit_v3
python script.py
```

평가 서버에서는 `script.py`가 `model/`을 불러와 `output/submission.csv`를 생성합니다.
실제 데이터 CSV는 저장소에 commit하지 마세요.

## 제출 아티팩트 출처

- 원본 `submit.zip` SHA-256:
  `ac3bebeab41bf2e5a76d9357d2e6ab0e8cfe6535935419e08fad75da50363f13`
- 이 폴더의 `script.py`, `requirements.txt`, `model/` 내용은 `/3/submit.zip`에서
  그대로 추출했습니다.
- 추가 학습이나 모델 변환을 하지 않았습니다.

## 참고 문서

- [3/submit.zip 모델 분석](docs/submit3_model_analysis.md)
