# Submit v3 candidate

개인 제출 v3의 실행 가능한 `/3/submit.zip`과 해당 제출을 만들 때 사용한 재현용
학습 코드 후보, 추론 코드, 설정, 분석 문서를 팀 검토용으로 정리한 폴더입니다.
원본 학습/평가 데이터 CSV는 저장소에 포함하지 않았습니다.

## 제출 결과

| 항목 | 값 |
|---|---:|
| 실제 리더보드 점수 | 914점 |
| 로컬 2024 Brier reference | 0.24803100 |
| Trackman | 미사용 |

## 목적

submit v3 패키지를 팀원이 그대로 실행해 제출 구조를 확인하고, `solution_v3/src`
기반의 학습 파이프라인을 검토할 수 있게 하는 것이 목적입니다. `submit.zip`과
zip 내부 파일은 원본 제출물을 보존하고, 학습 코드는 재현 후보로 함께 제공합니다.

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
├── requirements-train.txt
├── train.py
├── feature_engineering.py
├── calibration.py
├── validation.py
├── metrics.py
├── inference_utils.py
├── data_audit.py
├── final_verify.py
├── eval_2024.py
├── eval_2024_direct.py
├── eval_2024_exact.py
├── README.md
├── notebooks/
│   └── .gitkeep
├── reports/
│   ├── submit3_model_analysis.md
│   ├── final_v3_vs_v1_report.md
│   ├── context_feature_ablation_report.md
│   ├── season_game_type_dependency_report.md
│   ├── feature_importance_and_relationships.md
│   ├── data_description.md
│   ├── REQUIREMENTS_CHECKLIST.md
│   └── VALIDATION_REPORT.md
├── docs/
│   └── submit3_model_analysis.md
└── submit.zip
```

`docs/submit3_model_analysis.md`는 이전 공유 브랜치에서 유지된 문서이고,
새 분석 문서는 `reports/`에 모았습니다.

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

## 사용 피처 요약

submit v3는 공식 `train.csv`/`test.csv`에 포함된 투구 직전 정보와 사전 누적 통계를
사용합니다.

- 경기/시즌 정보: `season`, `game_type`, inning, count, score, leverage index
- 투수/타자 누적 성향: as-of success/reverse/middle/ball/strike rate
- 최근 투수 흐름: 직전 1/3/5경기 성공률과 누적 성향 대비 gap
- 카운트/주자/아웃 상황: count state, base/out state, close/late/pressure state
- 구종 운영 요약: fastball/breaking/offspeed 비율과 pitch mix entropy
- ID/팀/좌우 정보: pitcher/batter/team/hand matchup categorical feature

Trackman 원본 데이터와 `pitcher_id_mapping_clean.csv`는 submit v3에 사용하지 않았습니다.

## 학습 코드 실행

학습 코드는 `solution_v3/src`에서 가져온 재현용 후보입니다. 원본 데이터는 저장소에
포함하지 않았으므로, 실행 전 대회 데이터가 로컬 데이터 경로에 있어야 합니다.

```bash
cd candidates/submit_v3
pip install -r requirements-train.txt
python train.py
```

검증용 스크립트는 필요에 따라 아래처럼 실행합니다.

```bash
python eval_2024.py
python final_verify.py
```

## 추론 코드 실행

후보 디렉터리에서 제출 패키지 의존성을 설치하고 실행합니다.

```bash
cd candidates/submit_v3
pip install -r requirements.txt
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

## 대회 규칙 준수 사항

- 제출 zip은 `model/`, `script.py`, `requirements.txt` 구조를 유지합니다.
- 추론 시 `./open/test.csv`, `./open/sample_submission.csv`를 우선 사용하고 로컬
  검증용 `./data/` 경로도 fallback으로 지원합니다.
- `train.csv`, `test.csv`, `trackman_history.csv` 같은 원본 대용량 데이터는 커밋하지
  않았습니다.
- `lg.pem`, `.venv`, `__pycache__`, `.DS_Store`, `catboost_info` 같은 민감/불필요
  파일은 포함하지 않았습니다.
- Trackman은 submit v3에서 미사용입니다.

## 참고 문서

- [3/submit.zip 모델 분석](reports/submit3_model_analysis.md)
- [v3 vs v1 재개선 보고서](reports/final_v3_vs_v1_report.md)
- [context feature ablation 보고서](reports/context_feature_ablation_report.md)
- [season/game_type dependency 보고서](reports/season_game_type_dependency_report.md)
