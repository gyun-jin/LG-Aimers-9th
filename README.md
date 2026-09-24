# LG Aimers 9기 데이터톤

투구 단위 `control_success` 성공 확률을 예측하는 LG Aimers 9기 데이터톤 프로젝트 저장소입니다. 공식 경기·투구 상황 데이터를 기반으로 시간 누수를 통제한 feature를 만들고, CatBoost와 TabM 후보를 비교한 뒤 제출 가능한 모델 artifact로 패키징했습니다.

## 최종 모델

최종 모델은 `tM-v1` 브랜치의 `candidates/submit_tabmcat/` 패키지입니다.

| 항목 | 내용 |
|---|---|
| 최종 브랜치 | `tM-v1` |
| 최종 모델 패키지 | `candidates/submit_tabmcat/` |
| 최종 리더보드 점수 | **1029.04299점** |
| 예측 대상 | `control_success = 1` 확률 |
| 모델 구조 | CatBoost 3-seed + TabM 3-seed 앙상블 |
| 최종 가중치 | CatBoost 0.95, TabM 0.05 |
| 확률 보정 | Platt calibration |
| 학습 데이터 | 공식 2019~2024 train 데이터 |
| 추론 입력 | 제출용 test 행 단위 입력만 사용 |

최종 예측은 다음과 같이 계산합니다.

```text
cat_pred = mean(CatBoost(seed=42, 43, 44))
tabm_pred = mean(TabM(seed=0, 1, 2))
raw_pred = 0.95 * cat_pred + 0.05 * tabm_pred
final_pred = Platt(raw_pred)
```

TabM은 CatBoost를 대체하지 않고 작은 가중치로 보조한다. validation에서 2023 regime 전환 구간의 악화를 제한하기 위해 TabM 가중치를 0.05로 낮췄고, 0.2 이상에서는 악화폭이 커져 사용하지 않았다.

## 대회 문제와 제약

- 각 투구의 제구 성공 여부에 대한 확률 예측
- 주요 평가 지표는 Brier score 기반 확률 품질
- 단순 분류 정확도보다 calibration과 미래 시즌 일반화가 중요
- validation은 과거 시즌으로 학습하고 미래 시즌을 예측하는 chronological split 사용
- 현재 평가 투구의 미래 정보, test 전체의 통계, validation target을 profile에 사용하지 않음
- 최종 추론 script는 저장된 모델과 전처리 상태만 사용
- 제출물에는 원본 대용량 데이터와 외부 API, 사전학습 가중치를 포함하지 않음

## 저장소 구조

```text
.
├── candidates/
│   ├── baseline/                 # 새 후보 작성용 템플릿
│   ├── submit_v1/                # 초기 개인 제출 후보
│   ├── submit_v3/                # v3 CatBoost 중심 후보
│   ├── submit_v5_2/              # Trackman prior profile 후보
│   ├── submit_v5_3/              # current-season feature 후보
│   ├── submit_v5_8/              # CatBoost-only 5-seed 후보
│   └── submit_tabmcat/           # tM-v1 최종 CatBoost + TabM 후보
├── docs/
│   └── tM-v1-final-model.md      # 최종 모델 카드와 재현·규칙 검증 기록
├── features/                     # 공용 feature 아이디어 영역
├── submission/                   # 제출 구조 예시 및 공용 추론 진입점
├── CONTRIBUTING.md               # 브랜치·PR·모델 공유 규칙
└── README.md
```

## 최종 모델 패키지 구조

```text
candidates/submit_tabmcat/
├── model/
│   ├── final_model.joblib        # CatBoost 모델 bundle
│   ├── gbdt_cat3.joblib          # CatBoost seed ensemble state
│   ├── tabm_s0.pt                # TabM seed 0
│   ├── tabm_s1.pt                # TabM seed 1
│   ├── tabm_s2.pt                # TabM seed 2
│   ├── feature_schema.json       # 학습·추론 feature schema
│   ├── tabm_prep.json            # TabM 전처리 상태
│   └── platt_tabmcat.json        # 최종 확률 보정 상태
├── script.py                     # 제출 환경 추론 코드
├── requirements.txt              # 버전 고정 의존성
├── submit.zip                    # 제출용 압축 파일
├── README.md                     # 모델별 설명
└── train/                        # 학습·평가·패키징 보조 코드
```

## 주요 feature 방향

- 투수·타자의 과거 as-of 성공률과 표본 수
- 볼·스트라이크·아웃·주자·이닝·점수 차이 등 투구 직전 경기 상황
- hand matchup, count state, pressure state
- 최근 경기 form과 career 대비 변화량
- 과거 시즌 Trackman physical·pitchmix profile
- 결측, cold-start, profile availability와 reliability flag
- CatBoost native categorical 처리

현재 행 이후의 target은 사용하지 않으며, Trackman은 최종 제출에서 현재 투구의 측정값이 아니라 허용된 과거 profile 형태로 관리했습니다.

## 모델 비교 흐름

1. CatBoost-only baseline을 재현해 비교 기준을 고정
2. iteration, learning rate, seed ensemble 검증
3. current-season과 최근 form feature 검증
4. Trackman 과거 profile과 투구 단위 매칭 품질 검증
5. R/F 및 2023 전후 regime feature ablation
6. 구종 예측과 TabM을 별도 모델로 검증
7. TabM을 작은 가중치로 추가하고 2023 악화를 제한
8. 모델·calibration·schema·script를 하나의 제출 artifact로 고정

모든 feature를 통합하는 것이 항상 좋지는 않았다. 여러 시즌의 Brier, raw/calibrated 결과, 제출 규칙, 행 독립성을 함께 확인해 최종 구조를 선택했다.

## 실행

```bash
pip install -r candidates/submit_tabmcat/requirements.txt
python candidates/submit_tabmcat/script.py
```

실제 학습 서버 경로를 사용하는 재현 스크립트는 `candidates/submit_tabmcat/train/`에 있습니다. 경로와 데이터 위치는 실행 환경에 맞게 수정해야 합니다.

## 제출 전 확인

- `row_id, control_success` 컬럼 형식
- 예측값 finite 여부
- 예측값 `[0, 1]` 범위
- 전체 test와 한 행 test의 예측 일관성
- zip 내부 최상위 구조
- 원본 데이터·민감 파일 미포함
- 추론 시 외부 API와 test cross-row 집계 미사용

자세한 최종 모델 설명, 실험 선택 이유, 규칙 감사, 재현 절차는 [tM-v1 최종 모델 카드](docs/tM-v1-final-model.md)를 참고하세요.
