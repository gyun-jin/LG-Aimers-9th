# tM-v1 최종 모델 카드

## 1. 최종 결과

- 브랜치: `tM-v1`
- 모델 패키지: `candidates/submit_tabmcat/`
- 최종 리더보드 점수: **1029.04299점**
- 목적: 투구별 `control_success` 확률 예측
- 평가 관점: Brier score 중심의 확률 예측 품질

이 문서는 저장소의 최종 후보를 처음 보는 사람이 모델의 구조, 선택 이유, 실행 범위, 규칙 준수 여부를 이해할 수 있도록 작성한 모델 카드다.

## 2. 구조

최종 모델은 CatBoost를 주 모델로 사용하고 TabM을 작은 가중치로 보조한다.

```text
공식 투구 직전 입력
  -> 공통 feature 생성
  -> CatBoost seed 42/43/44 평균
  -> TabM seed 0/1/2 평균
  -> 0.95 * CatBoost + 0.05 * TabM
  -> Platt calibration
  -> control_success 확률
```

### CatBoost

- 3-seed ensemble: 42, 43, 44
- v5-3 계열의 안정적인 tabular feature와 native categorical 처리
- CatBoost를 0.95로 두어 주 모델 역할 유지

### TabM

- 3-seed ensemble: 0, 1, 2
- MLP 512×3, K=32 member 구조
- 범주형 입력은 embedding 사용
- Trackman 관련 auxiliary regression head를 학습 단계에서 사용
- 최종 test 추론에서는 test 컬럼과 저장된 전처리 상태만 사용
- downstream Brier의 변동을 제한하기 위해 최종 가중치를 0.05로 제한

### Calibration

CatBoost와 TabM의 혼합 raw probability에 Platt calibration을 적용했다. 모델의 ranking 성능만 보지 않고 확률의 절대적인 신뢰도를 보정하기 위한 단계다.

## 3. feature 설계

### 입력 상황

- balls, strikes, count state
- outs, runner/base state
- inning phase와 late inning
- 점수 차이, close game, pressure state
- leverage index bucket
- pitcher hand × batter hand matchup

### 과거 투수·타자 profile

- career 및 이전 시즌 성공률
- 최근 1·3·5경기 form
- 과거 투구·경기 표본 수
- career 대비 최근 변화량
- ball, strike, middle, reverse 관련 비율
- smoothing prior와 reliability
- profile availability와 cold-start flag

### Trackman

Trackman은 현재 test 투구의 물성을 직접 사용하지 않고, 허용된 과거 투수 profile로 변환했다. 매칭된 과거 기록에서 물성의 평균·변동성·pitchmix·history availability를 계산하고 저장된 상태를 추론에 재사용한다.

## 4. 왜 CatBoost + TabM인가

초기에는 CatBoost-only, Trackman profile, R/F regime, 구종 예측, TabM을 각각 독립적으로 비교했다. TabM 단독 또는 full-feature TabM은 CatBoost 기준선보다 안정적인 개선을 만들지 못했다.

그러나 TabM은 CatBoost가 놓칠 수 있는 연속형 조합과 보조적인 Trackman representation을 제공할 가능성이 있었다. 따라서 TabM을 주 모델로 교체하지 않고 0.05의 작은 weight로 제한했다.

검증에서 2023년 regime 전환 구간의 악화가 확인됐기 때문에 TabM weight를 0.2 이상으로 올리지 않았다. 최종 구조는 복잡한 모델을 무조건 강화한 것이 아니라, 보조 모델의 잠재적인 정보만 취하고 최악 fold의 하방 위험을 제한한 선택이다.

## 5. 데이터 누수와 규칙 통제

- validation 시즌 target은 해당 시즌 profile에 사용하지 않음
- 시즌 S의 profile은 S-1 이하 정보로 생성
- test 전체 행의 평균·빈도·rolling·순위·target encoding 미사용
- 현재 test 투구의 Trackman 측정값 직접 사용 금지
- 2025 Trackman 미사용
- 추론 script에서 원본 Trackman 재매칭 금지
- 공식 데이터와 저장된 모델·profile만 사용
- 다른 팀의 변수명과 코드를 그대로 복사하지 않음

특히 Trackman은 train과 직접 공통 ID가 없어 경기 fingerprint와 투구 sequence로 매칭했다. 매칭되지 않은 행은 억지로 채우지 않고 unmatched로 남겼으며, train row와 Trackman row의 중복 매칭도 검사했다.

## 6. 검증과 제출 artifact

제출 전 다음을 확인했다.

- Python syntax compile
- zip 무결성
- 임시 폴더 unzip 후 추론
- `row_id, control_success` 컬럼
- finite 및 `[0, 1]` 예측 범위
- 전체 test와 단일 행 추론 일관성
- 원본 대용량 데이터와 민감 파일 제외
- 모델 파일, preprocessing, calibration, schema의 동시 보존

## 7. 재현 가능한 파일

| 경로 | 역할 |
|---|---|
| `candidates/submit_tabmcat/script.py` | 제출 환경 추론 진입점 |
| `candidates/submit_tabmcat/requirements.txt` | 추론 의존성 |
| `candidates/submit_tabmcat/model/` | 학습된 모델과 전처리·calibration 상태 |
| `candidates/submit_tabmcat/train/tabm_bv.py` | TabM 학습 |
| `candidates/submit_tabmcat/train/stage3_cat.py` | CatBoost 학습 |
| `candidates/submit_tabmcat/train/eval_ens.py` | 혼합 가중치 평가 |
| `candidates/submit_tabmcat/train/package_tabmcat.py` | 제출 package 생성 |
| `candidates/submit_tabmcat/train/audit_tabmcat.py` | 규칙 및 행 독립성 감사 |
| `candidates/submit_tabmcat/train/measure_tabmcat.py` | 전량 추론 측정 |

학습 스크립트에 기록된 경로는 원래 학습 서버 기준일 수 있으므로, 다른 환경에서는 공식 데이터 위치와 실행 경로를 먼저 수정해야 한다. 저장소에는 공식 train/test/Trackman 원본을 포함하지 않는다.

## 8. 프로젝트에서 얻은 결론

성능 개선은 모델을 복잡하게 만드는 것만으로 얻어지지 않았다. 기준선 재현, 시간 누수 방지, Trackman 매칭 품질, R/F feature ablation, calibration, 제출 artifact 검증을 함께 관리해야 했다.

최종적으로 CatBoost를 중심에 두고 TabM을 작은 가중치로 제한한 이유는 다음과 같다.

1. CatBoost가 범주형과 sparse profile을 안정적으로 처리했다.
2. TabM 단독은 여러 fold에서 기준선보다 나빴다.
3. TabM을 크게 섞으면 2023 전환 구간 악화가 커졌다.
4. 작은 가중치는 보조 정보는 활용하면서 큰 분산을 제한했다.
5. 최종 제출은 모델 성능뿐 아니라 재현성과 규칙 준수가 필요했다.
