# submit_tabmcat (tM-v1): CatBoost + TabM 앙상블

최종 모델 브랜치 `tM-v1`에 포함된 제출 후보입니다.

최종 리더보드 점수는 **1029.04299점**입니다. 이 패키지의 초기 커밋 시점 기록은 1008.06이었으며, 이후 최종 제출 결과를 README에 반영했습니다.

## 구성

```
p = 0.95 * CatBoost(seed 42/43/44 평균) + 0.05 * TabM-mini(seed 0/1/2 평균)
final = Platt(p)   # coef 1.05442, intercept -0.04414 (2024 홀드아웃 기준)
```

- 피처: v5-3 파이프라인 138개 그대로 (6개 모델 동일 입력)
- CatBoost: v5-3 하이퍼 동일 (iterations 520, lr 0.03, depth 8, l2 20)
- TabM-mini: MLP 512×3, K=32 멤버(±1 입력 어댑터 공유형), 범주 13개는 임베딩(dim 16),
  트랙맨 물리량 8개(구속·회전·IVB·HB·릴리즈 등) 보조 회귀 head로 멀티태스크 학습
  (트랙맨은 학습 단계에서만 사용, 추론 입력은 test 컬럼뿐 — 운영진 허용 범위)
- LGBM 제외: 3개 fold 실측에서 CatBoost 단독과 ±0.000004 이내 동률이라 단순화
- 추론: TabM은 float64 CPU 순전파 → 행독립 diff 정확히 0.0

## 검증 (train<VAL → VAL 미래예측, Platt는 val 반분 교차적합)

| fold | CatBoost 단독 | +TabM w0.05 (d, 95% CI) |
|---|---|---|
| 2022 | 0.243055 | −0.000010 [−13, −6]e-6 개선* |
| 2023 | 0.249709 | +0.000006 [+5, +8]e-6 악화* |
| 2024 (3-seed) | 0.247609 | −0.000004 [−8, −0]e-6 개선* |

- 2023(체제전환 해)에서 소폭 악화가 남아 w=0.05로 하방을 제한함
- 가중치를 0.2 이상 올리면 2023 악화가 커져 금지 (+0.000025~)
- 리더보드 실측도 예측 범위(2호 1010 ± seed 노이즈) 안에 들어옴

## 규칙 준수 (audit 16항목 ALL PASS)

- 행독립 3종(단독행==배치 / 순서뒤집기 / 부분집합) diff 0.0
- 원격 API·외부 데이터·사전학습 가중치 없음 (TabM 가중치는 대회 train.csv로만 학습)
- requirements 전 줄 버전 핀, 채점환경(numpy 1.26.4/pandas 2.0.3) 재현 실행 통과
- 전량 리허설(25.4만행): 227초, 피크 4.7GB

## 재현 방법 (train/)

1. `tabm_bv.py` — TabM 학습. env: `VAL`(fold 연도, 2025=전체 학습) `SEED` `GPU_ID` `SAVE_STATE=1`(state+prep 저장)
2. `stage3_cat.py` — CatBoost 3-seed. env: `VAL` `RUN=holdout|final`
3. `eval_ens.py` — 3-fold(2022/2023/2024) 혼합 가중치 격자 평가
4. `package_tabmcat.py` — script.py 조립 + Platt 적합 + zip 생성
5. `audit_tabmcat.py` / `measure_tabmcat.py` — 규칙 검사 16항목 / 전량 리허설

경로는 학습 서버(~/jegu) 기준이므로 환경에 맞게 수정 필요.
`ext_tabmcat_snippet.py`는 script.py에 붙는 추론 확장부(패키징 시 자동 조립).
