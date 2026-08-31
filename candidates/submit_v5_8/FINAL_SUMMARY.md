# 5-8 CatBoost-only 개선 실험

## 실행 요약

- 목표: `/5-3` CatBoost-only 모델 성능 개선.
- LGBM: 사용하지 않음.
- 최종 후보 선택: no calibration 제외, calibration 적용 후보만 비교.
- 기준 `/5-3 cat only` calibrated mean Brier: `0.24688214`.
- best candidate: `cat_seed_ensemble_5`.
- best calibration: `global_platt`.
- best mean Brier: `0.24684710`.
- 기준 대비 차이: `-0.00003504`.
- submission_recommended: `true`.

## 후보 설계

- `cat_seed_ensemble_5`: 동일 피처 5-seed CatBoost 평균 ensemble.
- 이번 full 복구 실행 제외 후보: `cat53_repro_seed42, cat_seed_ensemble_3, cat_current_alpha_100_seed3, cat_season_stable_seed3`.

## Feature/전처리

- 기본 feature: `/5` v3 schema + 현재시즌 파생피처 + `tm_metadata_physical_pitchmix`.
- Trackman prior feature 수: `29`.
- 현재 투구 Trackman 측정값, 2025 Trackman, test 내부 groupby/rolling/분포/사후보정은 사용하지 않는다.
- script.py는 저장된 train-time state와 row별 입력만 사용한다.

## 후보별 결과

| candidate | calibration | raw mean Brier | calibrated mean Brier | worst Brier | mean BSS | vs 5-3 cat only |
|---|---|---:|---:|---:|---:|---:|
| cat_seed_ensemble_5 | global_platt | 0.24688986 | 0.24684710 | 0.24989712 | 1126.88 | -0.00003504 |

## Fold별 결과

### cat_seed_ensemble_5
| season | Brier | LogLoss | AUC | mean_pred | mean_target | BSS |
|---:|---:|---:|---:|---:|---:|---:|
| 2022 | 0.24305650 | 0.67890005 | 0.581440 | 0.526837 | 0.528921 | 2451.04 |
| 2023 | 0.24989712 | 0.69294172 | 0.530744 | 0.497732 | 0.499937 | 41.15 |
| 2024 | 0.24758768 | 0.68830153 | 0.553020 | 0.490172 | 0.486112 | 888.46 |

## 판단

- `5-3-4`는 서버 점수가 낮았으므로 이 실험의 주 타깃에서 제외했다.
- 목표는 CatBoost-only 자체의 안정적 개선이며, no calibration은 최종 후보에서 제외했다.
- 개선폭이 작으면 서버 일반화 리스크가 크므로 submit 추천 여부는 `submission_recommended` 값으로 판단한다.
