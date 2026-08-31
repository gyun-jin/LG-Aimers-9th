# 5-3 실험 보고서

## 결론

- 기준 모델: `/5-2`
- `/5-2` validation mean Brier: `0.24696139`
- `/5-2` validation worst Brier: `0.24986192`
- `/5-2` 실제 서버 점수: `957.6800554874`
- `/5-3` validation mean Brier: `0.24684020`
- `/5-3` validation worst Brier: `0.24982374`
- mean Brier 차이: `-0.00012119`
- 제출 추천 여부: `true`

`/5-3`은 hand 정제 train, 현재 시즌 성공률 복원 피처, `/5-2`와 같은 Trackman prior-season 투수 요약 피처를 사용했다. full 3-fold 기준 mean Brier와 worst Brier가 모두 `/5-2`보다 낮아졌다.

## 최종 모델

- CatBoost weight: `0.8`
- LightGBM weight: `0.2`
- 보정 방식: Platt 보정
- Trackman 전략: `mapping_all_shrink`
- Trackman 피처셋: `tm_metadata_physical_pitchmix`
- 최종 피처 수: `138`
- 학습 데이터: `hand_cleaning_analysis/train_hand_trackman_clean.csv`

## full validation 결과

| season | raw Brier | calibrated Brier | AUC |
|---:|---:|---:|---:|
| 2022 | 0.24311064 | 0.24309218 | 0.581083 |
| 2023 | 0.24983641 | 0.24982374 | 0.533098 |
| 2024 | 0.24769710 | 0.24760468 | 0.552749 |

## 정제 train 사용 효과

`/5-2`는 원본 `data/train.csv`를 사용했고, `/5-3`은 `train_hand_trackman_clean.csv`를 사용했다. 동시에 현재 시즌 성공률 피처도 추가되었기 때문에 full 결과는 두 변경의 결합 효과다.

- `/5-2` mean Brier: `0.24696139`
- `/5-3` mean Brier: `0.24684020`
- 개선폭: `-0.00012119`

## 현재 시즌 성공률 피처 효과

현재 시즌 피처를 켠 `/5-3` 최종 모델은 `/5-2`보다 mean/worst Brier가 모두 개선됐다. quick ablation에서 현재 시즌 피처를 포함한 no-Trackman 후보는 mean Brier `0.24696972`, worst Brier `0.24973173`이었다.

이 결과는 현재 시즌 성공률 복원 피처가 기존 누적/최근 피처와 별도의 신호를 일부 제공하지만, full 최종 선택에서는 Trackman까지 포함한 모델이 더 안정적인 제출 후보라는 판단으로 이어졌다.

## Trackman 효과

quick ablation 기준:

| feature set | selected calibration | mean Brier | worst Brier |
|---|---|---:|---:|
| `v5_no_trackman_recheck` | Platt | 0.24696972 | 0.24973173 |
| `tm_metadata_only` | prior correction | 0.24700277 | 0.24982940 |
| `tm_metadata_pitchmix` | Platt | 0.24698750 | 0.24981527 |
| `tm_metadata_physical_core` | Platt | 0.24698188 | 0.24980606 |
| `tm_metadata_physical_pitchmix` | prior correction | 0.24698039 | 0.24985762 |

quick sample에서는 no-Trackman 후보가 가장 좋았다. 다만 full 최종 학습은 `/5-2`와 같은 `tm_metadata_physical_pitchmix`를 유지했고, full 3-fold 기준으로 `/5-2`보다 개선됐다. 서버 점수는 실제 제출 전 알 수 없으므로, Trackman 유지가 서버에서 다시 유리할 가능성과 quick ablation의 경고를 함께 봐야 한다.

## 모델별 quick 비교

선택 피처셋 `tm_metadata_physical_pitchmix` 기준 quick sample 비교:

| model | mean Brier | worst Brier |
|---|---:|---:|
| CatBoost | 0.24708633 | 0.24977960 |
| LightGBM | 0.24855746 | 0.25267830 |
| Ensemble 0.8/0.2 | 0.24708110 | 0.24977163 |

LightGBM 단독은 약했고, CatBoost 중심 ensemble이 가장 안정적이었다.

## 제출 판단

- `submission_recommended=true`
- 이유: full 3-fold mean Brier와 worst Brier가 `/5-2`보다 개선됐다.
- 주의: quick ablation에서는 no-Trackman 후보가 더 좋았으므로 서버 제출 후 점수 확인이 필요하다.
