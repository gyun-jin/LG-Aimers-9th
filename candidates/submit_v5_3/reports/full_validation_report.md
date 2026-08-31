# 5-3 full validation 요약

## 결과

- mean Brier: `0.24684020`
- worst Brier: `0.24982374`
- `/5-2` mean Brier: `0.24696139`
- `/5-2` worst Brier: `0.24986192`
- 제출 추천 여부: `true`

## fold별 결과

| season | raw Brier | calibrated Brier | AUC |
|---:|---:|---:|---:|
| 2022 | 0.24311064 | 0.24309218 | 0.581083 |
| 2023 | 0.24983641 | 0.24982374 | 0.533098 |
| 2024 | 0.24769710 | 0.24760468 | 0.552749 |

세부 원본 결과는 `output/submit_build_report.json`에 저장했다.
