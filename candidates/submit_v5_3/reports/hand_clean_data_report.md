# hand 정제 train 사용 보고서

## 사용 데이터

- 원본 train: `data/train.csv`
- 정제 train: `hand_cleaning_analysis/train_hand_trackman_clean.csv`
- 정제 report: `hand_cleaning_analysis/hand_trackman_cleaning_report.md`

## 행 수

- 원본 train 행 수: `1475092`
- 제거된 pitcher hand mismatch 행 수: `77`
- 제거된 batter hand mismatch 행 수: `39`
- 총 제거 행 수: `116`
- 정제 train 행 수: `1474976`

## hand code 검증

`hand_cleaning_analysis`에서 Trackman majority hand와 비교해 train hand code를 다시 검증했다.

- `1=Left`
- `2=Right`
- `1=Left, 2=Right` 후보 일치율: `0.994737`
- 반대 후보 일치율: `0.005263`

## 제거 기준 요약

투수는 기존 `pitcher_id_mapping_clean.csv`를 사용했다. Trackman hand 표본이 충분하고, Trackman majority hand와 train majority hand가 일치하는 명확한 케이스에서만 mismatch row를 제거했다.

타자는 새 `batter_id_mapping_clean.csv`의 medium 이상 매핑만 사용했다. 스위치 히터 가능성을 고려해 train과 Trackman 모두 majority rate `0.98` 이상인 명확한 hand 오입력만 제거했다.

## 남은 mixed-hand ID

정제 summary 기준 decision 분포:

투수:

- `remove_mismatch_rows`: 756명
- `keep_train_tm_majority_conflict`: 2명
- `keep_train_hand_mixed`: 1명
- `keep_tm_hand_mixed`: 1명

타자:

- `remove_mismatch_rows`: 539명
- `keep_mapping_not_medium`: 180명
- `keep_possible_switch_or_tm_mixed`: 15명
- `keep_possible_switch_or_train_mixed`: 1명

애매한 투수/타자는 실제 스위치 히터, 표기 혼재, 매핑 불확실성 가능성이 있어 제거하지 않았다.
