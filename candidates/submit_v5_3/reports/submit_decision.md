# 5-3 제출 판단

## 결론

- 제출 추천 여부: `true`
- 최종 제출 파일: `5-3/submit.zip`
- SHA256: `8a3575f2554dfd584ee2af7b508966423e1c647b7ed130664de5db73945b3133`

## 근거

- `/5-2` mean Brier: `0.24696139`
- `/5-3` mean Brier: `0.24684020`
- `/5-2` worst Brier: `0.24986192`
- `/5-3` worst Brier: `0.24982374`
- test row 독립성 검증: 통과
- submit.zip 무결성 검사: 통과
- 임시 폴더 script.py 실행: 통과

## 최종 구성

- CatBoost weight: `0.8`
- LightGBM weight: `0.2`
- 보정: Platt
- Trackman: prior-season pitcher summary, `mapping_all_shrink`
- 현재 시즌 성공률 복원 피처: 사용
- 학습 데이터: `hand_cleaning_analysis/train_hand_trackman_clean.csv`
