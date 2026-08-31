# SCORE REPORT

## Version

- Version: submit v5-3
- Server score: `1014.5598924165`

## Local Validation

| season | calibrated Brier | AUC |
|---:|---:|---:|
| 2022 | 0.24309218 | 0.581083 |
| 2023 | 0.24982374 | 0.533098 |
| 2024 | 0.24760468 | 0.552749 |

- Mean Brier: `0.24684020`
- Worst Brier: `0.24982374`

## Improvements Over v5-2

v5-3 adds hand-cleaned train data and current-season pitcher success reconstruction while keeping the v5-2 Trackman and ensemble structure. The changes improved full 3-fold mean Brier versus v5-2 and produced the best observed server score.

## v5-4/v5-5 Comparison

v5-4 and v5-5 showed small local Brier improvements in some experiments, but their server scores were lower than v5-3. This suggests the more aggressive later adjustments had weaker leaderboard generalization.

## Why It Helped

Hand cleaning removes obvious label noise in pitcher/batter handedness, which matters for hand matchup and interaction features. Current-season reconstruction lets the model adapt to season-level pitcher form while remaining row-independent and competition-compliant.

## Limitations

The 2023 fold is close to a baseline Brier regime and remains hard. Trackman features are useful in the selected server model but have mixed local ablation signals.

## Next Directions

- Keep v5-3 as the primary reference
- Use AutoInt, TabM, or FT-Transformer mainly as ensemble candidates
- Test v5-3 plus a small 10-3 AutoInt ensemble
- Improve calibration, stacking, or gating without using test-internal statistics
