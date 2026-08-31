# Data Quality Summary

The full audit is copied to `reports/DATA_QUALITY_SUMMARY.md`.

## Main Findings

- Pitcher hand inconsistency candidates: `15` IDs
- Batter hand inconsistency or switch candidates: `32` IDs
- Trackman hand mismatch candidates: `724` mapping rows
- Current-season reconstruction anomaly rows: `1,533`
- Game roster extreme candidates: `33`
- Count/state clear error rows: `0`

## Deletion Candidates

The audit identified a narrow hand-minority candidate set of `71` rows across `23` IDs. This was a later diagnostic result. v5-3 itself used the earlier conservative hand-cleaned train file, which removed `116` rows based on Trackman-backed hand mismatch criteria.

## Flag Candidates

The following should generally be flag candidates, not broad deletion candidates:

- possible switch hitters
- low/medium-confidence Trackman mapping mismatches
- current-season reconstruction anomalies
- inferred game roster extremes
- Trackman physical outliers

## What v5-3 Actually Used

v5-3 used:

- the hand-cleaned train file from `hand_cleaning_analysis/train_hand_trackman_clean.csv`
- Trackman pitcher summary features from `mapping_all_shrink`
- current-season reconstruction features

v5-3 did not use the later data-quality flag features from `data_quality_check`. A later v5-6 experiment added those flags and worsened local validation, so v5-3 remains the preferred server model.
