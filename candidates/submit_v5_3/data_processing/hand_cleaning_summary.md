# Hand Cleaning Summary

## Source

- Original train: `data/train.csv`
- Cleaned train used by v5-3: `hand_cleaning_analysis/train_hand_trackman_clean.csv`
- The cleaned train file is not committed because it is a large derivative of the official train data.

## Row Counts

- Original train rows: `1,475,092`
- Cleaned train rows: `1,474,976`
- Total removed rows: `116`
- Removed pitcher hand mismatch rows: `77`
- Removed batter hand mismatch rows: `39`

## Criteria

The cleaning step used Trackman hand majority and conservative train hand majority checks. Rows were removed only when the mapped Trackman hand and the train majority hand were sufficiently consistent and the row's hand value contradicted that majority.

Mixed or uncertain IDs were kept when the case could be a true switch hitter, mixed Trackman record, low-confidence mapping, or ambiguous mapping. These are better handled as investigation/flag candidates than broad deletion candidates.

## Remaining Mixed Or Uncertain IDs

The detailed decisions are documented in `reports/hand_clean_data_report.md` and in the source analysis directory outside this shared package. Summary distributions:

- Pitcher: most mapped pitchers were `remove_mismatch_rows`; a small number were kept as train/Trackman majority conflicts or mixed-hand cases.
- Batter: many uncertain cases were kept because switch hitting is plausible and deletion can remove valid behavior.

## Reproduction Note

Place official `train.csv` and `trackman_history.csv` in the expected project data paths, then run the hand cleaning analysis scripts from the original worktree if exact regeneration is required. This package documents the cleaned data but intentionally does not include the cleaned train CSV.
