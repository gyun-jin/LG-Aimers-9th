# Current-Season Feature Summary

`asof_pitcher_n` and `asof_pitcher_success_rate` are cumulative career-style values. They do not reset at season boundaries.

v5-3 reconstructs current-season progress by subtracting a train-time table of each pitcher's previous season-ending cumulative values from the current row's own `asof_*` values:

```text
career_success_count = asof_pitcher_n * asof_pitcher_success_rate
current_season_n = asof_pitcher_n - prior_season_end_pitcher_n
current_season_success_count = career_success_count - prior_season_end_pitcher_success_count
current_season_success_rate = current_season_success_count / current_season_n
```

The approach uses only:

- the row's own `asof_pitcher_n` and `asof_pitcher_success_rate`
- constants learned from prior training seasons

It does not use other rows from `test.csv`, test ordering, rolling windows, lag features, or any future/current-pitch result. This row-independent use of official `asof_*` fields follows the permitted pattern confirmed in competition Q&A: official `asof_*` fields are precomputed from information available before the current pitch.

The generated feature family includes current-season sample size, log sample size, success count, success rate, smoothed success rate, differences versus career/recent windows, availability flags, small-sample flags, and categorical interactions with count, game type, leverage, base state, hand matchup, and Trackman mapping confidence.
