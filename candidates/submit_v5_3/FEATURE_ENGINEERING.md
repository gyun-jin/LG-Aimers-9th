# FEATURE ENGINEERING

## Original Features

The model starts from official train/test columns: season, month, day of week, inning, top/bottom, game type, count, score, base runners, win expectancy, leverage, pitcher/batter IDs, hands, and teams.

## asof Features

Official `asof_*` pitcher and batter rates are smoothed with train priors. Missing indicators are added for rate columns. Recent 1/3/5 game success and middle-rate summaries are converted into mean, range, standard deviation, and gap features.

## Current-Season Features

The model reconstructs current-season pitcher progress from career cumulative `asof_pitcher_n` and `asof_pitcher_success_rate` by subtracting train-time prior season-end constants.

## Count Features

`count_state` combines balls and strikes. Existing v5-3 features include count state and pressure state. These help capture different control expectations in hitter-friendly or pitcher-friendly counts.

## Runner/Base Features

`base_state`, runner indicators, runner/out state, close-game flag, late-inning flag, leverage, and pressure state represent the tactical context of each pitch.

## Hand Features

`hand_matchup` combines pitcher and batter hand codes. Hand-cleaned train data reduces obvious hand label noise.

## Game Type

`game_type` is used as a categorical input and also interacts with current-season success bins.

## Player/Team Features

Pitcher, batter, pitcher team, and batter team IDs are part of the saved feature schema before no-ID variant handling. The selected v5-3 training path uses the established v5 no-ID ensemble behavior where applicable.

## Trackman Features

Trackman features are prior-season pitcher summaries, not current-pitch measurements. Groups include:

- mapping metadata and confidence
- historical pitch count, game count, season count
- shrunk release speed, spin, release position, extension
- pitch group mix rates
- pitch type entropy

`mapping_all_shrink` shrinks pitcher-specific Trackman summaries toward prior hand/season aggregates to reduce low-sample and mapping-noise risk.
