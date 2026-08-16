import numpy as np
import pandas as pd

TARGET_COL = "control_success"

TABM_TRAIN_V1_CATEGORICAL = [
    "game_month","game_dayofweek","inning","top_bottom","game_type",
    "balls_before","strikes_before","outs_before","base_state",
    "pitcher_hand","batter_hand","pitcher_team_id","batter_team_id",
]

TABM_TRAIN_V1_BINARY = [
    "runner_on_1b","runner_on_2b","runner_on_3b",
    "pitcher_recent_history_missing","pitcher_history_missing",
    "batter_history_missing",
    "pitcher_current_season_available_flag",
    "pitcher_current_season_small_sample_flag",
]

TABM_TRAIN_V1_NUMERIC = [
    "run_top_before","run_bot_before","run_total_before",
    "score_diff_home","score_diff_pitcher_team","num_runners_on",
    "home_win_expectancy","li",
    "asof_pitcher_n","asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate","asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate","asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_batter_n","asof_batter_success_rate","asof_batter_middle_rate",
    "asof_pitcher_fastball_rate","asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    "log1p_asof_pitcher_n","log1p_asof_batter_n","log1p_li",
    "pitcher_current_season_success_rate",
    "pitcher_current_season_success_rate_smoothed",
    "pitcher_current_season_n",
    "pitcher_current_season_success_count",
    "log1p_pitcher_current_season_n",
    "pitcher_current_season_n_ratio_to_career",
]

TABM_TRAIN_V1_FEATURES = (
    TABM_TRAIN_V1_NUMERIC
    + TABM_TRAIN_V1_BINARY
    + TABM_TRAIN_V1_CATEGORICAL
)

def build_tabm_train_v1(train, small_sample_threshold=30, smoothing_strength=50):
    df = train.copy()

    df["pitcher_recent_history_missing"] = (
        df["asof_pitcher_prev1_game_success_rate"].isna()
    ).astype("int8")
    df["pitcher_history_missing"] = (
        df["asof_pitcher_success_rate"].isna()
    ).astype("int8")
    df["batter_history_missing"] = (
        df["asof_batter_success_rate"].isna()
    ).astype("int8")

    df["log1p_asof_pitcher_n"] = np.log1p(df["asof_pitcher_n"])
    df["log1p_asof_batter_n"] = np.log1p(df["asof_batter_n"])
    df["log1p_li"] = np.log1p(df["li"])

    df["asof_pitcher_success_count"] = (
        df["asof_pitcher_n"] * df["asof_pitcher_success_rate"]
    ).round()

    season_start = (
        df.sort_values(["pitcher_id","season","asof_pitcher_n"])
          .groupby(["pitcher_id","season"], as_index=False)
          .first()[[
              "pitcher_id","season","asof_pitcher_n",
              "asof_pitcher_success_count"
          ]]
          .rename(columns={
              "asof_pitcher_n":"prior_season_end_pitcher_n",
              "asof_pitcher_success_count":
                  "prior_season_end_pitcher_success_count",
          })
    )

    df = df.merge(
        season_start,
        on=["pitcher_id","season"],
        how="left",
        validate="many_to_one",
    )

    df["pitcher_current_season_n"] = (
        df["asof_pitcher_n"] - df["prior_season_end_pitcher_n"]
    )
    df["pitcher_current_season_success_count"] = (
        df["asof_pitcher_success_count"]
        - df["prior_season_end_pitcher_success_count"]
    )
    df["pitcher_current_season_success_rate"] = np.where(
        df["pitcher_current_season_n"] > 0,
        df["pitcher_current_season_success_count"]
        / df["pitcher_current_season_n"],
        np.nan,
    )
    df["log1p_pitcher_current_season_n"] = np.log1p(
        df["pitcher_current_season_n"]
    )
    df["pitcher_current_season_available_flag"] = (
        df["pitcher_current_season_n"] > 0
    ).astype("int8")
    df["pitcher_current_season_small_sample_flag"] = (
        df["pitcher_current_season_n"]
        .between(1, small_sample_threshold - 1)
    ).astype("int8")

    df["pitcher_current_season_success_rate_smoothed"] = (
        df["pitcher_current_season_success_count"]
        + smoothing_strength * df["asof_pitcher_success_rate"]
    ) / (
        df["pitcher_current_season_n"] + smoothing_strength
    )

    df["pitcher_current_season_n_ratio_to_career"] = (
        df["pitcher_current_season_n"]
        / (df["asof_pitcher_n"] + 1)
    )

    missing = [c for c in TABM_TRAIN_V1_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing TabM v1 features: {missing}")

    if len(TABM_TRAIN_V1_FEATURES) != 53:
        raise ValueError(
            f"Expected 53 features, got {len(TABM_TRAIN_V1_FEATURES)}"
        )

    return df
