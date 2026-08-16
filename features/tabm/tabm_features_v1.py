"""
LG Aimers - TabM Feature Set v1 (train + test)
================================================

Final feature set: B_Lean
- Numeric: 32
- Binary: 8
- Categorical: 13
- Total: 53

Designed for the competition execution setup:
- Public train.csv: through 2024, includes control_success
- Distributed test.csv: only 5 sample rows
- Evaluation server replaces test.csv with hidden 2025 evaluation rows
  using the same schema.

Important:
Current-season test features do NOT infer the 2025 season start from the
test rows. The 2025 starting career state is derived from the final observed
state in public train.csv. This makes the feature builder usable on both
the 5-row sample and the hidden evaluation test.
"""

import numpy as np
import pandas as pd


TARGET_COL = "control_success"


TABM_V1_CATEGORICAL = [
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]


TABM_V1_BINARY = [
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "pitcher_recent_history_missing",
    "pitcher_history_missing",
    "batter_history_missing",
    "pitcher_current_season_available_flag",
    "pitcher_current_season_small_sample_flag",
]


TABM_V1_NUMERIC = [
    # Game / score context
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "num_runners_on",
    "home_win_expectancy",
    "li",

    # Pitcher career / as-of level
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",

    # Recent success raw level
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",

    # Batter history
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",

    # Pitch mix
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",

    # Baseline reliability / transformed values
    "log1p_asof_pitcher_n",
    "log1p_asof_batter_n",
    "log1p_li",

    # Current-season Level
    "pitcher_current_season_success_rate",
    "pitcher_current_season_success_rate_smoothed",

    # Current-season Reliability
    "pitcher_current_season_n",
    "pitcher_current_season_success_count",
    "log1p_pitcher_current_season_n",
    "pitcher_current_season_n_ratio_to_career",
]


TABM_V1_FEATURES = (
    TABM_V1_NUMERIC
    + TABM_V1_BINARY
    + TABM_V1_CATEGORICAL
)


# Backward-compatible aliases for the previously shared train-only module.
TABM_TRAIN_V1_NUMERIC = TABM_V1_NUMERIC
TABM_TRAIN_V1_BINARY = TABM_V1_BINARY
TABM_TRAIN_V1_CATEGORICAL = TABM_V1_CATEGORICAL
TABM_TRAIN_V1_FEATURES = TABM_V1_FEATURES


def _add_common_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that can be computed row-wise for both train and test."""
    out = df.copy()

    out["pitcher_recent_history_missing"] = (
        out["asof_pitcher_prev1_game_success_rate"].isna()
    ).astype("int8")

    out["pitcher_history_missing"] = (
        out["asof_pitcher_success_rate"].isna()
    ).astype("int8")

    out["batter_history_missing"] = (
        out["asof_batter_success_rate"].isna()
    ).astype("int8")

    out["log1p_asof_pitcher_n"] = np.log1p(out["asof_pitcher_n"])
    out["log1p_asof_batter_n"] = np.log1p(out["asof_batter_n"])
    out["log1p_li"] = np.log1p(out["li"])

    # Count BEFORE the current pitch.
    out["asof_pitcher_success_count"] = np.where(
        out["asof_pitcher_n"].eq(0),
        0.0,
        np.rint(
            out["asof_pitcher_n"]
            * out["asof_pitcher_success_rate"]
        ),
    )

    return out


def _add_current_season_from_anchor(
    df: pd.DataFrame,
    anchor_n_col: str,
    anchor_success_col: str,
    small_sample_threshold: int = 30,
    smoothing_strength: int = 50,
) -> pd.DataFrame:
    """Create final Level + Reliability current-season features."""
    out = df.copy()

    out["pitcher_current_season_n"] = (
        out["asof_pitcher_n"] - out[anchor_n_col]
    )

    out["pitcher_current_season_success_count"] = (
        out["asof_pitcher_success_count"] - out[anchor_success_col]
    )

    valid_n = (
        out["pitcher_current_season_n"].notna()
        & out["pitcher_current_season_n"].gt(0)
    )

    out["pitcher_current_season_success_rate"] = np.where(
        valid_n,
        (
            out["pitcher_current_season_success_count"]
            / out["pitcher_current_season_n"]
        ),
        np.nan,
    )

    # log1p is only meaningful for known non-negative counts.
    out["log1p_pitcher_current_season_n"] = np.where(
        out["pitcher_current_season_n"].ge(0),
        np.log1p(out["pitcher_current_season_n"]),
        np.nan,
    )

    out["pitcher_current_season_available_flag"] = (
        out[anchor_n_col].notna()
        & out["pitcher_current_season_n"].ge(0)
    ).astype("int8")

    out["pitcher_current_season_small_sample_flag"] = (
        out["pitcher_current_season_n"]
        .between(1, small_sample_threshold - 1)
        .fillna(False)
    ).astype("int8")

    out["pitcher_current_season_success_rate_smoothed"] = np.where(
        out["pitcher_current_season_available_flag"].eq(1),
        (
            out["pitcher_current_season_success_count"]
            + smoothing_strength * out["asof_pitcher_success_rate"]
        ) / (
            out["pitcher_current_season_n"]
            + smoothing_strength
        ),
        np.nan,
    )

    out["pitcher_current_season_n_ratio_to_career"] = np.where(
        out["pitcher_current_season_available_flag"].eq(1),
        (
            out["pitcher_current_season_n"]
            / (out["asof_pitcher_n"] + 1)
        ),
        np.nan,
    )

    return out


def build_tabm_train_v1(
    train: pd.DataFrame,
    small_sample_threshold: int = 30,
    smoothing_strength: int = 50,
) -> pd.DataFrame:
    """
    Build the finalized TabM v1 training features.

    For each pitcher-season, the first observed pre-pitch career state is used
    as that season's anchor, matching the feature logic used during validation.
    """
    df = _add_common_features(train)

    season_anchor = (
        df
        .sort_values(["pitcher_id", "season", "asof_pitcher_n"])
        .groupby(["pitcher_id", "season"], as_index=False)
        .first()[
            [
                "pitcher_id",
                "season",
                "asof_pitcher_n",
                "asof_pitcher_success_count",
            ]
        ]
        .rename(
            columns={
                "asof_pitcher_n": "_season_anchor_n",
                "asof_pitcher_success_count":
                    "_season_anchor_success_count",
            }
        )
    )

    df = df.merge(
        season_anchor,
        on=["pitcher_id", "season"],
        how="left",
        validate="many_to_one",
    )

    df = _add_current_season_from_anchor(
        df,
        "_season_anchor_n",
        "_season_anchor_success_count",
        small_sample_threshold=small_sample_threshold,
        smoothing_strength=smoothing_strength,
    )

    validate_tabm_v1(df, dataset_name="train")
    return df


def build_2025_pitcher_anchor_from_train(
    train: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the career state immediately AFTER each pitcher's final public-train
    pitch. This becomes the 2025 season-start anchor.

    asof_pitcher_n / success_rate describe the state BEFORE a train pitch, so:
    - 2025 anchor_n = final pre-pitch n + 1
    - 2025 anchor_success_count = final pre-pitch success count
                                  + final pitch target
    """
    if TARGET_COL not in train.columns:
        raise ValueError(
            f"Public train must contain {TARGET_COL!r} "
            "to build the 2025 test anchor."
        )

    df = _add_common_features(train)

    # The row with the largest asof_pitcher_n is the latest observed pitch
    # for that pitcher because the count increases monotonically by pitch.
    idx = (
        df.groupby("pitcher_id")["asof_pitcher_n"]
        .idxmax()
    )

    last = df.loc[
        idx,
        [
            "pitcher_id",
            "asof_pitcher_n",
            "asof_pitcher_success_count",
            TARGET_COL,
        ],
    ].copy()

    last["_season_anchor_n"] = (
        last["asof_pitcher_n"] + 1
    )

    last["_season_anchor_success_count"] = (
        last["asof_pitcher_success_count"]
        + last[TARGET_COL].astype(float)
    )

    return last[
        [
            "pitcher_id",
            "_season_anchor_n",
            "_season_anchor_success_count",
        ]
    ].reset_index(drop=True)


def build_tabm_test_v1(
    train: pd.DataFrame,
    test: pd.DataFrame,
    small_sample_threshold: int = 30,
    smoothing_strength: int = 50,
) -> pd.DataFrame:
    """
    Build TabM v1 features for the distributed 5-row test sample OR the hidden
    2025 evaluation test.

    The test rows themselves are not used to infer a season-start anchor.
    Instead, each pitcher's final state in public train is used.

    For pitchers unseen in public train, current-season numeric features remain
    NaN and the availability flag is 0. The existing preprocessing imputer can
    handle these NaNs without inventing an unsafe test-derived history.
    """
    df = _add_common_features(test)

    anchor = build_2025_pitcher_anchor_from_train(train)

    df = df.merge(
        anchor,
        on="pitcher_id",
        how="left",
        validate="many_to_one",
    )

    # A test pitcher with exactly zero prior career pitches is safely known to
    # have a zero anchor even if absent from public train.
    zero_new = (
        df["_season_anchor_n"].isna()
        & df["asof_pitcher_n"].eq(0)
    )

    df.loc[zero_new, "_season_anchor_n"] = 0.0
    df.loc[zero_new, "_season_anchor_success_count"] = 0.0

    df = _add_current_season_from_anchor(
        df,
        "_season_anchor_n",
        "_season_anchor_success_count",
        small_sample_threshold=small_sample_threshold,
        smoothing_strength=smoothing_strength,
    )

    validate_tabm_v1(df, dataset_name="test")
    return df


def validate_tabm_v1(
    df: pd.DataFrame,
    dataset_name: str = "data",
) -> None:
    """Structural and sanity checks for the final 53-feature schema."""
    missing = [
        c for c in TABM_V1_FEATURES
        if c not in df.columns
    ]
    if missing:
        raise ValueError(
            f"{dataset_name}: missing TabM v1 features: {missing}"
        )

    if len(TABM_V1_FEATURES) != 53:
        raise ValueError(
            f"Expected 53 TabM v1 features, got {len(TABM_V1_FEATURES)}."
        )

    duplicates = (
        pd.Series(TABM_V1_FEATURES)
        .value_counts()
    )
    duplicates = duplicates[duplicates > 1]
    if len(duplicates):
        raise ValueError(
            f"Duplicated TabM v1 features: {duplicates.to_dict()}"
        )

    available = (
        df["pitcher_current_season_available_flag"].eq(1)
    )

    bad_n = (
        available
        & df["pitcher_current_season_n"].lt(0)
    )
    if bad_n.any():
        raise ValueError(
            f"{dataset_name}: negative current-season n "
            f"in {int(bad_n.sum())} rows."
        )

    bad_success = (
        available
        & (
            df["pitcher_current_season_success_count"].lt(0)
            | (
                df["pitcher_current_season_success_count"]
                > df["pitcher_current_season_n"]
            )
        )
    )
    if bad_success.any():
        raise ValueError(
            f"{dataset_name}: invalid current-season success count "
            f"in {int(bad_success.sum())} rows."
        )

    numeric_values = df[TABM_V1_NUMERIC].select_dtypes(
        include=["number"]
    )
    if np.isinf(numeric_values.to_numpy()).any():
        raise ValueError(
            f"{dataset_name}: infinite value found in numeric features."
        )


def get_tabm_v1_feature_groups():
    return {
        "numeric": TABM_V1_NUMERIC.copy(),
        "binary": TABM_V1_BINARY.copy(),
        "categorical": TABM_V1_CATEGORICAL.copy(),
    }


def select_tabm_v1_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the final 53 model-input columns in fixed order."""
    return df[TABM_V1_FEATURES].copy()
