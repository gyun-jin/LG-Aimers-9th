"""Submission inference for the calibrated Tree + Bagging model."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FEATURE_COLUMNS = [
    "season",
    "game_month",
    "game_type",
    "li",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    "success_recent_1_5_gap",
    "success_delta_3",
    "middle_delta_3",
    "pitcher_log_n",
    "pitchmix_log_n",
]


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Mirror the leakage-safe feature engineering in train.py exactly."""
    features = frame.copy()
    features["success_recent_1_5_gap"] = (
        features["asof_pitcher_prev1_game_success_rate"]
        - features["asof_pitcher_prev5_game_success_rate"]
    )
    features["success_delta_3"] = (
        features["asof_pitcher_prev3_game_success_rate"]
        - features["asof_pitcher_success_rate"]
    )
    features["middle_delta_3"] = (
        features["asof_pitcher_prev3_game_middle_rate"]
        - features["asof_pitcher_middle_rate"]
    )
    features["pitcher_log_n"] = np.log1p(features["asof_pitcher_n"].clip(lower=0))
    features["pitchmix_log_n"] = np.log1p(
        features["asof_pitcher_pitchmix_n"].clip(lower=0)
    )
    return features[FEATURE_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    features = make_features(frame)
    with (ROOT / "model" / "model.pkl").open("rb") as model_file:
        artifact = pickle.load(model_file)

    if artifact["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("model.pkl and script.py use different feature definitions.")
    raw_probability = artifact["model"].predict_proba(features)[:, 1]
    probability = artifact["calibrator"].predict_proba(raw_probability.reshape(-1, 1))[:, 1]
    pd.DataFrame({"prediction": probability}).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
