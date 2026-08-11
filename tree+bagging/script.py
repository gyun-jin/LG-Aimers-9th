"""Standalone submission inference for the Tree + Bagging model."""

from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

# The evaluation server allows at most six CPU threads.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "6"

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
SOURCE_COLUMNS = {
    "row_id",
    "season",
    "game_month",
    "game_type",
    "li",
    "asof_pitcher_n",
    "asof_pitcher_pitchmix_n",
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
}


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Mirror the leakage-safe feature engineering in train.py exactly."""
    missing = sorted(SOURCE_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required test columns: {missing}")
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


def find_input(explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Input CSV not found: {path}")
        return path

    candidates = [
        ROOT / "open" / "test.csv",
        ROOT / "data" / "test.csv",
        Path.cwd() / "open" / "test.csv",
        Path.cwd() / "data" / "test.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    checked = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(f"Could not find test.csv. Checked:\n{checked}")


def find_model() -> Path:
    """Locate model.pkl even if the evaluator flattens the model directory."""
    candidates = [ROOT / "model" / "model.pkl", ROOT / "model.pkl"]
    for path in candidates:
        if path.is_file():
            return path

    discovered = sorted(path for path in ROOT.rglob("model.pkl") if path.is_file())
    if len(discovered) == 1:
        return discovered[0]

    packaged_files = sorted(
        str(path.relative_to(ROOT)) for path in ROOT.rglob("*") if path.is_file()
    )
    raise FileNotFoundError(
        "Could not locate model.pkl in the extracted submission. "
        f"Packaged files: {packaged_files}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Optional test.csv path")
    parser.add_argument("--output", default=None, help="Optional submission.csv path")
    args = parser.parse_args()

    started = time.perf_counter()
    input_path = find_input(args.input)
    output_path = Path(args.output).resolve() if args.output else ROOT / "output" / "submission.csv"

    read_started = time.perf_counter()
    frame = pd.read_csv(input_path)
    if frame["row_id"].isna().any() or frame["row_id"].duplicated().any():
        raise ValueError("row_id must be non-null and unique.")
    features = make_features(frame)
    print(f"[1/4] Loaded {len(frame):,} rows in {time.perf_counter() - read_started:.3f}s", flush=True)

    load_started = time.perf_counter()
    model_path = find_model()
    with model_path.open("rb") as model_file:
        artifact = pickle.load(model_file)
    if artifact["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("model.pkl and script.py use different feature definitions.")
    model = artifact["model"]
    model.named_steps["model"].n_jobs = 6
    print(
        f"[2/4] Loaded {model_path.relative_to(ROOT)} in "
        f"{time.perf_counter() - load_started:.3f}s",
        flush=True,
    )

    predict_started = time.perf_counter()
    probability = np.clip(model.predict_proba(features)[:, 1], 0.0, 1.0)
    if not np.isfinite(probability).all():
        raise ValueError("Predictions contain NaN or infinity.")
    print(f"[3/4] Predicted in {time.perf_counter() - predict_started:.3f}s", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(
        {"row_id": frame["row_id"].to_numpy(), "control_success": probability}
    )
    submission.to_csv(output_path, index=False)
    print(
        f"[4/4] Saved {output_path} in {time.perf_counter() - started:.3f}s total",
        flush=True,
    )


if __name__ == "__main__":
    main()
