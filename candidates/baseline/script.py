"""Self-contained inference script for this model candidate."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
FEATURE_COLUMNS: list[str] = []  # Set this to the columns used for training.


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the same tabular feature engineering used in train.py."""
    return frame.copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not FEATURE_COLUMNS:
        raise ValueError("Set FEATURE_COLUMNS before using this model candidate.")

    frame = make_features(pd.read_csv(args.input))
    with (ROOT / "model" / "model.pkl").open("rb") as model_file:
        model = pickle.load(model_file)

    pd.DataFrame({"prediction": model.predict(frame[FEATURE_COLUMNS])}).to_csv(
        args.output, index=False
    )


if __name__ == "__main__":
    main()
