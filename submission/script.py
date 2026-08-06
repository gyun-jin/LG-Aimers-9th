"""Final submission inference script."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
FEATURE_COLUMNS: list[str] = []  # Set this when selecting the final model.


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Paste the selected model's finalized feature engineering here."""
    return frame.copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not FEATURE_COLUMNS:
        raise ValueError("Set FEATURE_COLUMNS before submission.")

    frame = make_features(pd.read_csv(args.input))
    with (ROOT / "model" / "model.pkl").open("rb") as model_file:
        model = pickle.load(model_file)

    pd.DataFrame({"prediction": model.predict(frame[FEATURE_COLUMNS])}).to_csv(
        args.output, index=False
    )


if __name__ == "__main__":
    main()
