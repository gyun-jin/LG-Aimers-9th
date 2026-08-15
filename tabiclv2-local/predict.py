"""Load the locally fitted TabICLv2 model and create a submission CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tabicl import TabICLClassifier


ROOT = Path(__file__).resolve().parent
ID_COLUMN = "row_id"
TARGET_COLUMN = "control_success"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT.parent / "data" / "test.csv")
    parser.add_argument("--sample-submission", type=Path, default=ROOT.parent / "data" / "sample_submission.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "model" / "classifier.pkl")
    parser.add_argument("--metadata", type=Path, default=ROOT / "model" / "metadata.json")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "submission.csv")
    return parser.parse_args()


def prepare_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: set[str],
) -> pd.DataFrame:
    missing = sorted(set(feature_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Input CSV is missing feature columns: {missing}")
    features = frame.loc[:, feature_columns].copy()
    for column in feature_columns:
        if column in categorical_columns:
            features[column] = features[column].astype("string").fillna("__MISSING__")
        else:
            features[column] = pd.to_numeric(features[column], errors="coerce").astype("float32")
    return features


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    test_frame = pd.read_csv(args.input)
    features = prepare_features(
        test_frame,
        metadata["feature_columns"],
        set(metadata["categorical_columns"]),
    )

    model = TabICLClassifier.load(str(args.model))
    probability_matrix = np.asarray(model.predict_proba(features))
    class_index = int(np.flatnonzero(np.asarray(model.classes_) == 1)[0])
    predictions = pd.DataFrame(
        {
            ID_COLUMN: test_frame[ID_COLUMN].astype(str),
            TARGET_COLUMN: np.clip(probability_matrix[:, class_index], 0.0, 1.0),
        }
    )

    if args.sample_submission.is_file():
        sample = pd.read_csv(args.sample_submission, usecols=[ID_COLUMN])
        sample[ID_COLUMN] = sample[ID_COLUMN].astype(str)
        predictions = sample.merge(
            predictions, on=ID_COLUMN, how="left", validate="one_to_one"
        )
        if predictions[TARGET_COLUMN].isna().any():
            raise ValueError("sample_submission.csv and input row_id values do not match")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"Saved {len(predictions):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
