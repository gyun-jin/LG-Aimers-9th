"""Offline TabICLv2 inference entry point for the evaluation server."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# The submitted model contains its checkpoint weights. Prevent accidental network
# access if Hugging Face environment settings are present on the server.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from tabicl import TabICLClassifier


# classifier.pkl was serialized with NumPy 2.x, whose internal pickle module
# paths moved from ``numpy.core`` to ``numpy._core``. The official server uses
# NumPy 1.26.4, so expose the two stable aliases referenced by this artifact.
# This changes no array data and remains a no-op on NumPy 2.x.
if int(np.__version__.split(".", maxsplit=1)[0]) < 2:
    import numpy.core as _numpy_core
    import numpy.core.multiarray as _numpy_multiarray
    import numpy.core.numeric as _numpy_numeric

    sys.modules["numpy._core"] = _numpy_core
    sys.modules["numpy._core.multiarray"] = _numpy_multiarray
    sys.modules["numpy._core.numeric"] = _numpy_numeric


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
OUTPUT_DIR = ROOT / "output"
ID_COLUMN = "row_id"
TARGET_COLUMN = "control_success"


def prepare_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: set[str],
) -> pd.DataFrame:
    missing = sorted(set(feature_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"test.csv is missing required columns: {missing}")

    features = frame.loc[:, feature_columns].copy()
    for column in feature_columns:
        if column in categorical_columns:
            features[column] = features[column].astype("string").fillna("__MISSING__")
        else:
            features[column] = pd.to_numeric(features[column], errors="coerce").astype(
                "float32"
            )
    return features


def main() -> None:
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    test_frame = pd.read_csv(DATA_DIR / "test.csv")
    features = prepare_features(
        test_frame,
        metadata["feature_columns"],
        set(metadata["categorical_columns"]),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TabICLClassifier.load(MODEL_DIR / "classifier.pkl", device=device)
    # Use server-local paths only. Auto offload keeps CUDA fast and falls back to
    # CPU/disk if the hidden server has less memory than expected.
    offload_dir = MODEL_DIR / "offload"
    offload_dir.mkdir(exist_ok=True)
    model.allow_auto_download = False
    model.offload_mode = "auto"
    model.disk_offload_dir = str(offload_dir)
    model.verbose = True

    probability_matrix = np.asarray(model.predict_proba(features))
    class_matches = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(class_matches) != 1:
        raise ValueError(f"Model classes do not contain binary class 1: {model.classes_}")
    probabilities = np.clip(probability_matrix[:, int(class_matches[0])], 0.0, 1.0)

    predictions = pd.DataFrame(
        {ID_COLUMN: test_frame[ID_COLUMN].astype(str), TARGET_COLUMN: probabilities}
    )
    sample_path = DATA_DIR / "sample_submission.csv"
    sample = pd.read_csv(sample_path, usecols=[ID_COLUMN])
    sample[ID_COLUMN] = sample[ID_COLUMN].astype(str)
    output = sample.merge(predictions, on=ID_COLUMN, how="left", validate="one_to_one")
    if output[TARGET_COLUMN].isna().any():
        raise ValueError("sample_submission.csv row_id values do not match test.csv")

    OUTPUT_DIR.mkdir(exist_ok=True)
    output.to_csv(OUTPUT_DIR / "submission.csv", index=False)
    print(f"Saved {len(output):,} predictions using device={device}")


if __name__ == "__main__":
    main()
