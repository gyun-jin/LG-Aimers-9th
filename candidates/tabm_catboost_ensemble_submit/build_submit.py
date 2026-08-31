"""Assemble trained artifacts and create the 95:5 submission archive."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = ROOT / "candidates" / "tabm_catboost_ensemble"
CATBOOST = ROOT / "candidates" / "submit_v5_8"
TABM = ROOT / "candidates" / "tabm_v1"
MODEL = HERE / "model"
CATBOOST_WEIGHT = 0.95
TABM_WEIGHT = 0.05
FOLDS = (2022, 2023, 2024)


def fit_calibrator() -> LogisticRegression:
    labels, predictions = [], []
    for season in FOLDS:
        cat = np.load(EXPERIMENT / "output" / f"catboost_oof_{season}.npz")
        tabm = np.load(EXPERIMENT / "output" / f"tabm_oof_{season}.npz")
        if not np.array_equal(cat["row_id"].astype(str), tabm["row_id"].astype(str)):
            raise ValueError(f"row_id mismatch for {season}")
        if not np.array_equal(cat["y_true"], tabm["y_true"]):
            raise ValueError(f"target mismatch for {season}")
        labels.append(cat["y_true"].astype(np.int8))
        predictions.append(
            CATBOOST_WEIGHT * cat["raw_pred"].astype(np.float64)
            + TABM_WEIGHT * tabm["raw_pred"].astype(np.float64)
        )
    y = np.concatenate(labels)
    probability = np.clip(np.concatenate(predictions), 1e-6, 1.0 - 1e-6)
    logits = np.log(probability / (1.0 - probability)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=42)
    calibrator.fit(logits, y)
    return calibrator


def main() -> None:
    required = [
        TABM / "model" / "tabm.pt",
        TABM / "model" / "preprocessor.json",
        CATBOOST / "model" / "final_model.joblib",
        CATBOOST / "model" / "feature_schema.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing trained artifacts: {missing}")
    MODEL.mkdir(parents=True, exist_ok=True)
    copies = {
        CATBOOST / "model" / "final_model.joblib": MODEL / "catboost_final_model.joblib",
        CATBOOST / "model" / "feature_schema.json": MODEL / "catboost_feature_schema.json",
        CATBOOST / "script.py": MODEL / "catboost_inference.py",
        TABM / "model" / "tabm.pt": MODEL / "tabm.pt",
        TABM / "model" / "preprocessor.json": MODEL / "tabm_preprocessor.json",
        TABM / "common.py": MODEL / "tabm_common.py",
        TABM / "tabm_model.py": MODEL / "tabm_model.py",
    }
    for source, destination in copies.items():
        shutil.copy2(source, destination)
    joblib.dump(fit_calibrator(), MODEL / "ensemble_calibrator.joblib")
    archive_path = HERE / "submit.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(HERE / "script.py", "script.py")
        archive.write(HERE / "requirements.txt", "requirements.txt")
        for path in sorted(MODEL.iterdir()):
            if path.is_file():
                archive.write(path, f"model/{path.name}")
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC failure: {bad}")
    print(f"created={archive_path} size={archive_path.stat().st_size:,}")


if __name__ == "__main__":
    main()
