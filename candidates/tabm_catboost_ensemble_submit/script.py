"""Inference for the conservative 95% CatBoost + 5% TabM probe."""

from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch

from model import catboost_inference as cat
from model.tabm_common import Preprocessor
from model.tabm_model import make_model


ID_COL = "row_id"
TARGET_COL = "control_success"
CATBOOST_WEIGHT = 0.95
TABM_WEIGHT = 0.05


def resolve_paths():
    checked = []
    for directory in ("./open", "./data"):
        test_path = os.path.join(directory, "test.csv")
        sample_path = os.path.join(directory, "sample_submission.csv")
        checked.extend([test_path, sample_path])
        if os.path.isfile(test_path) and os.path.isfile(sample_path):
            return test_path, sample_path
    raise FileNotFoundError(f"input files not found: {checked}")


def predict_catboost(test: pd.DataFrame) -> np.ndarray:
    bundle = joblib.load("./model/catboost_final_model.joblib")
    with open("./model/catboost_feature_schema.json", encoding="utf-8") as stream:
        schema = json.load(stream)
    actual = [column for column in test.columns if column != ID_COL]
    bundle_input_columns = bundle.get("input_columns", schema["input_columns"])
    if actual != bundle_input_columns or actual != schema["input_columns"]:
        raise ValueError("CatBoost input schema mismatch")

    predictions = np.zeros(len(test), dtype=np.float64)
    for component in bundle["components"]:
        features = cat.build_features(test, component["feature_state"])
        x = cat.component_frame(features, component, schema)
        if component["name"] != "catboost":
            raise ValueError("CatBoost bundle contains an unsupported component")
        for column in component["categorical_columns"]:
            x[column] = cat.safe_string(x[column])
        part = component["model"].predict_proba(x)[:, 1]
        predictions += float(component["weight"]) * np.asarray(part, dtype=np.float64)
    return predictions


@torch.inference_mode()
def predict_tabm(test: pd.DataFrame, batch_size: int = 16_384) -> np.ndarray:
    torch.set_num_threads(min(6, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        batch_size = min(batch_size, 2048)

    preprocessor = Preprocessor.load("./model/tabm_preprocessor.json")
    x_num, x_cat = preprocessor.transform(test)
    checkpoint = torch.load("./model/tabm.pt", map_location="cpu", weights_only=True)
    model = make_model(
        int(checkpoint["n_num_features"]),
        list(checkpoint["cat_cardinalities"]),
        list(checkpoint["bins"]),
        dict(checkpoint["model_config"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    predictions = []
    for start in range(0, len(test), batch_size):
        xn = torch.from_numpy(x_num[start : start + batch_size]).to(device)
        xc = torch.from_numpy(x_cat[start : start + batch_size]).to(device)
        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            logits = model(xn, xc).squeeze(-1)
        predictions.append(logits.float().sigmoid().mean(dim=1).cpu().numpy())
    return np.concatenate(predictions).astype(np.float64, copy=False)


def apply_platt(calibrator, predictions: np.ndarray) -> np.ndarray:
    probabilities = np.clip(predictions, 1e-6, 1.0 - 1e-6)
    logits = np.log(probabilities / (1.0 - probabilities)).reshape(-1, 1)
    return np.clip(calibrator.predict_proba(logits)[:, 1], 0.0, 1.0)


def main() -> None:
    test_path, sample_path = resolve_paths()
    started = time.perf_counter()
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    sample = pd.read_csv(sample_path, encoding="utf-8-sig")
    if list(sample.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(f"sample columns must be {[ID_COL, TARGET_COL]}")
    if test[ID_COL].duplicated().any() or sample[ID_COL].duplicated().any():
        raise ValueError("duplicate row_id")
    if set(test[ID_COL]) != set(sample[ID_COL]) or len(test) != len(sample):
        raise ValueError("test/sample row_id mismatch")

    catboost_prediction = predict_catboost(test)
    tabm_prediction = predict_tabm(test)
    blended = CATBOOST_WEIGHT * catboost_prediction + TABM_WEIGHT * tabm_prediction
    calibrator = joblib.load("./model/ensemble_calibrator.joblib")
    predictions = apply_platt(calibrator, blended)
    if not np.isfinite(predictions).all() or ((predictions < 0) | (predictions > 1)).any():
        raise ValueError("prediction is not finite or outside [0, 1]")

    prediction_by_id = pd.Series(predictions, index=test[ID_COL])
    result = sample[[ID_COL]].copy()
    result[TARGET_COL] = prediction_by_id.loc[result[ID_COL]].to_numpy()
    os.makedirs("./output", exist_ok=True)
    result.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(
        f"Saved: ./output/submission.csv rows={len(result)} "
        f"catboost_weight={CATBOOST_WEIGHT:.2f} tabm_weight={TABM_WEIGHT:.2f} "
        f"elapsed={time.perf_counter() - started:.2f}s"
    )


if __name__ == "__main__":
    main()
