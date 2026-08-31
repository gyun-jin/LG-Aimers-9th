from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE_ROOT = ROOT.parent / "5"
TM_ROOT = ROOT.parent / "5-1"
DATA_DIR = ROOT.parent / "data"
HAND_CLEAN_TRAIN_PATH = ROOT.parent / "hand_cleaning_analysis" / "train_hand_trackman_clean.csv"
MODEL_DIR = ROOT / "model"
OUTPUT_DIR = ROOT / "output"

SELECTED_FEATURE_SET = "tm_metadata_physical_pitchmix"
SELECTED_CALIBRATION = "platt"
TRACKMAN_STRATEGY = "mapping_all_shrink"
FOLD_SEASONS = [2022, 2023, 2024]
WEIGHTS = {"catboost": 0.8, "lightgbm": 0.2}
V52_MEAN_BRIER = 0.2469613893682675
V52_SERVER_SCORE = 957.6800554874


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


task5 = load_module("task5_train", BASE_ROOT / "train.py")
task51 = load_module("task51_train", TM_ROOT / "train.py")
task52 = load_module("task52_train", ROOT / "train.py")


def log(message: str) -> None:
    print(f"[task5-2-submit] {time.strftime('%H:%M:%S')} {message}", flush=True)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def zip_submission() -> list[str]:
    path = ROOT / "submit.zip"
    if path.exists():
        path.unlink()
    members = [
        "model/",
        "model/final_model.joblib",
        "model/calibration_model.joblib",
        "model/feature_schema.json",
        "model/ensemble_config.json",
        "script.py",
        "requirements.txt",
    ]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for member in members:
            source = ROOT / member
            if member.endswith("/"):
                zf.writestr(member, "")
            else:
                zf.write(source, member)
    with zipfile.ZipFile(path) as zf:
        return zf.namelist()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def setup_schema() -> dict[str, Any]:
    schema = task5.load_v3_schema()
    for module in (task5, task52):
        module.INPUT_COLUMNS = schema["input_columns"]
        module.V3_FEATURE_COLUMNS = schema["feature_columns"]
        module.V3_CATEGORICAL_COLUMNS = schema["categorical_columns"]
    task52.task5.INPUT_COLUMNS = schema["input_columns"]
    task52.task5.V3_FEATURE_COLUMNS = schema["feature_columns"]
    task52.task5.V3_CATEGORICAL_COLUMNS = schema["categorical_columns"]
    return schema


def build_oof_predictions(train: pd.DataFrame, y: np.ndarray, schema: dict[str, Any], tm_state: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seasons = train["season"].to_numpy()
    tm_features = task51.build_trackman_features(train, tm_state, TRACKMAN_STRATEGY)
    for valid_season in FOLD_SEASONS:
        train_idx = np.flatnonzero(seasons < valid_season)
        valid_idx = np.flatnonzero(seasons == valid_season)
        state = task52.fit_feature_state_53(train.iloc[train_idx], y[train_idx], schema["alpha"])
        base_features = task52.build_v53_features(train, state, tm_features)
        x_all, cats = task52.prepare_features(base_features, tm_features, SELECTED_FEATURE_SET, use_current_season=True)
        x_train = x_all.iloc[train_idx]
        x_valid = x_all.iloc[valid_idx]
        y_valid = y[valid_idx]
        cat_pred = task52.fit_cat(
            x_train,
            y[train_idx],
            x_valid,
            y_valid,
            cats,
            task52.CAT_CONFIGS["cat_current"],
        )
        lgb_pred = task52.fit_lgb(
            x_train,
            y[train_idx],
            x_valid,
            cats,
            task52.LGB_CONFIGS["lgb_current"],
        )
        pred = WEIGHTS["catboost"] * cat_pred + WEIGHTS["lightgbm"] * lgb_pred
        raw_metrics = task5.metrics(y_valid, pred)
        records.append({"season": valid_season, "y": y_valid, "pred": pred, "metrics": raw_metrics})
        log(f"oof fold={valid_season} raw_brier={raw_metrics['brier']:.8f} rows={len(valid_idx):,}")
    return records


def fit_final_models(train: pd.DataFrame, y: np.ndarray, schema: dict[str, Any], tm_state: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    state = task52.fit_feature_state_53(train, y, schema["alpha"])
    tm_features = task51.build_trackman_features(train, tm_state, TRACKMAN_STRATEGY)
    base_features = task52.build_v53_features(train, state, tm_features)
    x_train, cats = task52.prepare_features(base_features, tm_features, SELECTED_FEATURE_SET, use_current_season=True)
    components = []
    for kind, weight in WEIGHTS.items():
        model, _, extra = task5.fit_models(
            x_train,
            y,
            x_train.iloc[:0],
            None,
            cats,
            kind,
            False,
            False,
            42,
        )
        artifact = task5.model_artifact(model, kind, SELECTED_FEATURE_SET, state, x_train, cats, {}, extra.get("encoder"), weight, False)
        artifact["trackman_state"] = tm_state
        components.append(artifact)
        log(f"final fitted {kind} weight={weight:.2f} rows={len(x_train):,} features={len(x_train.columns)}")
    return components, list(x_train.columns), cats


def main() -> None:
    started = time.perf_counter()
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    schema = setup_schema()
    train = pd.read_csv(HAND_CLEAN_TRAIN_PATH)
    y = train[task5.TARGET_COL].to_numpy(dtype=int)
    log(f"loaded train rows={len(train):,}")
    tm_state = task51.build_trackman_state(train)
    log(f"built trackman state rows={tm_state['trackman_rows']:,}")

    oof_records = build_oof_predictions(train, y, schema, tm_state)
    calibrator = task5.fit_calibration(
        SELECTED_CALIBRATION,
        np.concatenate([record["y"] for record in oof_records]),
        np.concatenate([record["pred"] for record in oof_records]),
    )
    calibrated_metrics = []
    for record in oof_records:
        pred = task5.apply_calibration(calibrator, record["pred"])
        metric = task5.metrics(record["y"], pred)
        calibrated_metrics.append(metric)
        log(f"oof fold={record['season']} calibrated_brier={metric['brier']:.8f}")

    components, feature_columns, cat_cols = fit_final_models(train, y, schema, tm_state)
    bundle = {
        "input_columns": task5.INPUT_COLUMNS,
        "components": components,
        "selected_candidate": SELECTED_FEATURE_SET,
        "selected_calibration": SELECTED_CALIBRATION,
    }
    joblib.dump(bundle, MODEL_DIR / "final_model.joblib", compress=3)
    joblib.dump(calibrator, MODEL_DIR / "calibration_model.joblib", compress=3)
    write_json(
        MODEL_DIR / "feature_schema.json",
        {
            "input_columns": task5.INPUT_COLUMNS,
            "target": task5.TARGET_COL,
            "feature_columns": feature_columns,
            "categorical_columns": cat_cols,
            "selected_candidate": SELECTED_FEATURE_SET,
            "selected_variant": SELECTED_FEATURE_SET,
            "trackman_strategy": TRACKMAN_STRATEGY,
            "trackman_feature_set": task52.FEATURE_SETS[SELECTED_FEATURE_SET],
            "current_season_features": task52.CURRENT_SEASON_FEATURES,
            "current_season_categorical_features": task52.CURRENT_SEASON_CATEGORICAL_FEATURES,
            "components": [
                {
                    "name": component["name"],
                    "weight": component["weight"],
                    "feature_columns": component["feature_columns"],
                    "categorical_columns": component["categorical_columns"],
                }
                for component in components
            ],
        },
    )
    write_json(
        MODEL_DIR / "ensemble_config.json",
        {
            "selected_ensemble": SELECTED_FEATURE_SET,
            "weights": WEIGHTS,
            "calibration": SELECTED_CALIBRATION,
            "trackman_strategy": TRACKMAN_STRATEGY,
            "current_season_features": True,
            "training_data": str(HAND_CLEAN_TRAIN_PATH),
            "selection_metric": "full three-fold mean Brier compared with v5-2 reference",
        },
    )
    members = zip_submission()
    submit_path = ROOT / "submit.zip"
    report = {
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "selected_feature_set": SELECTED_FEATURE_SET,
        "selected_calibration": SELECTED_CALIBRATION,
        "weights": WEIGHTS,
        "trackman_strategy": TRACKMAN_STRATEGY,
        "folds": [
            {
                "season": record["season"],
                "raw_metrics": record["metrics"],
                "calibrated_metrics": metric,
            }
            for record, metric in zip(oof_records, calibrated_metrics)
        ],
        "mean_brier": float(np.mean([metric["brier"] for metric in calibrated_metrics])),
        "worst_brier": float(np.max([metric["brier"] for metric in calibrated_metrics])),
        "v5_2_reference_mean_brier": V52_MEAN_BRIER,
        "v5_2_reference_server_score": V52_SERVER_SCORE,
        "submission_recommended": bool(float(np.mean([metric["brier"] for metric in calibrated_metrics])) < V52_MEAN_BRIER),
        "zip_members": members,
        "submit_zip_sha256": file_sha256(submit_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(OUTPUT_DIR / "submit_build_report.json", report)
    log(f"wrote {submit_path} sha256={report['submit_zip_sha256']}")
    log(f"mean_brier={report['mean_brier']:.8f} worst_brier={report['worst_brier']:.8f}")


if __name__ == "__main__":
    main()
