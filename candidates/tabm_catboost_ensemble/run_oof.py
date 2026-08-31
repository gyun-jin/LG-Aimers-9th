from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TABM_DIR = ROOT / "candidates" / "tabm_v1"
V58_DIR = ROOT / "candidates" / "submit_v5_8"
DATA_PATH = ROOT / "data" / "train.csv"
OUTPUT_DIR = HERE / "output"
FOLDS = (2022, 2023, 2024)
SEEDS = (42, 123, 7, 2024, 99)

sys.path.insert(0, str(TABM_DIR))
from common import ID_COL, TARGET_COL, Preprocessor  # noqa: E402
from tabm_model import compute_numerical_bins, make_model  # noqa: E402
from train import predict as predict_tabm  # noqa: E402
from train import set_seed, train_epochs  # noqa: E402


def log(message: str) -> None:
    print(f"[ensemble-oof] {time.strftime('%H:%M:%S')} {message}", flush=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_v58_module():
    spec = importlib.util.spec_from_file_location("v58_inference", V58_DIR / "script.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load submit_v5_8/script.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def brier(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((prediction.astype(np.float64) - y.astype(np.float64)) ** 2))


def load_frame(smoke_rows_per_season: int | None) -> pd.DataFrame:
    log(f"loading {DATA_PATH}")
    frame = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    if smoke_rows_per_season:
        frame = (
            frame.groupby("season", sort=False, group_keys=False)
            .head(smoke_rows_per_season)
            .reset_index(drop=True)
        )
        log(f"smoke sample rows={len(frame):,}")
    return frame


def make_v58_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    v58 = load_v58_module()
    bundle = joblib.load(V58_DIR / "model" / "final_model.joblib")
    schema = json.loads((V58_DIR / "model" / "feature_schema.json").read_text(encoding="utf-8"))
    component = bundle["components"][0]
    log("building v5_8 features from saved full-training feature/TrackMan state")
    started = time.time()
    base = v58.build_features(frame, component["feature_state"])
    x = v58.component_frame(base, component, schema)
    categorical = list(component["categorical_columns"])
    for column in categorical:
        x[column] = v58.safe_string(x[column])
    metadata = {
        "feature_count": int(x.shape[1]),
        "categorical_count": len(categorical),
        "feature_state_source": "submit_v5_8 final_model.joblib (fit on full training data)",
        "clean_train_difference": "OOF uses official train.csv; original v5_8 removed 116 hand-mismatch rows",
        "elapsed_sec": time.time() - started,
    }
    del base, bundle
    gc.collect()
    log(f"v5_8 features ready shape={x.shape} elapsed={metadata['elapsed_sec']:.1f}s")
    return x, categorical, metadata


def run_catboost(frame: pd.DataFrame, args: argparse.Namespace) -> None:
    x, categorical, feature_metadata = make_v58_features(frame)
    seasons = pd.to_numeric(frame["season"], errors="raise").to_numpy(dtype=np.int16)
    y = frame[TARGET_COL].to_numpy(dtype=np.int8)
    row_id = np.asarray(frame[ID_COL].astype(str).tolist(), dtype="U")
    selected_folds = FOLDS if args.fold is None else (args.fold,)
    for valid_season in selected_folds:
        output_path = OUTPUT_DIR / f"catboost_oof_{valid_season}.npz"
        metadata_path = OUTPUT_DIR / f"catboost_oof_{valid_season}.json"
        if output_path.exists() and not args.force:
            log(f"skip cached {output_path.name}")
            continue
        train_idx = np.flatnonzero(seasons < valid_season)
        valid_idx = np.flatnonzero(seasons == valid_season)
        train_x = x.iloc[train_idx]
        valid_x = x.iloc[valid_idx]
        train_pool = Pool(train_x, y[train_idx], cat_features=categorical)
        valid_pool = Pool(valid_x, y[valid_idx], cat_features=categorical)
        seed_predictions: list[np.ndarray] = []
        model_meta: list[dict[str, Any]] = []
        fold_started = time.time()
        log(f"CatBoost fold={valid_season} train={len(train_idx):,} valid={len(valid_idx):,}")
        for seed in SEEDS[: args.cat_seeds]:
            model = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="BrierScore",
                iterations=args.cat_iterations,
                learning_rate=0.03,
                depth=8,
                l2_leaf_reg=20.0,
                random_seed=seed,
                allow_writing_files=False,
                verbose=50,
                thread_count=args.threads,
                od_type="Iter",
                od_wait=60,
            )
            seed_started = time.time()
            model.fit(train_pool, eval_set=valid_pool)
            prediction = model.predict_proba(valid_pool, thread_count=args.threads)[:, 1]
            seed_predictions.append(prediction)
            meta = {
                "seed": seed,
                "tree_count": int(model.tree_count_),
                "brier": brier(y[valid_idx], prediction),
                "elapsed_sec": time.time() - seed_started,
            }
            model_meta.append(meta)
            log(
                f"CatBoost fold={valid_season} seed={seed} trees={meta['tree_count']} "
                f"brier={meta['brier']:.8f} elapsed={meta['elapsed_sec']:.1f}s"
            )
            del model, prediction
            gc.collect()
        averaged = np.mean(seed_predictions, axis=0).astype(np.float32)
        np.savez_compressed(
            output_path,
            row_id=row_id[valid_idx],
            season=np.full(len(valid_idx), valid_season, dtype=np.int16),
            y_true=y[valid_idx],
            raw_pred=averaged,
        )
        write_json(
            metadata_path,
            {
                "model": "catboost_v5_8_retrained_oof",
                "season": valid_season,
                "train_rows": len(train_idx),
                "valid_rows": len(valid_idx),
                "seeds": list(SEEDS[: args.cat_seeds]),
                "iterations": args.cat_iterations,
                "brier": brier(y[valid_idx], averaged),
                "elapsed_sec": time.time() - fold_started,
                "models": model_meta,
                **feature_metadata,
            },
        )
        log(f"saved {output_path.name} brier={brier(y[valid_idx], averaged):.8f}")
        del train_x, valid_x, train_pool, valid_pool, seed_predictions, averaged
        gc.collect()


def run_tabm(frame: pd.DataFrame, args: argparse.Namespace) -> None:
    set_seed(42)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.tabm_batch_size
    input_columns = [column for column in frame.columns if column not in (ID_COL, TARGET_COL)]
    seasons = pd.to_numeric(frame["season"], errors="raise").to_numpy(dtype=np.int16)
    y = frame[TARGET_COL].to_numpy(dtype=np.float32)
    row_id = np.asarray(frame[ID_COL].astype(str).tolist(), dtype="U")
    selected_folds = FOLDS if args.fold is None else (args.fold,)
    for valid_season in selected_folds:
        output_path = OUTPUT_DIR / f"tabm_oof_{valid_season}.npz"
        metadata_path = OUTPUT_DIR / f"tabm_oof_{valid_season}.json"
        if output_path.exists() and not args.force:
            log(f"skip cached {output_path.name}")
            continue
        train_mask = seasons < valid_season
        valid_mask = seasons == valid_season
        fold_started = time.time()
        log(
            f"TabM fold={valid_season} train={int(train_mask.sum()):,} "
            f"valid={int(valid_mask.sum()):,} device={device}"
        )
        set_seed(42)
        prep = Preprocessor.fit(frame.loc[train_mask], input_columns)
        x_num_train, x_cat_train = prep.transform(frame.loc[train_mask])
        x_num_valid, x_cat_valid = prep.transform(frame.loc[valid_mask])
        y_train = y[train_mask]
        y_valid = y[valid_mask]
        bins = compute_numerical_bins(x_num_train, seed=42)
        model = make_model(x_num_train.shape[1], prep.cat_cardinalities, bins).to(device)
        best_state, best_epoch, best_brier = train_epochs(
            model,
            (x_num_train, x_cat_train, y_train),
            device,
            batch_size=batch_size,
            max_epochs=args.tabm_max_epochs,
            patience=args.tabm_patience,
            validation_data=(x_num_valid, x_cat_valid, y_valid),
        )
        model.load_state_dict(best_state)
        prediction = predict_tabm(
            model,
            x_num_valid,
            x_cat_valid,
            device,
            batch_size=args.tabm_predict_batch_size,
        ).astype(np.float32)
        np.savez_compressed(
            output_path,
            row_id=row_id[valid_mask],
            season=np.full(int(valid_mask.sum()), valid_season, dtype=np.int16),
            y_true=y_valid.astype(np.int8),
            raw_pred=prediction,
        )
        write_json(
            metadata_path,
            {
                "model": "tabm_v1_oof",
                "season": valid_season,
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
                "device": str(device),
                "batch_size": batch_size,
                "max_epochs": args.tabm_max_epochs,
                "patience": args.tabm_patience,
                "best_epoch": best_epoch + 1,
                "reported_best_brier": best_brier,
                "saved_prediction_brier": brier(y_valid, prediction),
                "elapsed_sec": time.time() - fold_started,
                "seed": 42,
            },
        )
        log(
            f"saved {output_path.name} epoch={best_epoch + 1} "
            f"brier={brier(y_valid, prediction):.8f} elapsed={time.time()-fold_started:.1f}s"
        )
        del prep, x_num_train, x_cat_train, x_num_valid, x_cat_valid
        del y_train, y_valid, bins, model, best_state, prediction
        gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate aligned CatBoost and TabM temporal OOF predictions")
    parser.add_argument("--model", choices=("catboost", "tabm", "both"), default="both")
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--threads", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--cat-seeds", type=int, default=5, choices=range(1, 6))
    parser.add_argument("--cat-iterations", type=int, default=600)
    parser.add_argument("--tabm-max-epochs", type=int, default=30)
    parser.add_argument("--tabm-patience", type=int, default=5)
    parser.add_argument("--tabm-batch-size", type=int, default=4096)
    parser.add_argument("--tabm-predict-batch-size", type=int, default=16_384)
    parser.add_argument("--smoke-rows-per-season", type=int)
    parser.add_argument("--output-tag")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    global OUTPUT_DIR
    args = parse_args()
    if not 1 <= args.threads <= 6:
        raise ValueError("--threads must be between 1 and 6")
    random.seed(42)
    np.random.seed(42)
    warnings.filterwarnings(
        "ignore",
        message=r"The \d+-th feature has just two bin edges.*",
    )
    if args.output_tag:
        OUTPUT_DIR = HERE / f"output_{args.output_tag}"
    elif args.smoke_rows_per_season:
        OUTPUT_DIR = HERE / "output_smoke"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_frame(args.smoke_rows_per_season)
    if args.model in ("catboost", "both"):
        run_catboost(frame, args)
    if args.model in ("tabm", "both"):
        run_tabm(frame, args)


if __name__ == "__main__":
    main()
