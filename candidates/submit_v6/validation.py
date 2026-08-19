"""시간 rolling 검증용 모델·인코더·측정 유틸."""

# [추가 구현]
# 목적: 각 fold에서 전처리, 범주 mapping, prior를 독립적으로 fit
from __future__ import annotations

import os
try:
    import resource
except ImportError:  # Windows에는 resource 모듈이 없음
    resource = None
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping as lgb_early_stopping
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

from feature_engineering import categorical_columns, prepare_catboost
from metrics import brier_score, probability_metrics

THREADS = 6


def rolling_splits(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, str, int]]:
    folds = []
    for train_end, val_season in [(2021, 2022), (2022, 2023), (2023, 2024)]:
        train_idx = np.flatnonzero((frame["season"] >= 2019) & (frame["season"] <= train_end))
        val_idx = np.flatnonzero(frame["season"] == val_season)
        folds.append((train_idx, val_idx, f"2019-{train_end}", val_season))
    return folds


def fit_category_encoder(frame: pd.DataFrame, cat_columns: list[str], kind: str) -> dict[str, Any]:
    mappings: dict[str, dict[str, float | int]] = {}
    for col in cat_columns:
        values = frame[col].astype("string").fillna("__MISSING__").astype(str)
        if kind == "category_code":
            categories = pd.Index(values.unique()).tolist()
            mappings[col] = {value: int(index) for index, value in enumerate(categories)}
        elif kind == "frequency":
            frequencies = values.value_counts(dropna=False, normalize=True)
            mappings[col] = {str(value): float(freq) for value, freq in frequencies.items()}
        else:
            raise ValueError(kind)
    return {"kind": kind, "mappings": mappings, "categorical_columns": list(cat_columns)}


def apply_category_encoder(frame: pd.DataFrame, encoder: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    kind = encoder["kind"]
    for col in encoder["categorical_columns"]:
        values = result[col].astype("string").fillna("__MISSING__").astype(str)
        default = -1 if kind == "category_code" else 0.0
        result[col] = values.map(encoder["mappings"][col]).fillna(default)
        result[col] = result[col].astype("int32" if kind == "category_code" else "float32")
    return result


def official_random_forest() -> Pipeline:
    # =============================================================================
    # [자료 제공 코드 시작]
    # 출처: Baseline Train PDF
    # =============================================================================
    cat_cols = ["top_bottom", "game_type", "base_state"]
    features = None  # 실제 컬럼은 fit 직전에 주입
    # =============================================================================
    # [자료 제공 코드 끝]
    # =============================================================================
    # [자료 제공 코드 기반 수정]
    # 수정 이유: 함수 안에서 실제 입력 컬럼을 받아 공식 ColumnTransformer 구조를 유지
    model = Pipeline(
        [
            ("pre", "passthrough"),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_leaf=200,
                    n_jobs=THREADS,
                    random_state=42,
                ),
            ),
        ]
    )
    model._official_cat_cols = cat_cols  # type: ignore[attr-defined]
    model._official_features = features  # type: ignore[attr-defined]
    return model


def fit_official_random_forest(model: Pipeline, x_train: pd.DataFrame, y_train: np.ndarray) -> Pipeline:
    cat_cols = model._official_cat_cols  # type: ignore[attr-defined]
    num_cols = [c for c in x_train.columns if c not in cat_cols]
    # =============================================================================
    # [자료 제공 코드 시작]
    # 출처: Baseline Train PDF
    # =============================================================================
    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                cat_cols,
            ),
            ("num", SimpleImputer(strategy="median"), num_cols),
        ]
    )
    model.set_params(pre=preprocessor)
    model.fit(x_train, y_train)
    # =============================================================================
    # [자료 제공 코드 끝]
    # =============================================================================
    return model


def create_model(name: str, params: dict[str, Any]) -> Any:
    if name == "catboost":
        config = {
            "iterations": 2500,
            "depth": 8,
            "learning_rate": 0.05,
            "l2_leaf_reg": 10.0,
            "loss_function": "Logloss",
            "eval_metric": "BrierScore",
            "random_seed": 42,
            "thread_count": THREADS,
            "allow_writing_files": False,
            "verbose": 100,
        }
        config.update(params)
        return CatBoostClassifier(**config)
    if name == "lightgbm":
        config = {
            "objective": "binary",
            "n_estimators": 2000,
            "learning_rate": 0.04,
            "num_leaves": 63,
            "min_child_samples": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 10.0,
            "random_state": 42,
            "n_jobs": THREADS,
            "verbosity": -1,
        }
        config.update(params)
        return LGBMClassifier(**config)
    if name == "xgboost":
        config = {
            "n_estimators": 1600,
            "max_depth": 7,
            "learning_rate": 0.04,
            "min_child_weight": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 10.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": THREADS,
            "random_state": 42,
            "early_stopping_rounds": 100,
        }
        config.update(params)
        return XGBClassifier(**config)
    if name == "histgb":
        config = {
            "learning_rate": 0.06,
            "max_iter": 500,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 200,
            "l2_regularization": 10.0,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 40,
            "random_state": 42,
        }
        config.update(params)
        return HistGradientBoostingClassifier(**config)
    if name == "random_forest":
        return official_random_forest()
    raise ValueError(name)


def fit_predict_model(
    name: str,
    params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
) -> tuple[Any, dict[str, Any] | None, np.ndarray, float, float, int | None]:
    model = create_model(name, params)
    cat_cols = categorical_columns(x_train)
    encoder: dict[str, Any] | None = None
    if name == "random_forest":
        train_ready, val_ready = x_train, x_val
    elif name == "catboost":
        train_ready = prepare_catboost(x_train, cat_cols)
        val_ready = prepare_catboost(x_val, cat_cols)
    elif name == "lightgbm":
        encoder = fit_category_encoder(x_train, cat_cols, "category_code")
        train_ready = apply_category_encoder(x_train, encoder)
        val_ready = apply_category_encoder(x_val, encoder)
        train_ready = train_ready.drop(columns=["season", "game_type"], errors="ignore")
        val_ready = val_ready.drop(columns=["season", "game_type"], errors="ignore")
    else:
        # 익명 ID를 포함한 범주형 값은 학습 fold 빈도로만 인코딩한다.
        encoder = fit_category_encoder(x_train, cat_cols, "frequency")
        train_ready = apply_category_encoder(x_train, encoder)
        val_ready = apply_category_encoder(x_val, encoder)
        if name == "xgboost":
            train_ready = train_ready.drop(columns=["season", "game_type"], errors="ignore")
            val_ready = val_ready.drop(columns=["season", "game_type"], errors="ignore")

    started = time.perf_counter()
    if name == "random_forest":
        model = fit_official_random_forest(model, train_ready, y_train)
    elif name == "catboost":
        valid_cats = [c for c in cat_cols if c in train_ready.columns]
        model.fit(
            train_ready,
            y_train,
            cat_features=valid_cats,
            eval_set=(val_ready, y_val),
            early_stopping_rounds=100,
            use_best_model=True,
        )
    elif name == "lightgbm":
        valid_cats = [c for c in cat_cols if c in train_ready.columns]
        model.fit(
            train_ready,
            y_train,
            eval_set=[(val_ready, y_val)],
            categorical_feature=valid_cats,
            callbacks=[lgb_early_stopping(100, verbose=False)],
        )
    elif name == "xgboost":
        model.fit(train_ready, y_train, eval_set=[(val_ready, y_val)], verbose=False)
    else:
        model.fit(train_ready, y_train)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    preds = model.predict_proba(val_ready)[:, 1]
    inference_seconds = time.perf_counter() - started
    best_iteration = getattr(model, "best_iteration_", None)
    if best_iteration is None:
        best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        best_iteration = getattr(model, "n_iter_", None)
    if isinstance(best_iteration, np.ndarray):
        best_iteration = int(best_iteration[0])
    return model, encoder, np.asarray(preds), fit_seconds, inference_seconds, best_iteration


def fit_final_model(
    name: str,
    params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple[Any, dict[str, Any] | None, float]:
    model = create_model(name, params)
    cat_cols = categorical_columns(x_train)
    encoder: dict[str, Any] | None = None
    if name == "random_forest":
        ready = x_train
    elif name == "catboost":
        ready = prepare_catboost(x_train, cat_cols)
    elif name == "lightgbm":
        encoder = fit_category_encoder(x_train, cat_cols, "category_code")
        ready = apply_category_encoder(x_train, encoder)
        ready = ready.drop(columns=["season", "game_type"], errors="ignore")
    else:
        encoder = fit_category_encoder(x_train, cat_cols, "frequency")
        ready = apply_category_encoder(x_train, encoder)
        if name == "xgboost":
            ready = ready.drop(columns=["season", "game_type"], errors="ignore")
    started = time.perf_counter()
    if name == "random_forest":
        model = fit_official_random_forest(model, ready, y_train)
    elif name == "catboost":
        valid_cats = [c for c in cat_cols if c in ready.columns]
        model.fit(ready, y_train, cat_features=valid_cats)
    elif name == "lightgbm":
        valid_cats = [c for c in cat_cols if c in ready.columns]
        model.fit(ready, y_train, categorical_feature=valid_cats)
    else:
        model.set_params(early_stopping_rounds=None)
        model.fit(ready, y_train)
    return model, encoder, time.perf_counter() - started


def serialized_size_bytes(value: Any, directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="model-size-", suffix=".joblib", dir=directory)
    os.close(fd)
    try:
        joblib.dump(value, path, compress=3)
        return int(os.path.getsize(path))
    finally:
        os.unlink(path)


def fold_report(
    y_val: np.ndarray,
    preds: np.ndarray,
    raw_train: pd.DataFrame,
    raw_val: pd.DataFrame,
    *,
    train_period: str,
    val_season: int,
    fit_seconds: float,
    inference_seconds: float,
    model_size_bytes: int,
    best_iteration: int | None,
) -> dict[str, Any]:
    result = probability_metrics(y_val, preds)
    old_pitchers = set(raw_train["pitcher_id"].unique())
    old_batters = set(raw_train["batter_id"].unique())
    new_pitcher = ~raw_val["pitcher_id"].isin(old_pitchers).to_numpy()
    new_batter = ~raw_val["batter_id"].isin(old_batters).to_numpy()

    def subgroup(mask: np.ndarray) -> dict[str, Any]:
        return {
            "rows": int(mask.sum()),
            "brier_score": brier_score(y_val[mask], preds[mask]) if mask.any() else None,
        }

    result.update(
        {
            "train_period": train_period,
            "validation_season": int(val_season),
            "train_rows": int(len(raw_train)),
            "validation_rows": int(len(raw_val)),
            "fit_seconds": float(fit_seconds),
            "inference_seconds": float(inference_seconds),
            "model_size_bytes": int(model_size_bytes),
            "best_iteration": int(best_iteration) if best_iteration is not None else None,
            "new_pitcher": subgroup(new_pitcher),
            "existing_pitcher": subgroup(~new_pitcher),
            "new_batter": subgroup(new_batter),
            "existing_batter": subgroup(~new_batter),
            "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) if resource is not None else 0,
        }
    )
    return result
