"""전체 데이터 감사, 시간 rolling 검증, 보정·앙상블 선택 및 최종 재학습."""

from __future__ import annotations

# [추가 구현]
# 목적: Brier/calibration 중심의 누수 없는 모델 선택을 한 명령으로 재현
import argparse
import gc
import json
import platform
import resource
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from calibration import apply_calibrator, compare_calibrators, fit_calibrator
from data_audit import run_audit
from feature_engineering import (
    ID_COL,
    REDUNDANT_COLUMNS,
    TARGET_COL,
    build_features,
    categorical_columns,
    fit_feature_state,
)
from metrics import brier_score, probability_metrics
from validation import (
    fit_final_model,
    fit_official_random_forest,
    fit_predict_model,
    fold_report,
    official_random_forest,
    rolling_splits,
    serialized_size_bytes,
)

MODEL_NAMES = ["random_forest", "catboost", "lightgbm", "xgboost", "histgb"]

BASE_PARAMS: dict[str, dict[str, Any]] = {
    "random_forest": {},
    "catboost": {
        "iterations": 2500,
        "depth": 8,
        "learning_rate": 0.05,
        "l2_leaf_reg": 10.0,
    },
    "lightgbm": {
        "n_estimators": 2000,
        "learning_rate": 0.04,
        "num_leaves": 63,
        "min_child_samples": 200,
        "reg_lambda": 10.0,
    },
    "xgboost": {},
    "histgb": {},
}

CATBOOST_CANDIDATES = [BASE_PARAMS["catboost"]]
LIGHTGBM_CANDIDATES = [BASE_PARAMS["lightgbm"]]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=_json_default)


def sampled_indexes(indexes: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(indexes) <= limit:
        return indexes
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indexes, size=limit, replace=False))


def raw_features(frame: pd.DataFrame, input_columns: list[str], drop_redundant: bool) -> pd.DataFrame:
    cols = [c for c in input_columns if not drop_redundant or c not in REDUNDANT_COLUMNS]
    return frame.loc[:, cols]


def tuning_fit(
    name: str,
    params: dict[str, Any],
    train: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    input_columns: list[str],
    drop_redundant: bool,
    size_dir: Path,
) -> dict[str, Any]:
    raw_train = train.iloc[train_idx]
    raw_val = train.iloc[val_idx]
    y_train = raw_train[TARGET_COL].to_numpy(dtype=int)
    y_val = raw_val[TARGET_COL].to_numpy(dtype=int)
    if name == "random_forest":
        x_train = raw_features(raw_train, input_columns, drop_redundant)
        x_val = raw_features(raw_val, input_columns, drop_redundant)
    else:
        state = fit_feature_state(
            raw_train, y_train, input_columns=input_columns, drop_redundant=drop_redundant
        )
        x_train = build_features(raw_train, state)
        x_val = build_features(raw_val, state)
    model, encoder, preds, fit_s, infer_s, best_iteration = fit_predict_model(
        name, params, x_train, y_train, x_val, y_val
    )
    metrics = probability_metrics(y_val, preds)
    metrics.update(
        {
            "params": params,
            "drop_redundant": drop_redundant,
            "train_rows": len(train_idx),
            "validation_rows": len(val_idx),
            "fit_seconds": fit_s,
            "inference_seconds": infer_s,
            "best_iteration": best_iteration,
            "model_size_bytes": serialized_size_bytes(
                {"model": model, "encoder": encoder}, size_dir
            ),
        }
    )
    del raw_train, raw_val, x_train, x_val, model, encoder, preds
    gc.collect()
    return metrics


def weighted_season_brier(y: np.ndarray, preds: np.ndarray, seasons: np.ndarray) -> float:
    weights = {2022: 0.2, 2023: 0.3, 2024: 0.5}
    return float(
        sum(weights[season] * brier_score(y[seasons == season], preds[seasons == season]) for season in weights)
    )


def search_ensembles(
    y: np.ndarray,
    seasons: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}

    def record(name: str, weights: dict[str, float]) -> None:
        pred = sum(predictions[model] * weight for model, weight in weights.items())
        results[name] = {
            "weights": weights,
            "weighted_season_brier": weighted_season_brier(y, pred, seasons),
            "overall": probability_metrics(y, pred),
            "season_metrics": {
                str(season): probability_metrics(y[seasons == season], pred[seasons == season])
                for season in [2022, 2023, 2024]
            },
        }

    record("catboost", {"catboost": 1.0})
    record("lightgbm", {"lightgbm": 1.0})
    record("xgboost", {"xgboost": 1.0})

    best_pair: tuple[float, dict[str, float]] | None = None
    for step in range(1, 20):
        weights = {"catboost": step * 0.05, "lightgbm": 1.0 - step * 0.05}
        pred = sum(predictions[name] * weight for name, weight in weights.items())
        score = weighted_season_brier(y, pred, seasons)
        if best_pair is None or score < best_pair[0]:
            best_pair = (score, weights)
    assert best_pair is not None
    record("catboost_lightgbm", best_pair[1])

    best_triple: tuple[float, dict[str, float]] | None = None
    for cat_step in range(1, 19):
        for lgb_step in range(1, 20 - cat_step):
            xgb_step = 20 - cat_step - lgb_step
            if xgb_step < 1:
                continue
            weights = {
                "catboost": cat_step * 0.05,
                "lightgbm": lgb_step * 0.05,
                "xgboost": xgb_step * 0.05,
            }
            pred = sum(predictions[name] * weight for name, weight in weights.items())
            score = weighted_season_brier(y, pred, seasons)
            if best_triple is None or score < best_triple[0]:
                best_triple = (score, weights)
    assert best_triple is not None
    record("catboost_lightgbm_xgboost", best_triple[1])

    boosting_candidates = [best_pair[1], best_triple[1]]
    best_rf: tuple[float, dict[str, float]] | None = None
    for boosting in boosting_candidates:
        for rf_weight in [0.05, 0.10, 0.15, 0.20]:
            weights = {name: value * (1.0 - rf_weight) for name, value in boosting.items()}
            weights["random_forest"] = rf_weight
            pred = sum(predictions[name] * weight for name, weight in weights.items())
            score = weighted_season_brier(y, pred, seasons)
            if best_rf is None or score < best_rf[0]:
                best_rf = (score, weights)
    assert best_rf is not None
    record("boosting_plus_random_forest", best_rf[1])
    return results


def complexity_aware_selection(ensemble_results: dict[str, dict[str, Any]]) -> str:
    """추가 라이브러리/모델은 실질적인 Brier 개선이 있을 때만 채택한다."""
    selected = "catboost"
    selected_score = ensemble_results[selected]["weighted_season_brier"]
    thresholds = {
        "catboost_lightgbm": 2e-5,
        "catboost_lightgbm_xgboost": 4e-5,
        "boosting_plus_random_forest": 3e-5,
        "lightgbm": 2e-5,
        "xgboost": 3e-5,
    }
    for name, result in sorted(
        ensemble_results.items(), key=lambda item: item[1]["weighted_season_brier"]
    ):
        improvement = selected_score - result["weighted_season_brier"]
        if improvement > thresholds.get(name, 0.0):
            selected = name
            selected_score = result["weighted_season_brier"]
    return selected


def final_iteration_params(
    name: str,
    params: dict[str, Any],
    fold_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    result = dict(params)
    iterations = [r["best_iteration"] for r in fold_reports if r["best_iteration"] is not None]
    if not iterations:
        return result
    # early stopping의 0-based/1-based 차이를 흡수하고 지나친 최종 학습을 피한다.
    chosen = max(100, int(np.median(iterations)) + 1)
    if name == "catboost":
        result["iterations"] = chosen
        result["verbose"] = 100
    elif name == "lightgbm":
        result["n_estimators"] = chosen
    elif name == "xgboost":
        result["n_estimators"] = chosen
    elif name == "histgb":
        result["max_iter"] = chosen
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--solution-dir", type=Path, default=Path("solution"))
    parser.add_argument("--tuning-train-rows", type=int, default=300_000)
    parser.add_argument("--tuning-val-rows", type=int, default=80_000)
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args()

    started_all = time.perf_counter()
    output_dir = args.solution_dir / "output"
    model_dir = args.solution_dir / "model"
    size_dir = output_dir / ".model_sizes"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_audit:
        run_audit(args.data_dir, output_dir / "data_audit.json")

    test_columns = pd.read_csv(
        args.data_dir / "test.csv", encoding="utf-8-sig", nrows=0
    ).columns.tolist()
    input_columns = [c for c in test_columns if c != ID_COL]
    train = pd.read_csv(args.data_dir / "train.csv", encoding="utf-8-sig")
    if [c for c in train.columns if c != TARGET_COL] != test_columns:
        raise ValueError("train/test 입력 컬럼 또는 순서 불일치")
    if train[ID_COL].duplicated().any():
        raise ValueError("train row_id 중복")

    tune_train_all = np.flatnonzero(train["season"].between(2019, 2023))
    tune_val_all = np.flatnonzero(train["season"] == 2024)
    tune_train_idx = sampled_indexes(tune_train_all, args.tuning_train_rows, 42)
    tune_val_idx = sampled_indexes(tune_val_all, args.tuning_val_rows, 43)

    drop_choice = {name: True for name in MODEL_NAMES}
    drop_choice["random_forest"] = False

    parameter_search: dict[str, list[dict[str, Any]]] = {"catboost": [], "lightgbm": []}
    for params in CATBOOST_CANDIDATES:
        print(f"[tune] catboost {params}", flush=True)
        parameter_search["catboost"].append(
            tuning_fit(
                "catboost",
                params,
                train,
                tune_train_idx,
                tune_val_idx,
                input_columns,
                drop_choice["catboost"],
                size_dir,
            )
        )
    for params in LIGHTGBM_CANDIDATES:
        print(f"[tune] lightgbm {params}", flush=True)
        parameter_search["lightgbm"].append(
            tuning_fit(
                "lightgbm",
                params,
                train,
                tune_train_idx,
                tune_val_idx,
                input_columns,
                drop_choice["lightgbm"],
                size_dir,
            )
        )

    selected_params = {name: dict(BASE_PARAMS[name]) for name in MODEL_NAMES}
    for name in ["catboost", "lightgbm"]:
        best = min(parameter_search[name], key=lambda result: result["brier_score"])
        selected_params[name] = dict(best["params"])

    validation_report: dict[str, Any] = {
        "duplicate_ablation": {},
        "drop_choice": drop_choice,
        "parameter_search": parameter_search,
        "selected_params": selected_params,
        "models": {name: {"folds": []} for name in MODEL_NAMES},
    }
    oof_full = {name: np.full(len(train), np.nan, dtype=np.float64) for name in MODEL_NAMES}

    for train_idx, val_idx, train_period, val_season in rolling_splits(train):
        raw_train = train.iloc[train_idx]
        raw_val = train.iloc[val_idx]
        y_train = raw_train[TARGET_COL].to_numpy(dtype=int)
        y_val = raw_val[TARGET_COL].to_numpy(dtype=int)
        for name in MODEL_NAMES:
            print(f"[rolling] model={name} train={train_period} val={val_season}", flush=True)
            if name == "random_forest":
                x_train = raw_features(raw_train, input_columns, False)
                x_val = raw_features(raw_val, input_columns, False)
            else:
                state = fit_feature_state(
                    raw_train,
                    y_train,
                    input_columns=input_columns,
                    drop_redundant=drop_choice[name],
                )
                x_train = build_features(raw_train, state)
                x_val = build_features(raw_val, state)
            model, encoder, preds, fit_s, infer_s, best_iteration = fit_predict_model(
                name,
                selected_params[name],
                x_train,
                y_train,
                x_val,
                y_val,
            )
            oof_full[name][val_idx] = preds
            size_bytes = serialized_size_bytes({"model": model, "encoder": encoder}, size_dir)
            report = fold_report(
                y_val,
                preds,
                raw_train,
                raw_val,
                train_period=train_period,
                val_season=val_season,
                fit_seconds=fit_s,
                inference_seconds=infer_s,
                model_size_bytes=size_bytes,
                best_iteration=best_iteration,
            )
            validation_report["models"][name]["folds"].append(report)
            del x_train, x_val, model, encoder, preds
            gc.collect()
        del raw_train, raw_val
        gc.collect()

    oof_mask = train["season"].isin([2022, 2023, 2024]).to_numpy()
    oof_y = train.loc[oof_mask, TARGET_COL].to_numpy(dtype=int)
    oof_seasons = train.loc[oof_mask, "season"].to_numpy(dtype=int)
    oof_predictions = {name: values[oof_mask] for name, values in oof_full.items()}
    for name, preds in oof_predictions.items():
        if not np.isfinite(preds).all():
            raise ValueError(f"{name} OOF 누락 또는 비정상 값")
        validation_report["models"][name]["overall_oof"] = probability_metrics(oof_y, preds)
        validation_report["models"][name]["weighted_season_brier"] = weighted_season_brier(
            oof_y, preds, oof_seasons
        )

    ensemble_results = search_ensembles(oof_y, oof_seasons, oof_predictions)
    selected_ensemble = complexity_aware_selection(ensemble_results)
    selected_weights = ensemble_results[selected_ensemble]["weights"]
    raw_ensemble_oof = sum(oof_predictions[name] * weight for name, weight in selected_weights.items())

    calibration_train_mask = oof_seasons < 2024
    calibration_eval_mask = oof_seasons == 2024
    calibration_2024, _ = compare_calibrators(
        raw_ensemble_oof[calibration_train_mask],
        oof_y[calibration_train_mask],
        raw_ensemble_oof[calibration_eval_mask],
        oof_y[calibration_eval_mask],
    )
    # 2022~2023에서 학습해 2024에 일반화되는 Brier를 기준으로 선택한다.
    selected_calibration = min(
        calibration_2024, key=lambda kind: calibration_2024[kind]["brier_score"]
    )
    final_calibrator = fit_calibrator(selected_calibration, raw_ensemble_oof, oof_y)
    calibrated_oof = apply_calibrator(final_calibrator, raw_ensemble_oof)
    calibration_all_oof = {
        kind: probability_metrics(
            oof_y, apply_calibrator(fit_calibrator(kind, raw_ensemble_oof, oof_y), raw_ensemble_oof)
        )
        for kind in ["none", "platt", "temperature", "isotonic"]
    }

    validation_report.update(
        {
            "ensemble_results": ensemble_results,
            "selected_ensemble": selected_ensemble,
            "selected_weights": selected_weights,
            "calibration_2024_temporal_evaluation": calibration_2024,
            "calibration_all_oof_fit_diagnostic": calibration_all_oof,
            "selected_calibration": selected_calibration,
            "selected_raw_oof": probability_metrics(oof_y, raw_ensemble_oof),
            "selected_calibrated_oof": probability_metrics(oof_y, calibrated_oof),
        }
    )
    write_json(output_dir / "validation_report.json", validation_report)
    np.savez_compressed(
        output_dir / "oof_predictions.npz",
        y=oof_y,
        season=oof_seasons,
        **oof_predictions,
        selected_raw=raw_ensemble_oof,
        selected_calibrated=calibrated_oof,
    )

    # 공식 RandomForest 전체 재학습 및 joblib.dump 재현본 저장.
    print("[final] reproduce official RandomForest on all 2019-2024", flush=True)
    rf_full = official_random_forest()
    rf_full = fit_official_random_forest(
        rf_full, raw_features(train, input_columns, False), train[TARGET_COL].to_numpy(dtype=int)
    )
    joblib.dump(rf_full, output_dir / "rf_baseline.pkl", compress=3)
    del rf_full
    gc.collect()

    # 선택한 구성만 2019~2024 전체 데이터로 재학습한다.
    final_components = []
    final_fit_seconds: dict[str, float] = {}
    for name, weight in selected_weights.items():
        if name == "random_forest":
            state_dict = None
            x_full = raw_features(train, input_columns, False)
            feature_cols = list(x_full.columns)
            cat_cols = ["top_bottom", "game_type", "base_state"]
        else:
            state = fit_feature_state(
                train,
                train[TARGET_COL].to_numpy(dtype=int),
                input_columns=input_columns,
                drop_redundant=drop_choice[name],
            )
            state_dict = state.to_dict()
            x_full = build_features(train, state)
            if name in ["lightgbm", "xgboost"]:
                x_full = x_full.drop(columns=["season", "game_type"], errors="ignore")
            feature_cols = list(x_full.columns)
            cat_cols = categorical_columns(x_full)
        params = final_iteration_params(
            name, selected_params[name], validation_report["models"][name]["folds"]
        )
        print(f"[final] model={name} weight={weight} params={params}", flush=True)
        model, encoder, fit_s = fit_final_model(
            name, params, x_full, train[TARGET_COL].to_numpy(dtype=int)
        )
        final_fit_seconds[name] = fit_s
        final_components.append(
            {
                "name": name,
                "weight": float(weight),
                "model": model,
                "encoder": encoder,
                "feature_state": state_dict,
                "feature_columns": feature_cols,
                "categorical_columns": cat_cols,
                "params": params,
            }
        )
        del x_full, model, encoder
        gc.collect()

    final_bundle = {
        "format_version": 1,
        "input_columns": input_columns,
        "components": final_components,
    }
    joblib.dump(final_bundle, model_dir / "final_model.joblib", compress=3)
    joblib.dump(final_calibrator, model_dir / "calibration_model.joblib", compress=3)
    schema = {
        "input_columns": input_columns,
        "input_column_order_required": True,
        "target": TARGET_COL,
        "target_definition": (
            "현재 투구가 가운데 위험 코스, 스트라이크존 대폭 이탈, 포수 요구 반대 방향에 "
            "해당하지 않는 유효한 제구 성공(control_success=1)의 사전 확률"
        ),
        "components": [
            {
                "name": item["name"],
                "feature_state": item["feature_state"],
                "feature_columns": item["feature_columns"],
                "categorical_columns": item["categorical_columns"],
                "encoder": item["encoder"],
            }
            for item in final_components
        ],
    }
    write_json(model_dir / "feature_schema.json", schema)
    write_json(
        model_dir / "ensemble_config.json",
        {
            "selected_ensemble": selected_ensemble,
            "weights": selected_weights,
            "calibration": selected_calibration,
            "weight_search_increment": 0.05,
            "selection_metric": "0.2*Brier(2022)+0.3*Brier(2023)+0.5*Brier(2024)",
        },
    )

    model_bytes = sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file())
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_rss_bytes = int(ru_maxrss if platform.system() == "Darwin" else ru_maxrss * 1024)
    training_report = {
        "total_seconds": time.perf_counter() - started_all,
        "final_fit_seconds": final_fit_seconds,
        "final_model_bytes": model_bytes,
        "max_rss_bytes": max_rss_bytes,
        "selected_ensemble": selected_ensemble,
        "selected_weights": selected_weights,
        "selected_calibration": selected_calibration,
        "trackman_used": False,
        "trackman_reason": "main anonymous player IDs and Trackman player IDs have zero intersection",
        "class_resampling": "none",
        "class_weight": "none",
    }
    write_json(output_dir / "training_report.json", training_report)
    print(json.dumps(training_report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
