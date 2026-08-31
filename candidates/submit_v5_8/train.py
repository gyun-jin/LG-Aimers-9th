from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


ROOT = Path(__file__).resolve().parent
BASE_ROOT = ROOT.parent / "5"
TM_ROOT = ROOT.parent / "5-1"
DATA_DIR = ROOT.parent / "data"
HAND_CLEAN_TRAIN_PATH = ROOT.parent / "hand_cleaning_analysis" / "train_hand_trackman_clean.csv"
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = ROOT / "model"

base_spec = importlib.util.spec_from_file_location("task5_train", BASE_ROOT / "train.py")
task5 = importlib.util.module_from_spec(base_spec)
assert base_spec and base_spec.loader
base_spec.loader.exec_module(task5)

tm_spec = importlib.util.spec_from_file_location("task51_train", TM_ROOT / "train.py")
task51 = importlib.util.module_from_spec(tm_spec)
assert tm_spec and tm_spec.loader
tm_spec.loader.exec_module(task51)


FOLD_SEASONS = [2022, 2023, 2024]
CAT_ONLY_REFERENCE_MEAN = 0.2468821392266395
CAT_ONLY_REFERENCE_FOLDS = {
    2022: 0.24310033697607944,
    2023: 0.24991831805583606,
    2024: 0.24762776264800307,
}
CAT_CONFIG = {
    "iterations": 520,
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 20.0,
    "od_wait": 60,
}
BASE_TRACKMAN_SET = "tm_metadata_physical_pitchmix"
TM_METADATA_STABLE = [
    "tm_has_mapping",
    "tm_has_pitcher_history",
    "tm_is_low_history",
    "tm_is_rookie_or_no_history",
    "tm_hist_pitch_count",
    "tm_hist_game_count",
    "tm_hist_season_count",
    "tm_mapping_exact_game_count",
    "tm_mapping_margin",
    "tm_mapping_ratio",
    "tm_mapping_n_candidates",
    "tm_mapping_confidence",
    "tm_mapping_confidence_bucket",
    "tm_mapping_is_high_confidence",
    "tm_mapping_selected",
]
TM_PHYSICAL_STABLE = [
    "tm_hist_release_speed_mean_shrunk",
    "tm_hist_release_speed_std_shrunk",
    "tm_hist_spin_rate_mean_shrunk",
    "tm_hist_spin_rate_std_shrunk",
    "tm_hist_release_pos_x_mean_shrunk",
    "tm_hist_release_pos_x_std_shrunk",
    "tm_hist_release_pos_z_mean_shrunk",
    "tm_hist_release_pos_z_std_shrunk",
    "tm_hist_extension_mean_shrunk",
    "tm_hist_extension_std_shrunk",
]
TM_PITCH_STABLE = [
    "tm_hist_pitch_group_fastball_shrunk",
    "tm_hist_pitch_group_breaking_shrunk",
    "tm_hist_pitch_group_offspeed_shrunk",
    "tm_hist_pitch_type_entropy",
]
FEATURE_SETS = {
    BASE_TRACKMAN_SET: TM_METADATA_STABLE + TM_PHYSICAL_STABLE + TM_PITCH_STABLE,
}
CURRENT_SEASON_FEATURES = [
    "pitcher_current_season_n",
    "log1p_pitcher_current_season_n",
    "pitcher_current_season_success_count",
    "pitcher_current_season_success_rate",
    "pitcher_current_season_success_rate_smoothed",
    "pitcher_current_season_success_minus_career",
    "pitcher_current_season_success_minus_prev1",
    "pitcher_current_season_success_minus_prev3",
    "pitcher_current_season_success_minus_prev5",
    "pitcher_current_season_n_ratio_to_career",
    "pitcher_current_season_available_flag",
    "pitcher_current_season_small_sample_flag",
    "pitcher_current_season_success_x_tm_available",
]
CURRENT_SEASON_CATEGORICAL_FEATURES = [
    "pitcher_current_season_success_rate_bin",
    "pitcher_current_season_success_x_count_state",
    "pitcher_current_season_success_x_game_type",
    "pitcher_current_season_success_x_li_bin",
    "pitcher_current_season_success_x_base_state",
    "pitcher_current_season_success_x_hand_matchup",
    "pitcher_current_season_success_x_tm_mapping_confidence_bucket",
]
CANDIDATES = {
    "cat53_repro_seed42": {
        "description": "5-3 CatBoost-only 재현 후보",
        "seeds": [42],
        "alpha": None,
        "drop_features": [],
    },
    "cat_seed_ensemble_3": {
        "description": "동일 피처 3-seed CatBoost 평균 ensemble",
        "seeds": [42, 123, 2024],
        "alpha": None,
        "drop_features": [],
    },
    "cat_seed_ensemble_5": {
        "description": "동일 피처 5-seed CatBoost 평균 ensemble",
        "seeds": [42, 123, 7, 2024, 99],
        "alpha": None,
        "drop_features": [],
    },
    "cat_current_alpha_100_seed3": {
        "description": "현재시즌 성공률 smoothing alpha=100 + 3-seed ensemble",
        "seeds": [42, 123, 2024],
        "alpha": 100.0,
        "drop_features": [],
    },
    "cat_season_stable_seed3": {
        "description": "숫자 season 피처 제거 + 3-seed ensemble",
        "seeds": [42, 123, 2024],
        "alpha": None,
        "drop_features": ["season"],
    },
}


def log(message: str) -> None:
    print(f"[5-8] {time.strftime('%H:%M:%S')} {message}", flush=True)


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


def load_training_frame() -> pd.DataFrame:
    return pd.read_csv(HAND_CLEAN_TRAIN_PATH)


def build_current_season_state(frame: pd.DataFrame, y: np.ndarray, alpha: float) -> dict[str, Any]:
    work = frame[["pitcher_id", "season", "asof_pitcher_n", "asof_pitcher_success_rate"]].copy()
    work["asof_pitcher_n"] = pd.to_numeric(work["asof_pitcher_n"], errors="coerce").fillna(0.0).clip(lower=0.0)
    work["asof_pitcher_success_rate"] = pd.to_numeric(work["asof_pitcher_success_rate"], errors="coerce")
    prior = float(np.mean(y)) if len(y) else 0.5
    work["asof_pitcher_success_count"] = work["asof_pitcher_n"] * work["asof_pitcher_success_rate"].fillna(prior)
    season_end = (
        work.groupby(["pitcher_id", "season"], observed=True, sort=True)
        .agg(
            season_end_pitcher_n=("asof_pitcher_n", "max"),
            season_end_pitcher_success_count=("asof_pitcher_success_count", "max"),
        )
        .reset_index()
    )
    seasons = sorted(pd.to_numeric(frame["season"], errors="coerce").dropna().astype(int).unique().tolist())
    if seasons:
        seasons = list(range(min(seasons), max(seasons) + 2))
    rows: list[dict[str, Any]] = []
    for pitcher_id, group in season_end.groupby("pitcher_id", sort=False):
        group = group.sort_values("season")
        season_to_n = dict(zip(group["season"].astype(int), group["season_end_pitcher_n"].astype(float)))
        season_to_s = dict(zip(group["season"].astype(int), group["season_end_pitcher_success_count"].astype(float)))
        last_n = 0.0
        last_s = 0.0
        for season in seasons:
            rows.append(
                {
                    "pitcher_id": pitcher_id,
                    "season": int(season),
                    "prior_season_end_pitcher_n": float(last_n),
                    "prior_season_end_pitcher_success_count": float(last_s),
                }
            )
            if season in season_to_n:
                last_n = max(last_n, float(season_to_n[season]))
                last_s = max(last_s, float(season_to_s[season]))
    return {"alpha": float(alpha), "target_prior": prior, "prior_table": pd.DataFrame(rows)}


def fit_feature_state_58(frame: pd.DataFrame, y: np.ndarray, alpha: float) -> dict[str, Any]:
    state = task5.fit_feature_state(frame, y, alpha)
    state["current_season_state"] = build_current_season_state(frame, y, alpha)
    return state


def rate_bin(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return pd.cut(
        values,
        bins=[-np.inf, 0.35, 0.45, 0.55, 0.65, np.inf],
        labels=["very_low", "low", "mid", "high", "very_high"],
    ).astype("string").fillna("missing").astype(str)


def add_current_season_features(features: pd.DataFrame, state: dict[str, Any], tm_features: pd.DataFrame | None = None) -> pd.DataFrame:
    out = features.copy()
    cs_state = state["current_season_state"]
    rows = out[["pitcher_id", "season", "asof_pitcher_n", "asof_pitcher_success_rate"]].merge(
        cs_state["prior_table"], on=["pitcher_id", "season"], how="left"
    )
    prior_n = pd.to_numeric(rows["prior_season_end_pitcher_n"], errors="coerce").fillna(0.0).clip(lower=0.0)
    prior_success = pd.to_numeric(rows["prior_season_end_pitcher_success_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    career_n = pd.to_numeric(rows["asof_pitcher_n"], errors="coerce").fillna(0.0).clip(lower=0.0)
    career_rate = pd.to_numeric(rows["asof_pitcher_success_rate"], errors="coerce").fillna(float(cs_state["target_prior"]))
    career_success = career_n * career_rate
    current_n = (career_n - prior_n).clip(lower=0.0)
    current_success = (career_success - prior_success).clip(lower=0.0)
    current_success = np.minimum(current_success, current_n)
    current_rate = current_success / current_n.replace(0.0, np.nan)
    current_rate = current_rate.fillna(float(cs_state["target_prior"])).clip(0.0, 1.0)
    alpha = float(cs_state["alpha"])
    smoothed = ((current_success + alpha * float(cs_state["target_prior"])) / (current_n + alpha)).clip(0.0, 1.0)

    out["pitcher_current_season_n"] = current_n
    out["log1p_pitcher_current_season_n"] = np.log1p(current_n)
    out["pitcher_current_season_success_count"] = current_success
    out["pitcher_current_season_success_rate"] = current_rate
    out["pitcher_current_season_success_rate_smoothed"] = smoothed
    out["pitcher_current_season_success_minus_career"] = smoothed - career_rate
    out["pitcher_current_season_success_minus_prev1"] = smoothed - pd.to_numeric(out["asof_pitcher_prev1_game_success_rate"], errors="coerce").fillna(float(cs_state["target_prior"]))
    out["pitcher_current_season_success_minus_prev3"] = smoothed - pd.to_numeric(out["asof_pitcher_prev3_game_success_rate"], errors="coerce").fillna(float(cs_state["target_prior"]))
    out["pitcher_current_season_success_minus_prev5"] = smoothed - pd.to_numeric(out["asof_pitcher_prev5_game_success_rate"], errors="coerce").fillna(float(cs_state["target_prior"]))
    out["pitcher_current_season_n_ratio_to_career"] = (current_n / career_n.replace(0.0, np.nan)).fillna(0.0).clip(0.0, 1.0)
    out["pitcher_current_season_available_flag"] = (current_n > 0).astype("int8")
    out["pitcher_current_season_small_sample_flag"] = ((current_n > 0) & (current_n < 30)).astype("int8")
    out["pitcher_current_season_success_rate_bin"] = rate_bin(smoothed)
    li = pd.to_numeric(out["li"], errors="coerce").fillna(0.0)
    out["li_bin"] = pd.cut(li, [-np.inf, 0.7, 1.5, np.inf], labels=["low", "mid", "high"]).astype("string").fillna("mid").astype(str)
    rb = out["pitcher_current_season_success_rate_bin"]
    out["pitcher_current_season_success_x_count_state"] = rb + "_count_" + task5.safe_string(out["count_state"])
    out["pitcher_current_season_success_x_game_type"] = rb + "_game_" + task5.safe_string(out["game_type"])
    out["pitcher_current_season_success_x_li_bin"] = rb + "_li_" + task5.safe_string(out["li_bin"])
    out["pitcher_current_season_success_x_base_state"] = rb + "_base_" + task5.safe_string(out["base_state"])
    out["pitcher_current_season_success_x_hand_matchup"] = rb + "_hand_" + task5.safe_string(out["hand_matchup"])
    out["pitcher_current_season_success_x_tm_available"] = smoothed * pd.to_numeric(tm_features["tm_has_mapping"], errors="coerce").fillna(0.0) if tm_features is not None and "tm_has_mapping" in tm_features.columns else 0.0
    bucket = task5.safe_string(tm_features["tm_mapping_confidence_bucket"]) if tm_features is not None and "tm_mapping_confidence_bucket" in tm_features.columns else pd.Series(["none_low"] * len(out), index=out.index)
    out["pitcher_current_season_success_x_tm_mapping_confidence_bucket"] = rb + "_tm_" + bucket
    return out


def build_v58_features(frame: pd.DataFrame, state: dict[str, Any], tm_features: pd.DataFrame | None) -> pd.DataFrame:
    return add_current_season_features(task5.build_v3_features(frame, state), state, tm_features)


def prepare_features(base_features: pd.DataFrame, tm_features: pd.DataFrame, candidate: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    frame = base_features.loc[:, list(task5.V3_FEATURE_COLUMNS or [])].copy()
    frame = pd.concat([frame, base_features.loc[:, CURRENT_SEASON_FEATURES + CURRENT_SEASON_CATEGORICAL_FEATURES]], axis=1)
    frame = pd.concat([frame, tm_features.loc[:, FEATURE_SETS[BASE_TRACKMAN_SET]]], axis=1)
    x = frame.drop(columns=[c for c in ["pitcher_id", "batter_id"] if c in frame.columns])
    x = x.drop(columns=[c for c in candidate.get("drop_features", []) if c in x.columns])
    cats = [c for c in task5.V3_CATEGORICAL_COLUMNS or [] if c in x.columns]
    for col in CURRENT_SEASON_CATEGORICAL_FEATURES + ["tm_mapping_confidence_bucket"]:
        if col in x.columns and col not in cats:
            cats.append(col)
    return x, cats


def fit_cat_model(x_train: pd.DataFrame, y_train: np.ndarray, x_valid: pd.DataFrame | None, y_valid: np.ndarray | None, cats: list[str], seed: int) -> tuple[CatBoostClassifier, np.ndarray | None, dict[str, Any]]:
    train_x = x_train.copy()
    valid_x = x_valid.copy() if x_valid is not None else None
    for col in cats:
        train_x[col] = task5.safe_string(train_x[col])
        if valid_x is not None:
            valid_x[col] = task5.safe_string(valid_x[col])
    params = dict(
        loss_function="Logloss",
        eval_metric="BrierScore",
        iterations=CAT_CONFIG["iterations"],
        learning_rate=CAT_CONFIG["learning_rate"],
        depth=CAT_CONFIG["depth"],
        l2_leaf_reg=CAT_CONFIG["l2_leaf_reg"],
        random_seed=seed,
        allow_writing_files=False,
        verbose=50,
        thread_count=min(12, os.cpu_count() or 1),
    )
    if y_valid is not None and not getattr(fit_cat_model, "disable_early_stopping", False):
        params["od_type"] = "Iter"
        params["od_wait"] = CAT_CONFIG["od_wait"]
    elif y_valid is not None:
        params["use_best_model"] = False
    model = CatBoostClassifier(**params)
    eval_set = Pool(valid_x, y_valid, cat_features=cats) if valid_x is not None and y_valid is not None else None
    model.fit(Pool(train_x, y_train, cat_features=cats), eval_set=eval_set)
    pred = model.predict_proba(valid_x)[:, 1] if valid_x is not None else None
    meta = {
        "seed": seed,
        "requested_iterations": CAT_CONFIG["iterations"],
        "actual_tree_count": int(model.tree_count_),
        "best_iteration": int(model.get_best_iteration() or model.tree_count_ - 1),
        "early_stopped": bool(model.tree_count_ < CAT_CONFIG["iterations"]),
        "learning_rate": CAT_CONFIG["learning_rate"],
        "depth": CAT_CONFIG["depth"],
        "l2_leaf_reg": CAT_CONFIG["l2_leaf_reg"],
        "od_wait": CAT_CONFIG["od_wait"],
    }
    return model, pred, meta


def bss_score(y: np.ndarray, p: np.ndarray) -> float:
    r = float(np.mean(y))
    baseline_brier = r * (1.0 - r)
    if baseline_brier <= 0:
        return 0.0
    return float(max(0.0, 100000.0 * (1.0 - task5.metrics(y, p)["brier"] / baseline_brier)))


def fold_metric(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    metric = task5.metrics(y, p)
    metric["bss_score"] = bss_score(y, p)
    return metric


def evaluate_calibrations(fold_preds: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    sorted_preds = sorted(fold_preds, key=lambda row: row["season"])
    for kind in ["none", "global_platt", "global_isotonic", "global_prior_correction"]:
        base_kind = kind.replace("global_", "")
        if kind == "none":
            cal = {"kind": "none"}
        else:
            cal = task5.fit_calibration(base_kind, np.concatenate([item["y"] for item in sorted_preds]), np.concatenate([item["pred"] for item in sorted_preds]))
        folds = []
        for item in sorted_preds:
            pred = task5.apply_calibration(cal, item["pred"])
            metric = fold_metric(item["y"], pred)
            metric["season"] = int(item["season"])
            folds.append(metric)
        result[kind] = {
            "folds": folds,
            "mean_brier": float(np.mean([m["brier"] for m in folds])),
            "worst_brier": float(np.max([m["brier"] for m in folds])),
            "mean_bss_score": float(np.mean([m["bss_score"] for m in folds])),
        }
    for kind in ["progressive_platt", "progressive_isotonic", "progressive_prior_correction"]:
        base_kind = kind.replace("progressive_", "")
        folds = []
        previous_y: list[np.ndarray] = []
        previous_p: list[np.ndarray] = []
        for item in sorted_preds:
            cal = task5.fit_calibration(base_kind, np.concatenate(previous_y), np.concatenate(previous_p)) if previous_y else {"kind": "none"}
            pred = task5.apply_calibration(cal, item["pred"])
            metric = fold_metric(item["y"], pred)
            metric["season"] = int(item["season"])
            folds.append(metric)
            previous_y.append(item["y"])
            previous_p.append(item["pred"])
        result[kind] = {
            "folds": folds,
            "mean_brier": float(np.mean([m["brier"] for m in folds])),
            "worst_brier": float(np.max([m["brier"] for m in folds])),
            "mean_bss_score": float(np.mean([m["bss_score"] for m in folds])),
        }
    return result


def initialize_schema() -> dict[str, Any]:
    schema = task5.load_v3_schema()
    task5.INPUT_COLUMNS = schema["input_columns"]
    task5.V3_FEATURE_COLUMNS = schema["feature_columns"]
    task5.V3_CATEGORICAL_COLUMNS = schema["categorical_columns"]
    return schema


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    schema = initialize_schema()
    alpha_default = float(schema.get("alpha", 50.0))
    train = load_training_frame()
    if args.smoke_rows:
        train = train.groupby("season", group_keys=False).head(args.smoke_rows).reset_index(drop=True)
        log(f"smoke mode: per-season rows={args.smoke_rows:,}")
    y = train[task5.TARGET_COL].to_numpy(dtype=int)
    tm_state = task51.build_trackman_state(train)
    tm_features = task51.build_trackman_features(train, tm_state, "mapping_all_shrink")
    seasons = pd.to_numeric(train["season"], errors="coerce").to_numpy()
    row_id = train[task5.ID_COL].to_numpy() if task5.ID_COL in train.columns else np.arange(len(train))
    candidate_names = args.candidates.split(",") if args.candidates else ["cat53_repro_seed42", "cat_seed_ensemble_3", "cat_current_alpha_100_seed3", "cat_season_stable_seed3"]
    candidate_names = [name.strip() for name in candidate_names if name.strip()]
    unknown = [name for name in candidate_names if name not in CANDIDATES]
    if unknown:
        raise ValueError(f"unknown candidates: {unknown}")
    log(f"loaded train={len(train):,}; trackman_rows={tm_state['trackman_rows']:,}; candidates={candidate_names}")

    all_fold_preds: dict[str, list[dict[str, Any]]] = {name: [] for name in candidate_names}
    fold_results: dict[str, list[dict[str, Any]]] = {name: [] for name in candidate_names}
    for valid_season in FOLD_SEASONS:
        tr_idx = np.flatnonzero(seasons < valid_season)
        va_idx = np.flatnonzero(seasons == valid_season)
        log(f"fold={valid_season} train={len(tr_idx):,} valid={len(va_idx):,}")
        for name in candidate_names:
            candidate = CANDIDATES[name]
            result_path = OUTPUT_DIR / f"fold_{valid_season}_{name}_result.json"
            oof_path = OUTPUT_DIR / f"oof_{name}_fold_{valid_season}.npz"
            if not args.force_retrain and result_path.exists() and oof_path.exists():
                cached = json.loads(result_path.read_text(encoding="utf-8"))
                if int(cached.get("train_rows", -1)) == len(tr_idx) and int(cached.get("valid_rows", -1)) == len(va_idx):
                    data = np.load(oof_path, allow_pickle=True)
                    fold_results[name].append(cached)
                    all_fold_preds[name].append(
                        {
                            "season": int(valid_season),
                            "y": data["y_true"],
                            "pred": data["raw_pred"],
                            "row_id": data["row_id"],
                        }
                    )
                    log(f"  skip cached {name} fold={valid_season} brier={cached['metrics']['brier']:.8f}")
                    continue
            alpha = float(candidate["alpha"] if candidate["alpha"] is not None else alpha_default)
            state = fit_feature_state_58(train.iloc[tr_idx], y[tr_idx], alpha)
            base_features = build_v58_features(train, state, tm_features)
            x_all, cats = prepare_features(base_features, tm_features, candidate)
            x_train = x_all.iloc[tr_idx]
            x_valid = x_all.iloc[va_idx]
            seed_preds: list[np.ndarray] = []
            seed_meta: list[dict[str, Any]] = []
            start = time.time()
            log(f"  candidate={name} seeds={candidate['seeds']} features={x_all.shape[1]} cats={len(cats)} alpha={alpha}")
            for seed in candidate["seeds"]:
                model, pred, meta = fit_cat_model(x_train, y[tr_idx], x_valid, y[va_idx], cats, int(seed))
                assert pred is not None
                seed_preds.append(pred)
                seed_meta.append(meta)
                model_path = MODEL_DIR / f"fold_{valid_season}_{name}_seed_{seed}.cbm"
                model.save_model(model_path)
            pred = np.mean(seed_preds, axis=0)
            metric = fold_metric(y[va_idx], pred)
            elapsed = time.time() - start
            record = {
                "candidate": name,
                "season": int(valid_season),
                "train_rows": int(len(tr_idx)),
                "valid_rows": int(len(va_idx)),
                "metrics": metric,
                "model_meta": seed_meta,
                "elapsed_sec": elapsed,
                "feature_count": int(x_all.shape[1]),
                "categorical_feature_count": int(len(cats)),
            }
            fold_results[name].append(record)
            all_fold_preds[name].append({"season": int(valid_season), "y": y[va_idx], "pred": pred, "row_id": row_id[va_idx]})
            np.savez_compressed(
                OUTPUT_DIR / f"oof_{name}_fold_{valid_season}.npz",
                row_id=row_id[va_idx],
                season=np.full(len(va_idx), valid_season),
                y_true=y[va_idx],
                raw_pred=pred,
            )
            write_json(OUTPUT_DIR / f"fold_{valid_season}_{name}_result.json", record)
            log(f"  done {name} fold={valid_season} brier={metric['brier']:.8f} auc={metric['auc']:.6f} bss={metric['bss_score']:.2f} elapsed={elapsed:.1f}s")

    raw_candidates = {}
    calibrated_candidates = {}
    calibrations = {}
    for name, records in fold_results.items():
        folds = [dict(r["metrics"], season=r["season"]) for r in records]
        raw_candidates[name] = {
            "folds": folds,
            "mean_brier": float(np.mean([m["brier"] for m in folds])),
            "worst_brier": float(np.max([m["brier"] for m in folds])),
            "mean_bss_score": float(np.mean([m["bss_score"] for m in folds])),
        }
        calibrations[name] = evaluate_calibrations(all_fold_preds[name])
        eligible_calibrations = {k: v for k, v in calibrations[name].items() if "platt" in k}
        best_cal = min(eligible_calibrations, key=lambda k: (eligible_calibrations[k]["mean_brier"], eligible_calibrations[k]["worst_brier"]))
        calibrated_candidates[name] = {
            **eligible_calibrations[best_cal],
            "selected_calibration": best_cal,
            "raw_mean_brier": raw_candidates[name]["mean_brier"],
            "raw_worst_brier": raw_candidates[name]["worst_brier"],
        }
    ranked = sorted(calibrated_candidates.items(), key=lambda pair: (pair[1]["mean_brier"], pair[1]["worst_brier"]))
    best_name, best_metrics = ranked[0]
    recommended = best_metrics["mean_brier"] < CAT_ONLY_REFERENCE_MEAN
    report = {
        "training_completed": True,
        "settings": {"folds": FOLD_SEASONS, "catboost_only": True, "lgbm_used": False, "cat_config": CAT_CONFIG, "data_mode": "full" if not args.smoke_rows else "smoke"},
        "reference": {"name": "5-3 CatBoost-only", "mean_brier": CAT_ONLY_REFERENCE_MEAN, "fold_brier": CAT_ONLY_REFERENCE_FOLDS},
        "candidates": {name: CANDIDATES[name] for name in candidate_names},
        "raw_candidates": raw_candidates,
        "calibrations": calibrations,
        "calibrated_candidates": calibrated_candidates,
        "best_candidate": best_name,
        "best_metrics": best_metrics,
        "submission_recommended": bool(recommended and not args.smoke_rows),
    }
    write_json(OUTPUT_DIR / "validation_results.json", report)
    write_summary(report)
    return report


def fit_final_submit(best_name: str, calibration_kind: str) -> None:
    schema = initialize_schema()
    train = load_training_frame()
    y = train[task5.TARGET_COL].to_numpy(dtype=int)
    tm_state = task51.build_trackman_state(train)
    tm_features = task51.build_trackman_features(train, tm_state, "mapping_all_shrink")
    candidate = CANDIDATES[best_name]
    alpha = float(candidate["alpha"] if candidate["alpha"] is not None else schema.get("alpha", 50.0))
    state = fit_feature_state_58(train, y, alpha)
    base_features = build_v58_features(train, state, tm_features)
    x_all, cats = prepare_features(base_features, tm_features, candidate)
    lowfreq_values: dict[str, list[str]] = {}
    x_final = x_all
    components = []
    for seed in candidate["seeds"]:
        model, _, _ = fit_cat_model(x_final, y, None, None, cats, int(seed))
        artifact = task5.model_artifact(model, "catboost", BASE_TRACKMAN_SET, state, x_final, cats, lowfreq_values, None, 1.0 / len(candidate["seeds"]), False)
        artifact["seed"] = int(seed)
        artifact["trackman_state"] = tm_state
        components.append(artifact)
    bundle = {
        "version": "5-8-catboost-only",
        "input_columns": schema["input_columns"],
        "components": components,
        "weights": {"catboost_total": 1.0, "lightgbm_total": 0.0},
        "selected_candidate": best_name,
        "selected_calibration": calibration_kind,
        "feature_variant": BASE_TRACKMAN_SET,
    }
    validation = json.loads((OUTPUT_DIR / "validation_results.json").read_text(encoding="utf-8"))
    oofs = []
    for fold in FOLD_SEASONS:
        data = np.load(OUTPUT_DIR / f"oof_{best_name}_fold_{fold}.npz")
        oofs.append((data["y_true"], data["raw_pred"]))
    cal = task5.fit_calibration(calibration_kind.replace("global_", "").replace("progressive_", ""), np.concatenate([x[0] for x in oofs]), np.concatenate([x[1] for x in oofs]))
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_DIR / "final_model.joblib")
    joblib.dump(cal, MODEL_DIR / "calibration_model.joblib")
    feature_schema = {
        "feature_columns": list(x_final.columns),
        "categorical_columns": cats,
        "input_columns": schema["input_columns"],
        "target_col": task5.TARGET_COL,
        "alpha": alpha,
        "catboost_only": True,
        "lgbm_used": False,
        "selected_candidate": best_name,
        "selected_calibration": calibration_kind,
        "validation_mean_brier": validation["best_metrics"]["mean_brier"],
    }
    write_json(MODEL_DIR / "feature_schema.json", feature_schema)
    zip_path = ROOT / "submit.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in ["final_model.joblib", "calibration_model.joblib", "feature_schema.json"]:
            zf.write(MODEL_DIR / name, f"model/{name}")
        zf.write(ROOT / "script.py", "script.py")
        zf.write(ROOT / "requirements.txt", "requirements.txt")
    log(f"submit.zip saved: {zip_path}")


def write_summary(report: dict[str, Any]) -> None:
    lines = [
        "# 5-8 CatBoost-only 개선 실험",
        "",
        "## 실행 요약",
        "",
        "- 목표: `/5-3` CatBoost-only 모델 성능 개선.",
        "- LGBM: 사용하지 않음.",
        "- 최종 후보 선택: no calibration 제외, calibration 적용 후보만 비교.",
        f"- 기준 `/5-3 cat only` calibrated mean Brier: `{CAT_ONLY_REFERENCE_MEAN:.8f}`.",
        f"- best candidate: `{report['best_candidate']}`.",
        f"- best calibration: `{report['best_metrics']['selected_calibration']}`.",
        f"- best mean Brier: `{report['best_metrics']['mean_brier']:.8f}`.",
        f"- 기준 대비 차이: `{report['best_metrics']['mean_brier'] - CAT_ONLY_REFERENCE_MEAN:+.8f}`.",
        f"- submission_recommended: `{str(report['submission_recommended']).lower()}`.",
        "",
        "## 후보 설계",
        "",
    ]
    for name in report["candidates"]:
        lines.append(f"- `{name}`: {CANDIDATES[name]['description']}.")
    omitted = [name for name in CANDIDATES if name not in report["candidates"]]
    if omitted:
        lines.append(f"- 이번 full 복구 실행 제외 후보: `{', '.join(omitted)}`.")
    lines.extend([
        "",
        "## Feature/전처리",
        "",
        f"- 기본 feature: `/5` v3 schema + 현재시즌 파생피처 + `{BASE_TRACKMAN_SET}`.",
        f"- Trackman prior feature 수: `{len(FEATURE_SETS[BASE_TRACKMAN_SET])}`.",
        "- 현재 투구 Trackman 측정값, 2025 Trackman, test 내부 groupby/rolling/분포/사후보정은 사용하지 않는다.",
        "- script.py는 저장된 train-time state와 row별 입력만 사용한다.",
        "",
        "## 후보별 결과",
        "",
        "| candidate | calibration | raw mean Brier | calibrated mean Brier | worst Brier | mean BSS | vs 5-3 cat only |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for name, item in sorted(report["calibrated_candidates"].items(), key=lambda pair: (pair[1]["mean_brier"], pair[1]["worst_brier"])):
        lines.append(
            f"| {name} | {item['selected_calibration']} | {item['raw_mean_brier']:.8f} | {item['mean_brier']:.8f} | {item['worst_brier']:.8f} | {item['mean_bss_score']:.2f} | {item['mean_brier'] - CAT_ONLY_REFERENCE_MEAN:+.8f} |"
        )
    lines.extend(["", "## Fold별 결과", ""])
    for name, item in sorted(report["calibrated_candidates"].items(), key=lambda pair: (pair[1]["mean_brier"], pair[1]["worst_brier"])):
        lines.append(f"### {name}")
        lines.append("| season | Brier | LogLoss | AUC | mean_pred | mean_target | BSS |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for fold in item["folds"]:
            lines.append(f"| {fold['season']} | {fold['brier']:.8f} | {fold['log_loss']:.8f} | {fold['auc']:.6f} | {fold['mean_pred']:.6f} | {fold['mean_target']:.6f} | {fold['bss_score']:.2f} |")
        lines.append("")
    lines.extend([
        "## 판단",
        "",
        "- `5-3-4`는 서버 점수가 낮았으므로 이 실험의 주 타깃에서 제외했다.",
        "- 목표는 CatBoost-only 자체의 안정적 개선이며, no calibration은 최종 후보에서 제외했다.",
        "- 개선폭이 작으면 서버 일반화 리스크가 크므로 submit 추천 여부는 `submission_recommended` 값으로 판단한다.",
    ])
    (ROOT / "FINAL_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_package() -> dict[str, Any]:
    zip_path = ROOT / "submit.zip"
    result = {"exists": zip_path.exists()}
    if not zip_path.exists():
        return result
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        names = sorted(zf.namelist())
    result.update({"zip_test": bad is None, "members": names[:20]})
    tmp = Path("/tmp") / f"validate_5_8_{int(time.time())}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
    (tmp / "data").mkdir(exist_ok=True)
    shutil.copy2(DATA_DIR / "test.csv", tmp / "data" / "test.csv")
    shutil.copy2(DATA_DIR / "sample_submission.csv", tmp / "data" / "sample_submission.csv")
    command = f"cd {tmp} && python script.py"
    code = os.system(command)
    sub_path = tmp / "output" / "submission.csv"
    result["script_exit_code"] = int(code)
    result["submission_exists"] = sub_path.exists()
    if sub_path.exists():
        sub = pd.read_csv(sub_path)
        result["columns_ok"] = list(sub.columns) == ["row_id", "control_success"]
        result["row_count"] = int(len(sub))
        result["finite"] = bool(np.isfinite(sub["control_success"].to_numpy()).all())
        result["range_ok"] = bool(sub["control_success"].between(0, 1).all())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5-8 CatBoost-only improvement experiments")
    parser.add_argument("--smoke-rows", type=int, default=0, help="Use first N rows per season only for smoke testing.")
    parser.add_argument("--make-submit", action="store_true", help="Train final full model and create submit.zip if validation recommends it.")
    parser.add_argument("--force-submit", action="store_true", help="Create submit.zip even if validation does not beat baseline.")
    parser.add_argument("--validate-submit", action="store_true", help="Run package validation after submit.zip creation.")
    parser.add_argument("--candidates", default="", help="Comma-separated candidate names. Defaults to the core 4 candidates.")
    parser.add_argument("--force-retrain", action="store_true", help="Ignore cached fold result/oof files.")
    parser.add_argument("--iterations", type=int, default=None, help="Override CatBoost iteration count for this run.")
    parser.add_argument("--disable-early-stopping", action="store_true", help="Train exactly --iterations trees during validation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations is not None:
        if args.iterations < 1:
            raise ValueError("--iterations must be positive")
        CAT_CONFIG["iterations"] = int(args.iterations)
    fit_cat_model.disable_early_stopping = bool(args.disable_early_stopping)
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    report = run_validation(args)
    print("[FINAL SUMMARY]", flush=True)
    print(f"training_completed={report['training_completed']}", flush=True)
    print(f"model=CatBoost-only", flush=True)
    print(f"best_candidate={report['best_candidate']}", flush=True)
    print(f"best_calibration={report['best_metrics']['selected_calibration']}", flush=True)
    print(f"mean_brier={report['best_metrics']['mean_brier']:.8f}", flush=True)
    print(f"local_converted_score={report['best_metrics']['mean_bss_score']:.4f}", flush=True)
    print(f"submission_recommended={str(report['submission_recommended']).lower()}", flush=True)
    if args.make_submit and (report["submission_recommended"] or args.force_submit) and not args.smoke_rows:
        fit_final_submit(report["best_candidate"], report["best_metrics"]["selected_calibration"])
        if args.validate_submit:
            validation = validate_package()
            write_json(OUTPUT_DIR / "submit_validation.json", validation)
            log(f"submit validation: {validation}")


if __name__ == "__main__":
    main()
