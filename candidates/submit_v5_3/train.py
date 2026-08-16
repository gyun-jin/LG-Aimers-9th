from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier


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


V5_REFERENCE_MEAN = 0.24717070
V5_REFERENCE_WORST = 0.24979486
V5_REFERENCE_FOLDS = [0.24342932, 0.24979486, 0.24828793]
FOLD_SEASONS = [2022, 2023, 2024]

CAT_CONFIGS = {
    "cat_current": {"iterations": 520, "learning_rate": 0.03, "depth": 8, "l2_leaf_reg": 20.0},
}
LGB_CONFIGS = {
    "lgb_current": {"n_estimators": 520, "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 100, "feature_fraction": 0.85, "bagging_fraction": 0.85, "reg_lambda": 10.0},
}

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
    "v5_no_trackman_recheck": [],
    "tm_metadata_only": TM_METADATA_STABLE,
    "tm_metadata_pitchmix": TM_METADATA_STABLE + TM_PITCH_STABLE,
    "tm_metadata_physical_core": TM_METADATA_STABLE + TM_PHYSICAL_STABLE,
    "tm_metadata_physical_pitchmix": TM_METADATA_STABLE + TM_PHYSICAL_STABLE + TM_PITCH_STABLE,
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


def load_training_frame(clean: bool = True) -> pd.DataFrame:
    path = HAND_CLEAN_TRAIN_PATH if clean else DATA_DIR / "train.csv"
    return pd.read_csv(path)


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
    table = pd.DataFrame(rows)
    return {
        "alpha": float(alpha),
        "target_prior": prior,
        "prior_table": table,
    }


def fit_feature_state_53(frame: pd.DataFrame, y: np.ndarray, alpha: float) -> dict[str, Any]:
    state = task5.fit_feature_state(frame, y, alpha)
    state["current_season_state"] = build_current_season_state(frame, y, alpha)
    return state


def _rate_bin(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return pd.cut(
        values,
        bins=[-np.inf, 0.35, 0.45, 0.55, 0.65, np.inf],
        labels=["very_low", "low", "mid", "high", "very_high"],
    ).astype("string").fillna("missing").astype(str)


def add_current_season_features(features: pd.DataFrame, state: dict[str, Any], tm_features: pd.DataFrame | None = None) -> pd.DataFrame:
    out = features.copy()
    cs_state = state["current_season_state"]
    table = cs_state["prior_table"]
    rows = out[["pitcher_id", "season", "asof_pitcher_n", "asof_pitcher_success_rate"]].copy()
    rows = rows.merge(table, on=["pitcher_id", "season"], how="left")
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
    out["pitcher_current_season_success_rate_bin"] = _rate_bin(smoothed)
    li = pd.to_numeric(out["li"], errors="coerce").fillna(0.0)
    out["li_bin"] = pd.cut(li, [-np.inf, 0.7, 1.5, np.inf], labels=["low", "mid", "high"]).astype("string").fillna("mid").astype(str)
    rate_bin = out["pitcher_current_season_success_rate_bin"]
    out["pitcher_current_season_success_x_count_state"] = rate_bin + "_count_" + task5.safe_string(out["count_state"])
    out["pitcher_current_season_success_x_game_type"] = rate_bin + "_game_" + task5.safe_string(out["game_type"])
    out["pitcher_current_season_success_x_li_bin"] = rate_bin + "_li_" + task5.safe_string(out["li_bin"])
    out["pitcher_current_season_success_x_base_state"] = rate_bin + "_base_" + task5.safe_string(out["base_state"])
    out["pitcher_current_season_success_x_hand_matchup"] = rate_bin + "_hand_" + task5.safe_string(out["hand_matchup"])
    if tm_features is not None and "tm_has_mapping" in tm_features.columns:
        out["pitcher_current_season_success_x_tm_available"] = smoothed * pd.to_numeric(tm_features["tm_has_mapping"], errors="coerce").fillna(0.0)
    else:
        out["pitcher_current_season_success_x_tm_available"] = 0.0
    if tm_features is not None and "tm_mapping_confidence_bucket" in tm_features.columns:
        bucket = task5.safe_string(tm_features["tm_mapping_confidence_bucket"])
    else:
        bucket = pd.Series(["none_low"] * len(out), index=out.index)
    out["pitcher_current_season_success_x_tm_mapping_confidence_bucket"] = rate_bin + "_tm_" + bucket
    return out


def build_v53_features(frame: pd.DataFrame, state: dict[str, Any], tm_features: pd.DataFrame | None = None) -> pd.DataFrame:
    base = task5.build_v3_features(frame, state)
    return add_current_season_features(base, state, tm_features)


def log(message: str) -> None:
    print(f"[task5-2] {time.strftime('%H:%M:%S')} {message}", flush=True)


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


def fit_cat(x_train: pd.DataFrame, y_train: np.ndarray, x_valid: pd.DataFrame, y_valid: np.ndarray, cats: list[str], config: dict[str, Any], seed: int = 42) -> np.ndarray:
    train_x = x_train.copy()
    valid_x = x_valid.copy()
    for col in cats:
        train_x[col] = task5.safe_string(train_x[col])
        valid_x[col] = task5.safe_string(valid_x[col])
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="BrierScore",
        iterations=config["iterations"],
        learning_rate=config["learning_rate"],
        depth=config["depth"],
        l2_leaf_reg=config["l2_leaf_reg"],
        random_seed=seed,
        od_type="Iter",
        od_wait=60,
        allow_writing_files=False,
        verbose=False,
        thread_count=min(12, os.cpu_count() or 1),
    )
    model.fit(Pool(train_x, y_train, cat_features=cats), eval_set=Pool(valid_x, y_valid, cat_features=cats))
    return model.predict_proba(valid_x)[:, 1]


def fit_lgb(x_train: pd.DataFrame, y_train: np.ndarray, x_valid: pd.DataFrame, cats: list[str], config: dict[str, Any], seed: int = 42) -> np.ndarray:
    train_x, maps = task5.encode_for_lightgbm(x_train, cats)
    valid_x, _ = task5.encode_for_lightgbm(x_valid, cats, maps)
    model = LGBMClassifier(
        objective="binary",
        n_estimators=config["n_estimators"],
        learning_rate=config["learning_rate"],
        num_leaves=config["num_leaves"],
        min_child_samples=config["min_child_samples"],
        feature_fraction=config["feature_fraction"],
        bagging_fraction=config["bagging_fraction"],
        bagging_freq=1,
        reg_lambda=config["reg_lambda"],
        random_state=seed,
        n_jobs=min(12, os.cpu_count() or 1),
        verbosity=-1,
    )
    model.fit(train_x, y_train, callbacks=[])
    return model.predict_proba(valid_x)[:, 1]


def sample_indices(index: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(index) <= limit:
        return index
    return np.sort(np.random.default_rng(seed).choice(index, size=limit, replace=False))


def prepare_features(base_features: pd.DataFrame, tm_features: pd.DataFrame | None, feature_set: str, use_current_season: bool = True) -> tuple[pd.DataFrame, list[str]]:
    base_cols = list(task5.V3_FEATURE_COLUMNS or [])
    frame = base_features.loc[:, base_cols].copy()
    if use_current_season:
        frame = pd.concat([frame, base_features.loc[:, CURRENT_SEASON_FEATURES + CURRENT_SEASON_CATEGORICAL_FEATURES]], axis=1)
    selected_tm = FEATURE_SETS[feature_set]
    if selected_tm:
        if tm_features is None:
            raise ValueError("tm_features is required")
        frame = pd.concat([frame, tm_features.loc[:, selected_tm]], axis=1)
    x = frame.drop(columns=[c for c in ["pitcher_id", "batter_id"] if c in frame.columns])
    cats = [c for c in task5.V3_CATEGORICAL_COLUMNS or [] if c in x.columns]
    if "tm_mapping_confidence_bucket" in x.columns and "tm_mapping_confidence_bucket" not in cats:
        cats.append("tm_mapping_confidence_bucket")
    if use_current_season:
        for col in CURRENT_SEASON_CATEGORICAL_FEATURES:
            if col in x.columns and col not in cats:
                cats.append(col)
    return x, cats


def evaluate_calibrations(fold_preds: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for kind in ["none", "platt", "isotonic", "prior_correction"]:
        fold_metrics = []
        previous_y: list[np.ndarray] = []
        previous_p: list[np.ndarray] = []
        for item in fold_preds:
            if previous_y:
                cal = task5.fit_calibration(kind, np.concatenate(previous_y), np.concatenate(previous_p))
            else:
                cal = {"kind": "none"}
            pred = task5.apply_calibration(cal, item["pred"])
            fold_metrics.append(task5.metrics(item["y"], pred))
            previous_y.append(item["y"])
            previous_p.append(item["pred"])
        result[kind] = {
            "folds": fold_metrics,
            "mean_brier": float(np.mean([m["brier"] for m in fold_metrics])),
            "worst_brier": float(np.max([m["brier"] for m in fold_metrics])),
        }
    return result


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [record["metrics"] for record in records]
    return {
        "folds": metrics,
        "mean_brier": float(np.mean([m["brier"] for m in metrics])),
        "worst_brier": float(np.max([m["brier"] for m in metrics])),
        "mean_metrics": {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]},
    }


def run_quick_experiment() -> dict[str, Any]:
    schema = task5.load_v3_schema()
    task5.INPUT_COLUMNS = schema["input_columns"]
    task5.V3_FEATURE_COLUMNS = schema["feature_columns"]
    task5.V3_CATEGORICAL_COLUMNS = schema["categorical_columns"]
    train = load_training_frame(clean=True)
    y = train[task5.TARGET_COL].to_numpy(dtype=int)
    tm_state = task51.build_trackman_state(train)
    log(f"loaded train={len(train):,}; trackman_rows={tm_state['trackman_rows']:,}")

    records: dict[str, list[dict[str, Any]]] = {name: [] for name in FEATURE_SETS}
    raw_fold_preds: dict[str, list[dict[str, Any]]] = {name: [] for name in FEATURE_SETS}
    for valid_season in FOLD_SEASONS:
        seasons = train["season"].to_numpy()
        tr_idx = np.flatnonzero(seasons < valid_season)
        va_idx = np.flatnonzero(seasons == valid_season)
        tr_sample = sample_indices(tr_idx, 180_000, 7000 + valid_season)
        va_sample = sample_indices(va_idx, 70_000, 8000 + valid_season)
        tm_features = task51.build_trackman_features(train, tm_state, "mapping_all_shrink")
        state = fit_feature_state_53(train.iloc[tr_idx], y[tr_idx], schema["alpha"])
        base_features = build_v53_features(train, state, tm_features)
        y_valid = y[va_sample]
        log(f"fold={valid_season} train={len(tr_sample):,}/{len(tr_idx):,} valid={len(va_sample):,}/{len(va_idx):,}")
        for feature_set in FEATURE_SETS:
            x_all, cats = prepare_features(base_features, tm_features, feature_set, use_current_season=True)
            x_train = x_all.iloc[tr_sample]
            x_valid = x_all.iloc[va_sample]
            cat_pred = fit_cat(x_train, y[tr_sample], x_valid, y_valid, cats, CAT_CONFIGS["cat_current"])
            lgb_pred = fit_lgb(x_train, y[tr_sample], x_valid, cats, LGB_CONFIGS["lgb_current"])
            pred = 0.8 * cat_pred + 0.2 * lgb_pred
            metric = task5.metrics(y_valid, pred)
            records[feature_set].append({"season": valid_season, "metrics": metric})
            raw_fold_preds[feature_set].append({"season": valid_season, "y": y_valid, "pred": pred})
            log(f"  {feature_set} brier={metric['brier']:.8f} auc={metric['auc']:.6f} mean_pred={metric['mean_pred']:.6f}")

    candidates = {name: summarize(items) for name, items in records.items()}
    calibrations = {name: evaluate_calibrations(raw_fold_preds[name]) for name in FEATURE_SETS}
    calibrated_candidates: dict[str, Any] = {}
    for name in FEATURE_SETS:
        best_cal = min(calibrations[name], key=lambda kind: (calibrations[name][kind]["mean_brier"], calibrations[name][kind]["worst_brier"]))
        calibrated_candidates[name] = {
            **calibrations[name][best_cal],
            "selected_calibration": best_cal,
            "raw_mean_brier": candidates[name]["mean_brier"],
            "raw_worst_brier": candidates[name]["worst_brier"],
        }
    ranked = sorted(calibrated_candidates.items(), key=lambda item: (item[1]["mean_brier"], item[1]["worst_brier"]))
    eligible = [
        (name, item)
        for name, item in ranked
        if item["mean_brier"] < V5_REFERENCE_MEAN and item["worst_brier"] <= V5_REFERENCE_WORST
    ]
    selected = eligible[0] if eligible else ranked[0]
    return {
        "settings": {"folds": FOLD_SEASONS, "train_sample": 180000, "valid_sample": 70000, "cat_weight": 0.8, "lgb_weight": 0.2},
        "references": {"v5_mean_brier": V5_REFERENCE_MEAN, "v5_worst_brier": V5_REFERENCE_WORST, "v5_folds": V5_REFERENCE_FOLDS},
        "feature_sets": FEATURE_SETS,
        "raw_candidates": candidates,
        "calibrations": calibrations,
        "calibrated_candidates": calibrated_candidates,
        "best_candidate": ranked[0][0],
        "best_metrics": ranked[0][1],
        "eligible_candidates": {name: item for name, item in eligible},
        "selected_candidate": selected[0],
        "selected_metrics": selected[1],
    }


def write_report(report: dict[str, Any]) -> None:
    lines = [
        "# 5-2 Trackman 제한 피처 실험",
        "",
        "## 결론",
        "",
        f"- best candidate: `{report['best_candidate']}`",
        f"- selected calibration: `{report['best_metrics']['selected_calibration']}`",
        f"- best mean Brier: `{report['best_metrics']['mean_brier']:.8f}`",
        f"- best worst Brier: `{report['best_metrics']['worst_brier']:.8f}`",
        f"- stable selected candidate: `{report['selected_candidate']}`",
        f"- stable selected calibration: `{report['selected_metrics']['selected_calibration']}`",
        f"- stable selected mean/worst Brier: `{report['selected_metrics']['mean_brier']:.8f}` / `{report['selected_metrics']['worst_brier']:.8f}`",
        f"- `/5` reference mean/worst: `{V5_REFERENCE_MEAN:.8f}` / `{V5_REFERENCE_WORST:.8f}`",
        f"- stable selected `/5` 대비 mean 차이: `{report['selected_metrics']['mean_brier'] - V5_REFERENCE_MEAN:+.8f}`",
        f"- stable selected `/5` 대비 worst 차이: `{report['selected_metrics']['worst_brier'] - V5_REFERENCE_WORST:+.8f}`",
        "",
        "## 설계",
        "",
        "- `/5`와 같은 no_ids CatBoost 0.8 + LightGBM 0.2 구조를 유지했다.",
        "- `/5`와 같은 quick validation sample 크기인 train 180,000 / valid 70,000을 사용했다.",
        "- Trackman은 `mapping_all_shrink` 방식으로 prior-season pitcher summary만 만들었다.",
        "- 현재 투구 단위 Trackman 결합, 2025 Trackman, test 내부 통계는 사용하지 않았다.",
        "",
        "## 후보별 결과",
        "",
        "| feature set | selected calibration | mean Brier | worst Brier | raw mean Brier | raw worst Brier | mean AUC | mean_pred | mean_target |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in sorted(report["calibrated_candidates"].items(), key=lambda pair: (pair[1]["mean_brier"], pair[1]["worst_brier"])):
        raw = report["raw_candidates"][name]["mean_metrics"]
        lines.append(f"| {name} | {item['selected_calibration']} | {item['mean_brier']:.8f} | {item['worst_brier']:.8f} | {item['raw_mean_brier']:.8f} | {item['raw_worst_brier']:.8f} | {raw['auc']:.6f} | {raw['mean_pred']:.6f} | {raw['mean_target']:.6f} |")
    lines.extend(["", "## Feature set 구성", ""])
    for name, cols in report["feature_sets"].items():
        lines.append(f"### {name}")
        if cols:
            for col in cols:
                lines.append(f"- `{col}`")
        else:
            lines.append("- Trackman 피처 없음. `/5` no_ids baseline 재확인용.")
        lines.append("")
    recommendation = "전체 fold 재검증 필요" if report["eligible_candidates"] else "제출 비추천"
    lines.extend([
        "## 판단",
        "",
        f"- 판단: `{recommendation}`",
        "- `/5` 기준을 평균과 최악 fold에서 동시에 넘어야 `submit.zip`을 만들 후보로 본다.",
        f"- quick 기준 eligible 후보: `{list(report['eligible_candidates'])}`",
    ])
    (ROOT / "trackman_limited_experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    report = run_quick_experiment()
    write_json(OUTPUT_DIR / "trackman_limited_experiment_results.json", report)
    write_report(report)
    log(f"best={report['best_candidate']} mean={report['best_metrics']['mean_brier']:.8f} worst={report['best_metrics']['worst_brier']:.8f}")
    log(f"selected={report['selected_candidate']} mean={report['selected_metrics']['mean_brier']:.8f} worst={report['selected_metrics']['worst_brier']:.8f}")
    if not report["eligible_candidates"]:
        (ROOT / "submit_not_created.txt").write_text(
            "5-2 Trackman 제한 피처 후보가 /5 기준을 넘지 못해 submit.zip을 만들지 않음.\n"
            f"best_candidate={report['best_candidate']}\n"
            f"best_mean_brier={report['best_metrics']['mean_brier']:.8f}\n"
            f"best_worst_brier={report['best_metrics']['worst_brier']:.8f}\n"
            f"v5_reference_mean_brier={V5_REFERENCE_MEAN:.8f}\n"
            f"v5_reference_worst_brier={V5_REFERENCE_WORST:.8f}\n",
            encoding="utf-8",
        )
    else:
        (ROOT / "submit_not_created.txt").write_text(
            "5-2 quick 검증에서 /5 기준을 넘는 후보가 있어 전체 fold 재검증이 필요함. 아직 submit.zip은 만들지 않음.\n"
            f"selected_candidate={report['selected_candidate']}\n"
            f"selected_mean_brier={report['selected_metrics']['mean_brier']:.8f}\n"
            f"selected_worst_brier={report['selected_metrics']['worst_brier']:.8f}\n"
            f"eligible_candidates={list(report['eligible_candidates'])}\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
