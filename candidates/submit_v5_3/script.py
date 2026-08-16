"""Independent inference entry point for the Task5 submission."""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"
RATE_COUNT_MAP = {
    "asof_pitcher_success_rate": "asof_pitcher_n",
    "asof_pitcher_reverse_rate": "asof_pitcher_n",
    "asof_pitcher_middle_rate": "asof_pitcher_n",
    "asof_pitcher_ball_rate": "asof_pitcher_n",
    "asof_pitcher_strike_rate": "asof_pitcher_n",
    "asof_batter_success_rate": "asof_batter_n",
    "asof_batter_middle_rate": "asof_batter_n",
    "asof_pitcher_fastball_rate": "asof_pitcher_pitchmix_n",
    "asof_pitcher_breaking_rate": "asof_pitcher_pitchmix_n",
    "asof_pitcher_offspeed_rate": "asof_pitcher_pitchmix_n",
}
REDUNDANT_COLUMNS = {"asof_pitcher_pitchmix_n", "run_total_before", "num_runners_on", "away_win_expectancy", "asof_pitcher_offspeed_rate"}
TM_PHYSICAL_SOURCE_COLUMNS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break", "rel_side", "rel_height", "extension", "zone_speed"]
TM_PHYSICAL_OUTPUT_NAMES = {"rel_speed": "release_speed", "spin_rate": "spin_rate", "induced_vert_break": "induced_vertical_break", "horz_break": "horizontal_break", "rel_side": "release_pos_x", "rel_height": "release_pos_z", "extension": "extension", "zone_speed": "zone_speed"}
TM_METADATA_COLUMNS = ["tm_has_mapping", "tm_has_pitcher_history", "tm_is_low_history", "tm_is_rookie_or_no_history", "tm_hist_pitch_count", "tm_hist_game_count", "tm_hist_season_count", "tm_mapping_exact_game_count", "tm_mapping_margin", "tm_mapping_ratio", "tm_mapping_n_candidates", "tm_mapping_confidence", "tm_mapping_confidence_bucket", "tm_mapping_is_high_confidence", "tm_mapping_selected"]
TM_PHYSICAL_COLUMNS = [f"tm_hist_{TM_PHYSICAL_OUTPUT_NAMES[source]}_{stat}_shrunk" for source in TM_PHYSICAL_SOURCE_COLUMNS for stat in ("mean", "median", "std")]
TM_PITCH_COLUMNS = ["tm_hist_pitch_group_fastball_shrunk", "tm_hist_pitch_group_breaking_shrunk", "tm_hist_pitch_group_offspeed_shrunk", "tm_hist_pitch_group_other_shrunk", "tm_hist_pitch_type_entropy"]
TM_FEATURE_COLUMNS = TM_METADATA_COLUMNS + TM_PHYSICAL_COLUMNS + TM_PITCH_COLUMNS
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


def safe_string(series):
    return series.astype("string").fillna("__MISSING__").astype(str)


def rate_bin(series):
    values = pd.to_numeric(series, errors="coerce")
    return pd.cut(
        values,
        bins=[-np.inf, 0.35, 0.45, 0.55, 0.65, np.inf],
        labels=["very_low", "low", "mid", "high", "very_high"],
    ).astype("string").fillna("missing").astype(str)


def build_features(frame, state):
    cols = state["input_columns"]
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"missing input columns: {missing}")
    x = frame.loc[:, cols].copy()
    x["hand_matchup"] = safe_string(x["pitcher_hand"]) + "_" + safe_string(x["batter_hand"])
    x["count_state"] = safe_string(x["balls_before"]) + "-" + safe_string(x["strikes_before"])
    x["runner_out_state"] = safe_string(x["base_state"]) + "_o" + safe_string(x["outs_before"])
    x["close_game"] = (x["score_diff_pitcher_team"].abs() <= 2).astype("int8")
    x["late_inning"] = (x["inning"] >= 7).astype("int8")
    high_li = (x["li"] >= 1.5).astype("int8")
    x["pressure_state"] = "late" + x["late_inning"].astype(str) + "_close" + x["close_game"].astype(str) + "_li" + high_li.astype(str)
    x["log1p_asof_pitcher_n"] = np.log1p(x["asof_pitcher_n"].clip(lower=0))
    x["log1p_asof_batter_n"] = np.log1p(x["asof_batter_n"].clip(lower=0))
    for rate_col, count_col in RATE_COUNT_MAP.items():
        count = pd.to_numeric(x[count_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        rate = pd.to_numeric(x[rate_col], errors="coerce")
        prior = float(state["rate_priors"][rate_col])
        alpha = float(state["alpha"])
        x[f"{rate_col}_smoothed"] = (count * rate.fillna(prior) + alpha * prior) / (count + alpha)
    success_cols = ["asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate"]
    middle_cols = ["asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate"]
    success = x[success_cols]
    middle = x[middle_cols]
    x["recent_success_mean_1_3_5"] = success.mean(axis=1, skipna=True)
    x["recent_success_range_1_3_5"] = success.max(axis=1, skipna=True) - success.min(axis=1, skipna=True)
    x["recent_success_std_1_3_5"] = success.std(axis=1, skipna=True, ddof=0)
    for window, col in zip((1, 3, 5), success_cols):
        x[f"recent_gap_{window}"] = x[col] - x["asof_pitcher_success_rate_smoothed"]
    x["recent_middle_mean_1_3_5"] = middle.mean(axis=1, skipna=True)
    x["recent_middle_range_1_3_5"] = middle.max(axis=1, skipna=True) - middle.min(axis=1, skipna=True)
    x["recent_middle_vs_cumulative"] = middle.mean(axis=1, skipna=True) - x["asof_pitcher_middle_rate_smoothed"]
    for col in state["missing_columns"]:
        x[f"{col}__missing"] = x[col].isna().astype("int8")
    x["is_pitcher_cold_start"] = (x["asof_pitcher_n"].fillna(0) <= 0).astype("int8")
    x["is_batter_cold_start"] = (x["asof_batter_n"].fillna(0) <= 0).astype("int8")
    mix = x[["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]].clip(0.0, 1.0)
    x["pitchmix_entropy"] = -(mix * np.log(mix.clip(lower=1e-12))).sum(axis=1, min_count=1)
    x["score_abs"] = x["score_diff_pitcher_team"].abs()
    x["is_pitcher_team_leading"] = (x["score_diff_pitcher_team"] > 0).astype("int8")
    x["is_pitcher_team_trailing"] = (x["score_diff_pitcher_team"] < 0).astype("int8")
    return x.drop(columns=[c for c in REDUNDANT_COLUMNS if c in x.columns])


def add_current_season_features(features, state, tm_features=None):
    out = features.copy()
    cs_state = state.get("current_season_state")
    if cs_state is None:
        return out
    table = cs_state["prior_table"]
    if not isinstance(table, pd.DataFrame):
        table = pd.DataFrame(table)
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
    out["pitcher_current_season_success_rate_bin"] = rate_bin(smoothed)
    li = pd.to_numeric(out["li"], errors="coerce").fillna(0.0)
    out["li_bin"] = pd.cut(li, [-np.inf, 0.7, 1.5, np.inf], labels=["low", "mid", "high"]).astype("string").fillna("mid").astype(str)
    rb = out["pitcher_current_season_success_rate_bin"]
    out["pitcher_current_season_success_x_count_state"] = rb + "_count_" + safe_string(out["count_state"])
    out["pitcher_current_season_success_x_game_type"] = rb + "_game_" + safe_string(out["game_type"])
    out["pitcher_current_season_success_x_li_bin"] = rb + "_li_" + safe_string(out["li_bin"])
    out["pitcher_current_season_success_x_base_state"] = rb + "_base_" + safe_string(out["base_state"])
    out["pitcher_current_season_success_x_hand_matchup"] = rb + "_hand_" + safe_string(out["hand_matchup"])
    if tm_features is not None and "tm_has_mapping" in tm_features.columns:
        out["pitcher_current_season_success_x_tm_available"] = smoothed * pd.to_numeric(tm_features["tm_has_mapping"], errors="coerce").fillna(0.0)
    else:
        out["pitcher_current_season_success_x_tm_available"] = 0.0
    if tm_features is not None and "tm_mapping_confidence_bucket" in tm_features.columns:
        bucket = safe_string(tm_features["tm_mapping_confidence_bucket"])
    else:
        bucket = pd.Series(["none_low"] * len(out), index=out.index)
    out["pitcher_current_season_success_x_tm_mapping_confidence_bucket"] = rb + "_tm_" + bucket
    return out


def build_trackman_features(frame, state, strategy):
    mapping = state["mapping"].copy()
    if strategy == "mapping_exact5_margin4_ratio5":
        selected = (mapping["exact_game_count"] >= 5) & (mapping["margin"] >= 4) & (mapping["ratio_safe"] >= 5)
    elif strategy == "mapping_exact10_ratio5":
        selected = (mapping["exact_game_count"] >= 10) & (mapping["ratio_safe"] >= 5)
    elif strategy == "mapping_exact20_ratio10":
        selected = (mapping["exact_game_count"] >= 20) & (mapping["ratio_safe"] >= 10)
    else:
        selected = mapping["pitcher_trackman_id"].notna()
    mapping["mapping_selected"] = selected.astype("int8")
    rows = frame[["pitcher_id", "season", "pitcher_hand"]].copy()
    rows["hand_code"] = safe_string(rows["pitcher_hand"])
    rows = rows.merge(mapping, on="pitcher_id", how="left")
    rows.loc[rows["mapping_selected"].fillna(0).eq(0), "pitcher_trackman_id"] = np.nan
    rows = rows.merge(state["lookup"], on=["pitcher_trackman_id", "season"], how="left")
    rows = rows.merge(state["prior_lookup"], on=["season", "hand_code"], how="left", suffixes=("", "_prior"))
    rows["tm_has_mapping"] = ((rows["mapping_selected"].fillna(0) > 0) & rows["pitcher_trackman_id"].notna()).astype("int8")
    rows["tm_has_pitcher_history"] = (pd.to_numeric(rows["tm_hist_pitch_count"], errors="coerce").fillna(0) > 0).astype("int8")
    rows["tm_is_low_history"] = ((rows["tm_hist_pitch_count"].fillna(0) > 0) & (rows["tm_hist_pitch_count"].fillna(0) < 20)).astype("int8")
    rows["tm_is_rookie_or_no_history"] = (rows["tm_hist_pitch_count"].fillna(0) <= 0).astype("int8")
    for col in ["tm_hist_pitch_count", "tm_hist_game_count", "tm_hist_season_count"]:
        rows[col] = pd.to_numeric(rows[col], errors="coerce").fillna(0.0)
    for col in ["exact_game_count", "margin", "ratio_safe", "n_candidates", "mapping_confidence", "mapping_selected"]:
        rows[col] = pd.to_numeric(rows[col], errors="coerce").fillna(0.0)
    rows = rows.rename(columns={"exact_game_count": "tm_mapping_exact_game_count", "margin": "tm_mapping_margin", "ratio_safe": "tm_mapping_ratio", "n_candidates": "tm_mapping_n_candidates"})
    rows["tm_mapping_confidence"] = rows["mapping_confidence"]
    rows["tm_mapping_is_high_confidence"] = (rows["tm_mapping_confidence"] >= 0.5).astype("int8")
    rows["tm_mapping_confidence_bucket"] = pd.cut(rows["tm_mapping_confidence"], [-0.01, 0.25, 0.5, 0.75, 1.01], labels=["none_low", "low", "medium", "high"]).astype("string").fillna("none_low").astype(str)
    rows["tm_mapping_selected"] = rows["mapping_selected"].astype("int8")
    for feature in TM_PHYSICAL_COLUMNS + TM_PITCH_COLUMNS:
        prior_feature = f"{feature}_prior"
        rows[feature] = pd.to_numeric(rows[feature], errors="coerce").fillna(pd.to_numeric(rows.get(prior_feature), errors="coerce")).fillna(0.0)
        if strategy == "mapping_confidence_weighted":
            prior_value = pd.to_numeric(rows.get(prior_feature), errors="coerce").fillna(0.0)
            confidence = rows["tm_mapping_confidence"].clip(0.0, 1.0)
            rows[feature] = confidence * rows[feature] + (1.0 - confidence) * prior_value
    return rows[TM_FEATURE_COLUMNS].copy()


def apply_low_frequency(frame, values):
    out = frame.copy()
    for col, allowed in (values or {}).items():
        allowed_set = set(allowed)
        current = safe_string(out[col])
        out[col] = current.where(current.isin(allowed_set), "__LOW_FREQ__")
    return out


def encode_for_lightgbm(frame, cat_cols, maps):
    out = frame.copy()
    for col in cat_cols:
        out[col] = safe_string(out[col]).map(maps.get(col, {})).fillna(-1).astype("int32")
    for col in out.columns:
        if col not in cat_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def component_frame(features, component, schema):
    features = features.reset_index(drop=True)
    variant = component["variant"]
    tm_features = None
    if component.get("trackman_state") is not None:
        tm_features = build_trackman_features(features, component["trackman_state"], variant)
        tm_features = tm_features.reset_index(drop=True)
        features = pd.concat([features, tm_features], axis=1)
    features = add_current_season_features(features, component["feature_state"], tm_features)
    x = features.loc[:, schema["feature_columns"]].copy()
    if variant in {"no_ids", "recent_weighted"}:
        x = x.drop(columns=[c for c in ["pitcher_id", "batter_id"] if c in x.columns])
    elif variant == "lowfreq_ids":
        x = apply_low_frequency(x, component.get("lowfreq_values", {}))
    x = x.loc[:, component["feature_columns"]]
    return x


def apply_calibration(calibrator, predictions):
    p = np.clip(np.asarray(predictions, dtype=float), 1e-6, 1.0 - 1e-6)
    if calibrator.get("kind") == "none":
        return p
    if calibrator["kind"] == "platt":
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        return np.clip(calibrator["model"].predict_proba(logits)[:, 1], 0.0, 1.0)
    if calibrator["kind"] == "isotonic":
        return np.clip(calibrator["model"].predict(p), 0.0, 1.0)
    if calibrator["kind"] == "prior_correction":
        target_prior = float(calibrator["target_prior"])
        prediction_prior = float(calibrator["prediction_prior"])
        odds = p / (1.0 - p)
        correction = (target_prior / (1.0 - target_prior)) / (prediction_prior / (1.0 - prediction_prior))
        corrected = odds * correction
        return np.clip(corrected / (1.0 + corrected), 0.0, 1.0)
    raise ValueError(f"unsupported calibration: {calibrator.get('kind')}")


def resolve_paths():
    checked = []
    for directory in ("./open", "./data"):
        test_path = os.path.join(directory, "test.csv")
        sample_path = os.path.join(directory, "sample_submission.csv")
        checked.extend([test_path, sample_path])
        if os.path.isfile(test_path) and os.path.isfile(sample_path):
            return test_path, sample_path
    raise FileNotFoundError(f"input files not found: {checked}")


def main():
    test_path, sample_path = resolve_paths()
    started = time.perf_counter()
    bundle = joblib.load("./model/final_model.joblib")
    calibrator = joblib.load("./model/calibration_model.joblib")
    with open("./model/feature_schema.json", encoding="utf-8") as stream:
        schema = json.load(stream)
    load_seconds = time.perf_counter() - started
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    sample = pd.read_csv(sample_path, encoding="utf-8-sig")
    if list(sample.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(f"sample columns must be {[ID_COL, TARGET_COL]}")
    actual = [c for c in test.columns if c != ID_COL]
    if actual != bundle["input_columns"] or actual != schema["input_columns"]:
        raise ValueError("test input schema mismatch")
    if test[ID_COL].duplicated().any() or sample[ID_COL].duplicated().any():
        raise ValueError("duplicate row_id")
    started = time.perf_counter()
    predictions = np.zeros(len(test), dtype=float)
    for component in bundle["components"]:
        features = build_features(test, component["feature_state"])
        x = component_frame(features, component, schema)
        if component["name"] == "catboost":
            for col in component["categorical_columns"]:
                x[col] = safe_string(x[col])
        elif component["name"] == "lightgbm":
            x = encode_for_lightgbm(x, component["categorical_columns"], component.get("encoder") or {})
        part = component["model"].predict_proba(x)[:, 1]
        predictions += float(component["weight"]) * np.asarray(part, dtype=float)
    predictions = apply_calibration(calibrator, predictions)
    inference_seconds = time.perf_counter() - started
    if not np.isfinite(predictions).all() or ((predictions < 0) | (predictions > 1)).any():
        raise ValueError("prediction is not finite or outside [0, 1]")
    if set(test[ID_COL]) != set(sample[ID_COL]) or len(test) != len(sample):
        raise ValueError("test/sample row_id mismatch")
    pred_map = pd.Series(predictions, index=test[ID_COL])
    result = sample[[ID_COL]].copy()
    result[TARGET_COL] = pred_map.loc[result[ID_COL]].to_numpy()
    os.makedirs("./output", exist_ok=True)
    result.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"Saved: ./output/submission.csv rows={len(result)} load={load_seconds:.4f}s inference={inference_seconds:.4f}s")


if __name__ == "__main__":
    main()
