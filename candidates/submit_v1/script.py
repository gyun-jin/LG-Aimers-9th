"""평가 서버용 독립 실행 추론 코드."""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"


# =============================================================================
# [자료 제공 코드 시작]
# 출처: baseline_submit/script.py
# =============================================================================
def load_test(path):
    """평가 데이터(csv) 로드. 한 행이 투구 하나."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path):
    """sample_submission.csv 로드 — 제출 파일의 row_id 순서/컬럼 기준."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    # [자료 제공 코드 기반 수정]
    # 수정 이유: 첫 두 컬럼만 보지 않고 출력 컬럼이 정확히 두 개인지 강제
    if list(df.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: {list(df.columns)}"
        )
    return df
# =============================================================================
# [자료 제공 코드 끝]
# =============================================================================


# [추가 구현]
# 목적: fold-fit prior와 한 행의 투구 직전 정보만 사용한 학습 동일 피처
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
REDUNDANT_COLUMNS = [
    "asof_pitcher_pitchmix_n",
    "run_total_before",
    "num_runners_on",
    "away_win_expectancy",
    "asof_pitcher_offspeed_rate",
]


def _safe_string(series):
    return series.astype("string").fillna("__MISSING__").astype(str)


def build_features(frame, state):
    """test 내부 통계 없이 한 행과 저장된 학습 prior만 사용한다."""
    input_columns = state["input_columns"]
    missing = [c for c in input_columns if c not in frame.columns]
    extra = [c for c in frame.columns if c not in input_columns + [ID_COL, TARGET_COL]]
    if missing or extra:
        raise ValueError(f"입력 schema 불일치: missing={missing}, extra={extra}")
    x = frame.loc[:, input_columns].copy()
    x["hand_matchup"] = _safe_string(x["pitcher_hand"]) + "_" + _safe_string(x["batter_hand"])
    x["count_state"] = _safe_string(x["balls_before"]) + "-" + _safe_string(x["strikes_before"])
    x["runner_out_state"] = _safe_string(x["base_state"]) + "_o" + _safe_string(x["outs_before"])
    x["close_game"] = (x["score_diff_pitcher_team"].abs() <= 2).astype("int8")
    x["late_inning"] = (x["inning"] >= 7).astype("int8")
    high_li = (x["li"] >= 1.5).astype("int8")
    x["pressure_state"] = (
        "late" + x["late_inning"].astype(str)
        + "_close" + x["close_game"].astype(str)
        + "_li" + high_li.astype(str)
    )
    x["log1p_asof_pitcher_n"] = np.log1p(x["asof_pitcher_n"].clip(lower=0))
    x["log1p_asof_batter_n"] = np.log1p(x["asof_batter_n"].clip(lower=0))
    for rate_col, count_col in RATE_COUNT_MAP.items():
        count = pd.to_numeric(x[count_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        rate = pd.to_numeric(x[rate_col], errors="coerce")
        prior = state["rate_priors"][rate_col]
        observed = rate.fillna(prior)
        x[f"{rate_col}_smoothed"] = (
            count * observed + state["alpha"] * prior
        ) / (count + state["alpha"])
    recent_success_cols = [
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    ]
    recent_middle_cols = [
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
    ]
    success = x[recent_success_cols]
    middle = x[recent_middle_cols]
    x["recent_success_mean_1_3_5"] = success.mean(axis=1, skipna=True)
    x["recent_success_range_1_3_5"] = success.max(axis=1, skipna=True) - success.min(axis=1, skipna=True)
    x["recent_success_std_1_3_5"] = success.std(axis=1, skipna=True, ddof=0)
    for window, col in zip((1, 3, 5), recent_success_cols):
        x[f"recent_gap_{window}"] = x[col] - x["asof_pitcher_success_rate_smoothed"]
    x["recent_middle_mean_1_3_5"] = middle.mean(axis=1, skipna=True)
    x["recent_middle_range_1_3_5"] = middle.max(axis=1, skipna=True) - middle.min(axis=1, skipna=True)
    x["recent_middle_vs_cumulative"] = (
        x["recent_middle_mean_1_3_5"] - x["asof_pitcher_middle_rate_smoothed"]
    )
    for col in state["missing_columns"]:
        x[f"{col}__missing"] = x[col].isna().astype("int8")
    x["is_pitcher_cold_start"] = (x["asof_pitcher_n"].fillna(0) <= 0).astype("int8")
    x["is_batter_cold_start"] = (x["asof_batter_n"].fillna(0) <= 0).astype("int8")
    mix_cols = [
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ]
    mix = x[mix_cols].clip(lower=0.0, upper=1.0)
    x["pitchmix_entropy"] = -(mix * np.log(mix.clip(lower=1e-12))).sum(axis=1, min_count=1)
    x["score_abs"] = x["score_diff_pitcher_team"].abs()
    x["is_pitcher_team_leading"] = (x["score_diff_pitcher_team"] > 0).astype("int8")
    x["is_pitcher_team_trailing"] = (x["score_diff_pitcher_team"] < 0).astype("int8")
    if state["drop_redundant"]:
        x = x.drop(columns=[c for c in REDUNDANT_COLUMNS if c in x.columns])
    return x


def apply_encoder(frame, encoder):
    result = frame.copy()
    kind = encoder["kind"]
    for col in encoder["categorical_columns"]:
        values = result[col].astype("string").fillna("__MISSING__").astype(str)
        default = -1 if kind == "category_code" else 0.0
        result[col] = values.map(encoder["mappings"][col]).fillna(default)
        result[col] = result[col].astype("int32" if kind == "category_code" else "float32")
    return result


def predict_bundle(bundle, test):
    expected = bundle["input_columns"]
    actual = [c for c in test.columns if c != ID_COL]
    if actual != expected:
        raise ValueError(f"입력 컬럼/순서 불일치: expected={expected}, actual={actual}")
    predictions = np.zeros(len(test), dtype=float)
    for component in bundle["components"]:
        name = component["name"]
        if name == "random_forest":
            x = test.loc[:, component["feature_columns"]].copy()
        else:
            x = build_features(test, component["feature_state"])
            if list(x.columns) != component["feature_columns"]:
                raise ValueError(f"{name} 파생 피처 schema 불일치")
            if name == "catboost":
                for col in component["categorical_columns"]:
                    x[col] = _safe_string(x[col])
            else:
                x = apply_encoder(x, component["encoder"])
        part = component["model"].predict_proba(x)[:, 1]
        predictions += float(component["weight"]) * np.asarray(part, dtype=float)
    return predictions


def apply_calibrator(calibrator, preds):
    p = np.asarray(preds, dtype=float)
    kind = calibrator["kind"]
    clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    if kind == "none":
        result = p
    elif kind == "platt":
        result = calibrator["model"].predict_proba(logits.reshape(-1, 1))[:, 1]
    elif kind == "temperature":
        z = logits / float(calibrator["temperature"])
        result = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
    elif kind == "isotonic":
        result = calibrator["model"].predict(p)
    else:
        raise ValueError(f"지원하지 않는 calibrator: {kind}")
    return np.clip(np.asarray(result, dtype=float), 0.0, 1.0)


# [자료 제공 코드 기반 수정]
# 수정 이유: 누락 ID placeholder를 금지하고 ID·행 수·유한성·범위·순서를 모두 강제
def merge_predictions(sub, ids, preds):
    if list(sub.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(f"출력 컬럼 오류: {list(sub.columns)}")
    ids = pd.Series(ids)
    if ids.duplicated().any() or sub[ID_COL].duplicated().any():
        raise ValueError("중복 row_id")
    if set(ids) != set(sub[ID_COL]):
        raise ValueError(
            f"ID 집합 불일치: missing={set(sub[ID_COL]) - set(ids)}, extra={set(ids) - set(sub[ID_COL])}"
        )
    p = np.asarray(preds, dtype=float)
    if len(p) != len(ids):
        raise ValueError(f"예측 행 수 불일치: ids={len(ids)}, preds={len(p)}")
    if not np.isfinite(p).all() or ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("NaN/무한대 또는 [0,1] 밖 예측")
    pred_map = pd.Series(p, index=ids)
    result = sub[[ID_COL]].copy()
    result[TARGET_COL] = pred_map.loc[result[ID_COL]].to_numpy()
    if result[ID_COL].tolist() != sub[ID_COL].tolist():
        raise ValueError("sample_submission 순서 불일치")
    return result


# =============================================================================
# [자료 제공 코드 시작]
# 출처: baseline_submit/script.py
# =============================================================================
def save_submission(path, sub):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")
# =============================================================================
# [자료 제공 코드 끝]
# =============================================================================


def resolve_input_paths():
    """평가 서버의 open/을 우선하고 배포 샘플의 data/도 호환한다."""
    # [자료 제공 코드 기반 수정]
    # 수정 이유: 최신 제출 규칙의 ./open/과 기존 배포 명세의 ./data/를 모두 안전하게 지원
    checked = []
    for directory in ("./open", "./data"):
        test_path = os.path.join(directory, "test.csv")
        sample_path = os.path.join(directory, "sample_submission.csv")
        checked.extend([test_path, sample_path])
        if os.path.isfile(test_path) and os.path.isfile(sample_path):
            return test_path, sample_path
    raise FileNotFoundError(f"평가 입력 파일을 찾을 수 없음: checked={checked}")


def main():
    # 최신 평가 서버 open/ 경로를 우선하고 기존 data/ 배포본도 지원한다.
    test_path, sample_path = resolve_input_paths()
    model_path = "./model/final_model.joblib"
    calibration_path = "./model/calibration_model.joblib"
    schema_path = "./model/feature_schema.json"
    output_path = "./output/submission.csv"

    started = time.perf_counter()
    bundle = joblib.load(model_path)
    calibrator = joblib.load(calibration_path)
    with open(schema_path, encoding="utf-8") as stream:
        schema = json.load(stream)
    load_seconds = time.perf_counter() - started

    test = load_test(test_path)
    sub = load_sample_submission(sample_path)
    if test[ID_COL].duplicated().any():
        raise ValueError("test row_id 중복")
    actual_columns = [c for c in test.columns if c != ID_COL]
    if actual_columns != schema["input_columns"] or actual_columns != bundle["input_columns"]:
        raise ValueError("저장 schema와 test 입력 컬럼/순서 불일치")

    started = time.perf_counter()
    raw_preds = predict_bundle(bundle, test)
    preds = apply_calibrator(calibrator, raw_preds)
    inference_seconds = time.perf_counter() - started

    result = merge_predictions(sub, test[ID_COL].tolist(), preds)
    if list(result.columns) != [ID_COL, TARGET_COL] or len(result) != len(test):
        raise ValueError("최종 submission 구조 오류")
    save_submission(output_path, result)
    print(
        f"Saved: {output_path} rows={len(result)} load={load_seconds:.4f}s "
        f"inference={inference_seconds:.4f}s"
    )


if __name__ == "__main__":
    main()
