"""Final TabM v1: Current Core (50) + exact previous-season TrackMan (14)."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from features.tabm.tabm_features_v1 import (
    TABM_V1_NUMERIC, TABM_V1_BINARY, TABM_V1_CATEGORICAL,
    build_tabm_train_v1, build_tabm_test_v1,
)

TARGET_COL = "control_success"

CURRENT_CORE_NUMERIC = [c for c in TABM_V1_NUMERIC if not any(x in c for x in ("prev1","prev3","prev5"))]
CURRENT_CORE_BINARY = [c for c in TABM_V1_BINARY if not any(x in c for x in ("prev1","prev3","prev5"))]
CURRENT_CORE_CATEGORICAL = TABM_V1_CATEGORICAL.copy()

TM_PREV1_NUMERIC = [
    "tm_prev1_n","tm_prev1_rel_speed_mean","tm_prev1_spin_rate_mean",
    "tm_prev1_extension_mean","tm_prev1_rel_height_std","tm_prev1_rel_side_std",
    "tm_prev1_fastball_n","tm_prev1_fastball_rel_speed_mean",
    "tm_prev1_fastball_ivb_mean","tm_prev1_fastball_hb_mean",
    "tm_prev1_breaking_n","tm_prev1_breaking_ivb_mean","tm_prev1_breaking_hb_mean",
]
TM_PREV1_BINARY = ["tm_prev1_available_flag"]

TABM_FINAL_V1_NUMERIC = CURRENT_CORE_NUMERIC + TM_PREV1_NUMERIC
TABM_FINAL_V1_BINARY = CURRENT_CORE_BINARY + TM_PREV1_BINARY
TABM_FINAL_V1_CATEGORICAL = CURRENT_CORE_CATEGORICAL
TABM_FINAL_V1_FEATURES = TABM_FINAL_V1_NUMERIC + TABM_FINAL_V1_BINARY + TABM_FINAL_V1_CATEGORICAL

TM_REQUIRED = ["pitcher_id","season","pitch_type_group","rel_speed","spin_rate",
               "induced_vert_break","horz_break","extension","rel_height","rel_side"]

def build_trackman_prev1(trackman: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(TM_REQUIRED) - set(trackman.columns))
    if missing:
        raise ValueError(f"TrackMan missing columns: {missing}")
    tm = trackman.loc[trackman.pitcher_id.notna() & trackman.season.notna(), TM_REQUIRED].copy()
    tm["pitcher_id"] = tm["pitcher_id"].astype("int64")
    tm["season"] = tm["season"].astype("int16")

    overall = tm.groupby(["pitcher_id","season"], as_index=False).agg(
        tm_prev1_n=("rel_speed","count"),
        tm_prev1_rel_speed_mean=("rel_speed","mean"),
        tm_prev1_spin_rate_mean=("spin_rate","mean"),
        tm_prev1_extension_mean=("extension","mean"),
        tm_prev1_rel_height_std=("rel_height","std"),
        tm_prev1_rel_side_std=("rel_side","std"),
    )
    fb = tm.loc[tm.pitch_type_group.eq("fastball")].groupby(["pitcher_id","season"],as_index=False).agg(
        tm_prev1_fastball_n=("rel_speed","count"),
        tm_prev1_fastball_rel_speed_mean=("rel_speed","mean"),
        tm_prev1_fastball_ivb_mean=("induced_vert_break","mean"),
        tm_prev1_fastball_hb_mean=("horz_break","mean"),
    )
    br = tm.loc[tm.pitch_type_group.eq("breaking")].groupby(["pitcher_id","season"],as_index=False).agg(
        tm_prev1_breaking_n=("rel_speed","count"),
        tm_prev1_breaking_ivb_mean=("induced_vert_break","mean"),
        tm_prev1_breaking_hb_mean=("horz_break","mean"),
    )
    ps = overall.merge(fb,on=["pitcher_id","season"],how="left",validate="one_to_one").merge(
        br,on=["pitcher_id","season"],how="left",validate="one_to_one")
    ps["season"] = ps["season"] + 1
    ps["tm_prev1_available_flag"] = 1
    return ps[["pitcher_id","season"] + TM_PREV1_NUMERIC + TM_PREV1_BINARY]

def _finalize(base, raw, prev1, include_target):
    keys = raw[["row_id","pitcher_id","season"]]
    tm_rows = keys.merge(prev1,on=["pitcher_id","season"],how="left",validate="many_to_one")
    tm_rows["tm_prev1_available_flag"] = tm_rows["tm_prev1_available_flag"].fillna(0).astype("int8")
    out = base.merge(tm_rows[["row_id"]+TM_PREV1_NUMERIC+TM_PREV1_BINARY],
                     on="row_id",how="left",validate="one_to_one")
    cols = ["row_id"] + ([TARGET_COL] if include_target else []) + TABM_FINAL_V1_FEATURES
    out = out[cols].copy()
    if len(TABM_FINAL_V1_FEATURES) != 64 or len(set(TABM_FINAL_V1_FEATURES)) != 64:
        raise ValueError("Final schema must contain 64 unique features.")
    if out.row_id.isna().any() or out.row_id.duplicated().any():
        raise ValueError("Invalid row_id.")
    return out

def build_tabm_final_v1_train(train, trackman):
    return _finalize(build_tabm_train_v1(train), train, build_trackman_prev1(trackman), True)

def build_tabm_final_v1_test(train, test, trackman):
    return _finalize(build_tabm_test_v1(train,test), test, build_trackman_prev1(trackman), False)

def get_feature_spec():
    return {"version":"tabm_features_final_v1","numeric":TABM_FINAL_V1_NUMERIC,
            "binary":TABM_FINAL_V1_BINARY,"categorical":TABM_FINAL_V1_CATEGORICAL,
            "features":TABM_FINAL_V1_FEATURES,
            "trackman":{"selected":"exact_previous_season",
                        "definition":"TrackMan season S-1 for target season S"}}

def save_tabm_final_v1(train, test, trackman, output_dir="./features/tabm/generated"):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    tr = build_tabm_final_v1_train(train,trackman)
    te = build_tabm_final_v1_test(train,test,trackman)
    if [c for c in tr.columns if c not in ("row_id",TARGET_COL)] != [c for c in te.columns if c!="row_id"]:
        raise ValueError("Train/test feature order mismatch.")
    tp=output_dir/"tabm_features_final_v1_train.parquet"
    ep=output_dir/"tabm_features_final_v1_test.parquet"
    sp=output_dir/"tabm_features_final_v1_feature_spec.json"
    tr.to_parquet(tp,index=False); te.to_parquet(ep,index=False)
    sp.write_text(json.dumps(get_feature_spec(),ensure_ascii=False,indent=2),encoding="utf-8")
    return tp,ep,sp
