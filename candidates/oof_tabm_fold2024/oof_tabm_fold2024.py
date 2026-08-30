"""
[팀원 실행] TabM OOF Fold 2024
- 2019-2023으로 학습 → 2024 예측 (진짜 OOF, 누수 없음)
- 결과: oof_results/oof_tabm_2024.npz
- 소요: RTX 계열 GPU 기준 약 20-30분

팀원 실행 전 아래 경로 수정:
  DATA_CSV  = 본인 train.csv 경로
  TM_LOOKUP = model_v9/tm_lookup.pkl 경로
  V9_SCHEMA = model_v9/feature_schema.json 경로
"""
import os, json, time, warnings, gc
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import QuantileTransformer, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss
import joblib

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── [팀원] 아래 경로 본인 기기에 맞게 수정 ───────────────────────────────────
DATA_CSV  = r"경로\train.csv"
TM_LOOKUP = r"경로\model_v9\tm_lookup.pkl"
V9_SCHEMA = r"경로\model_v9\feature_schema.json"
OUT_DIR   = Path(r"경로\oof_results")
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(exist_ok=True)
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED          = 42
ALPHA         = 50.0
MATCHUP_ALPHA = 50.0
BATCH_TR      = 4096
N_EPOCHS      = 10
USE_AMP       = (DEVICE.type == "cuda")
VALID_SEASON  = 2024

torch.manual_seed(SEED); np.random.seed(SEED)
if DEVICE.type == "cuda": torch.cuda.manual_seed_all(SEED)
print(f"Device: {DEVICE}" + (f"  GPU: {torch.cuda.get_device_name(0)}" if DEVICE.type=='cuda' else ""), flush=True)

TABM9_NUM = [
    "run_top_before","run_bot_before","run_total_before","score_diff_home",
    "score_diff_pitcher_team","num_runners_on","home_win_expectancy","li",
    "asof_pitcher_n","asof_pitcher_success_rate","asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate","asof_pitcher_ball_rate","asof_pitcher_strike_rate",
    "asof_batter_n","asof_batter_success_rate","asof_batter_middle_rate",
    "asof_pitcher_fastball_rate","asof_pitcher_breaking_rate","asof_pitcher_offspeed_rate",
    "log1p_asof_pitcher_n","log1p_asof_batter_n","log1p_li",
    "pitcher_current_season_success_rate","pitcher_current_season_success_rate_smoothed",
    "pitcher_current_season_n","pitcher_current_season_success_count",
    "log1p_pitcher_current_season_n","pitcher_current_season_n_ratio_to_career",
    "tm_prev1_n","tm_prev1_rel_speed_mean","tm_prev1_spin_rate_mean",
    "tm_prev1_extension_mean","tm_prev1_rel_height_std","tm_prev1_rel_side_std",
    "tm_prev1_fastball_n","tm_prev1_fastball_rel_speed_mean",
    "tm_prev1_fastball_ivb_mean","tm_prev1_fastball_hb_mean",
    "tm_prev1_breaking_n","tm_prev1_breaking_ivb_mean","tm_prev1_breaking_hb_mean",
    "matchup_n","log1p_matchup_n","matchup_success_rate_smoothed",
]
TABM9_BIN = [
    "runner_on_1b","runner_on_2b","runner_on_3b",
    "pitcher_recent_history_missing","pitcher_history_missing","batter_history_missing",
    "pitcher_current_season_available_flag","pitcher_current_season_small_sample_flag",
    "tm_prev1_available_flag","matchup_available_flag","matchup_small_sample_flag",
]
TABM9_CAT = [
    "game_month","game_dayofweek","inning","top_bottom","game_type",
    "balls_before","strikes_before","outs_before","base_state",
    "pitcher_hand","batter_hand","pitcher_team_id","batter_team_id",
]
PREV1_COLS = [
    "tm_prev1_n","tm_prev1_rel_speed_mean","tm_prev1_spin_rate_mean",
    "tm_prev1_extension_mean","tm_prev1_rel_height_std","tm_prev1_rel_side_std",
    "tm_prev1_fastball_n","tm_prev1_fastball_rel_speed_mean",
    "tm_prev1_fastball_ivb_mean","tm_prev1_fastball_hb_mean",
    "tm_prev1_breaking_n","tm_prev1_breaking_ivb_mean","tm_prev1_breaking_hb_mean",
    "tm_prev1_available_flag",
]


def build_pitcher_cs_state(df, y, alpha):
    work = df[["pitcher_id","season","asof_pitcher_n","asof_pitcher_success_rate"]].copy()
    work["asof_pitcher_n"] = pd.to_numeric(work["asof_pitcher_n"], errors="coerce").fillna(0.0).clip(lower=0.0)
    work["asof_pitcher_success_rate"] = pd.to_numeric(work["asof_pitcher_success_rate"], errors="coerce")
    prior = float(np.mean(y))
    work["asof_pitcher_success_count"] = work["asof_pitcher_n"] * work["asof_pitcher_success_rate"].fillna(prior)
    season_end = (work.groupby(["pitcher_id","season"], observed=True)
                  .agg(season_end_pitcher_n=("asof_pitcher_n","max"),
                       season_end_pitcher_success_count=("asof_pitcher_success_count","max")).reset_index())
    seasons_raw = sorted(pd.to_numeric(df["season"], errors="coerce").dropna().astype(int).unique().tolist())
    seasons = list(range(min(seasons_raw), max(seasons_raw) + 2))
    rows = []
    for pid, grp in season_end.groupby("pitcher_id", sort=False):
        grp = grp.sort_values("season")
        s2n = dict(zip(grp["season"].astype(int), grp["season_end_pitcher_n"].astype(float)))
        s2s = dict(zip(grp["season"].astype(int), grp["season_end_pitcher_success_count"].astype(float)))
        ln, ls = 0.0, 0.0
        for s in seasons:
            rows.append({"pitcher_id": pid, "season": int(s),
                         "prior_season_end_pitcher_n": float(ln),
                         "prior_season_end_pitcher_success_count": float(ls)})
            if s in s2n: ln = max(ln, float(s2n[s])); ls = max(ls, float(s2s[s]))
    return {"alpha": float(alpha), "target_prior": float(prior), "prior_table": pd.DataFrame(rows)}


def build_matchup_state(raw_df, y_arr):
    tp = float(np.mean(y_arr))
    st = (raw_df.assign(_y=y_arr.astype(np.float32))
          .groupby(["pitcher_id","batter_id"], observed=True, sort=False)
          .agg(matchup_n_total=("_y","count"), matchup_success_total=("_y","sum")).reset_index())
    return {"alpha": float(MATCHUP_ALPHA), "target_prior": float(tp), "matchup_table": st}


def build_tabm69(raw_df, y_arr, tm_lookup, pcs_state, matchup_state, is_train):
    df = raw_df.reset_index(drop=True).copy()
    df["pitcher_recent_history_missing"] = df["asof_pitcher_prev1_game_success_rate"].isna().astype("int8")
    df["pitcher_history_missing"]        = df["asof_pitcher_success_rate"].isna().astype("int8")
    df["batter_history_missing"]         = df["asof_batter_success_rate"].isna().astype("int8")
    df["log1p_asof_pitcher_n"] = np.log1p(df["asof_pitcher_n"].fillna(0))
    df["log1p_asof_batter_n"]  = np.log1p(df["asof_batter_n"].fillna(0))
    df["log1p_li"]             = np.log1p(df["li"].fillna(0))
    # Pitcher CS
    table = pcs_state["prior_table"]; alpha_cs = float(pcs_state["alpha"]); tp_cs = float(pcs_state["target_prior"])
    raw_p = df[["pitcher_id","season","asof_pitcher_n","asof_pitcher_success_rate"]].merge(table, on=["pitcher_id","season"], how="left")
    pn = raw_p["prior_season_end_pitcher_n"].fillna(0.0).clip(lower=0.0).values
    ps = raw_p["prior_season_end_pitcher_success_count"].fillna(0.0).clip(lower=0.0).values
    cn = pd.to_numeric(raw_p["asof_pitcher_n"], errors="coerce").fillna(0.0).clip(lower=0.0).values
    cr = pd.to_numeric(raw_p["asof_pitcher_success_rate"], errors="coerce").fillna(tp_cs).values
    cur_n = np.clip(cn - pn, 0.0, None)
    cur_s = np.minimum(np.clip(cn * cr - ps, 0.0, None), cur_n)
    with np.errstate(divide="ignore", invalid="ignore"):
        cur_r = np.where(cur_n > 0, cur_s / cur_n, np.nan)
    cur_r = np.where(np.isfinite(cur_r), cur_r, tp_cs); cur_r = np.clip(cur_r, 0.0, 1.0)
    smoothed = np.clip((cur_s + alpha_cs * tp_cs) / (cur_n + alpha_cs), 0.0, 1.0)
    df["pitcher_current_season_n"]                     = cur_n
    df["log1p_pitcher_current_season_n"]               = np.log1p(cur_n)
    df["pitcher_current_season_success_count"]         = cur_s
    df["pitcher_current_season_success_rate"]          = cur_r
    df["pitcher_current_season_success_rate_smoothed"] = smoothed
    df["pitcher_current_season_n_ratio_to_career"]     = np.where(cn > 0, cur_n / cn, 0.0)
    df["pitcher_current_season_available_flag"]        = (cur_n > 0).astype("int8")
    df["pitcher_current_season_small_sample_flag"]     = (cur_n < 50).astype("int8")
    # Prev1 TrackMan
    lk = tm_lookup.copy(); lk["season"] = pd.to_numeric(lk["season"], errors="coerce").astype("Int64")
    mk = df[["pitcher_id","season"]].copy(); mk["season"] = pd.to_numeric(mk["season"], errors="coerce").astype("Int64")
    cols = [c for c in PREV1_COLS if c in lk.columns]
    merged = mk.merge(lk[["pitcher_id","season"] + cols], on=["pitcher_id","season"], how="left")
    for c in PREV1_COLS:
        df[c] = merged[c].fillna(0).values if c in merged.columns else 0.0
    # Matchup
    if is_train:
        tp = float(np.mean(y_arr))
        work = pd.DataFrame({"pitcher_id":raw_df["pitcher_id"].values,"batter_id":raw_df["batter_id"].values,
                              "asof_n":pd.to_numeric(raw_df["asof_pitcher_n"],errors="coerce").fillna(0.0).values,
                              "_y":y_arr.astype(np.float32),"_orig_idx":np.arange(len(raw_df),dtype=np.int32)})
        ws = work.sort_values(["pitcher_id","asof_n"]).reset_index(drop=True)
        grp = ws.groupby(["pitcher_id","batter_id"], sort=False)
        ws["matchup_n"] = grp.cumcount().astype(np.float32)
        ws["matchup_s"] = (grp["_y"].cumsum() - ws["_y"]).clip(lower=0.0)
        ws = ws.sort_values("_orig_idx").reset_index(drop=True)
        mn = ws["matchup_n"].values.astype(np.float32); ms = ws["matchup_s"].values.astype(np.float32)
        ms_sm = np.clip((ms + MATCHUP_ALPHA * tp) / (mn + MATCHUP_ALPHA), 0.0, 1.0)
    else:
        m = matchup_state["matchup_table"]; ma = float(matchup_state["alpha"]); mtp = float(matchup_state["target_prior"])
        merged_m = raw_df[["pitcher_id","batter_id"]].merge(m, on=["pitcher_id","batter_id"], how="left")
        mn = merged_m["matchup_n_total"].fillna(0.0).values.astype(np.float32)
        ms = merged_m["matchup_success_total"].fillna(0.0).values.astype(np.float32)
        ms_sm = np.clip((ms + ma * mtp) / (mn + ma), 0.0, 1.0)
    df["matchup_n"]                     = mn
    df["log1p_matchup_n"]               = np.log1p(mn)
    df["matchup_success_rate_smoothed"] = ms_sm
    df["matchup_available_flag"]        = (mn > 0).astype("int8")
    df["matchup_small_sample_flag"]     = (mn < 20).astype("int8")
    for c in TABM9_CAT:
        df[c] = df[c].fillna("__MISSING__").astype(str)
    return df[TABM9_NUM + TABM9_BIN + TABM9_CAT].copy()


class BEL(nn.Module):
    def __init__(self, in_f, out_f, k):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f)); self.bias = nn.Parameter(torch.zeros(out_f))
        self.r = nn.Parameter(torch.empty(k, in_f)); self.s = nn.Parameter(torch.empty(k, out_f))
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        nn.init.normal_(self.r, 1.0, 0.5); nn.init.normal_(self.s, 1.0, 0.5)
    def forward(self, x): return (x * self.r) @ self.weight.t() * self.s + self.bias

class TabM(nn.Module):
    def __init__(self, n_features, d_hidden=512, n_layers=5, k=32, dropout=0.05):
        super().__init__()
        self.k = k; self.input_proj = nn.Linear(n_features, d_hidden)
        self.blocks = nn.ModuleList([BEL(d_hidden, d_hidden, k) for _ in range(n_layers)])
        self.norms  = nn.ModuleList([nn.LayerNorm(d_hidden) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout); self.head = BEL(d_hidden, 1, k)
    def forward(self, x):
        h = self.input_proj(x).unsqueeze(1).expand(-1, self.k, -1)
        for be, norm in zip(self.blocks, self.norms):
            res = h; h = self.drop(F.gelu(be(h))); h = norm(h + res)
        return self.head(h).squeeze(-1).mean(dim=1, keepdim=True)


# ── 실행 ─────────────────────────────────────────────────────────────────────
print("[1/4] Loading data...", flush=True)
train_raw = pd.read_csv(DATA_CSV, encoding="utf-8-sig")
seasons   = pd.to_numeric(train_raw["season"], errors="coerce").to_numpy()
y_all     = train_raw["control_success"].to_numpy(dtype=np.float32)
tr_idx    = np.flatnonzero(seasons < VALID_SEASON)
va_idx    = np.flatnonzero(seasons == VALID_SEASON)
train_fold = train_raw.iloc[tr_idx].reset_index(drop=True)
valid_fold = train_raw.iloc[va_idx].reset_index(drop=True)
y_tr = y_all[tr_idx]; y_va = y_all[va_idx]
print(f"  train={len(tr_idx):,}  valid={len(va_idx):,}", flush=True)
tm_lookup = pd.read_pickle(TM_LOOKUP)

print("[2/4] Building fold states...", flush=True)
pcs_state  = build_pitcher_cs_state(train_fold, y_tr, ALPHA)
matchup_st = build_matchup_state(train_fold, y_tr)

print("[3/4] Building TabM features...", flush=True)
t0 = time.time()
feat_tr = build_tabm69(train_fold, y_tr, tm_lookup, pcs_state, matchup_st, is_train=True)
feat_va = build_tabm69(valid_fold, None,  tm_lookup, pcs_state, matchup_st, is_train=False)
print(f"  done ({time.time()-t0:.1f}s)", flush=True)

num_cols_t = TABM9_NUM + TABM9_BIN
t9_imp = SimpleImputer(strategy="median")
_imp   = t9_imp.fit_transform(feat_tr[num_cols_t].replace([np.inf,-np.inf], np.nan).astype(np.float32))
t9_qt  = QuantileTransformer(output_distribution="normal", random_state=SEED, n_quantiles=1000)
Xn_tr  = t9_qt.fit_transform(_imp).astype(np.float32)
Xn_va  = t9_qt.transform(t9_imp.transform(feat_va[num_cols_t].replace([np.inf,-np.inf], np.nan).astype(np.float32))).astype(np.float32)
t9_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
Xc_tr  = t9_enc.fit_transform(feat_tr[TABM9_CAT].astype(str)).astype(np.float32)
Xc_va  = t9_enc.transform(feat_va[TABM9_CAT].astype(str)).astype(np.float32)
X_tr   = np.hstack([Xn_tr, Xc_tr]); X_va = np.hstack([Xn_va, Xc_va])
assert X_tr.shape[1] == 69, f"TabM input dim 오류: {X_tr.shape[1]}"
del feat_tr, feat_va; gc.collect()

print(f"[4/4] Training TabM ({N_EPOCHS} epochs)...", flush=True)
torch.manual_seed(SEED)
model = TabM(69).to(DEVICE)
ds    = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
ld    = DataLoader(ds, BATCH_TR, shuffle=True, pin_memory=False, num_workers=0)
opt   = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-4, steps_per_epoch=len(ld), epochs=N_EPOCHS, pct_start=0.1)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

t0 = time.time()
for ep in range(1, N_EPOCHS + 1):
    model.train(); ep_loss = 0.0
    for xb, yb in ld:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE); opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            loss = F.binary_cross_entropy_with_logits(model(xb).squeeze(-1), yb)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        ep_loss += loss.item()
    print(f"  ep{ep:02d}  loss={ep_loss/len(ld):.5f}  ({time.time()-t0:.0f}s)", flush=True)

model.eval(); preds = []
X_va_t = torch.tensor(X_va)
with torch.no_grad():
    for i in range(0, len(X_va), BATCH_TR):
        xb = X_va_t[i:i+BATCH_TR].to(DEVICE)
        preds.append(torch.sigmoid(model(xb).squeeze(-1)).cpu().numpy())
preds_tabm = np.concatenate(preds)

print(f"\nTabM OOF Fold {VALID_SEASON} Brier: {brier_score_loss(y_va, preds_tabm):.6f}", flush=True)
out_path = OUT_DIR / "oof_tabm_2024.npz"
np.savez_compressed(out_path, y_true=y_va, pred=preds_tabm)
print(f"Saved → {out_path}", flush=True)
