"""TabM-mini fold 러너: train<VAL -> val VAL. env: VAL(2022|2023|2024|2025), SEED, GPU_ID, SAVE_STATE(1=state+prep 저장).
VAL=2024 -> preds_b2024/tabmb_s{SEED}.npz, 그 외 -> preds_st{VAL}/tabmb_s{SEED}.npz. VAL=2025는 전체 학습(최종모델용, val 없음).
SAVE_STATE=1 -> /root/jegu/work/tabm_final/tabm_s{SEED}.pt + prep.json(SEED=0일 때)"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
import importlib.util
import numpy as np, pandas as pd, joblib
V53 = "/root/jegu/work/submit_v5_3"; DATA = "/root/jegu/data"
VAL = int(os.environ.get("VAL", "2024"))
SEED = int(os.environ.get("SEED", "0")); GPU = os.environ.get("GPU_ID", "0"); os.environ["CUDA_VISIBLE_DEVICES"] = GPU
SAVE_STATE = os.environ.get("SAVE_STATE", "0") == "1"
OUTD = "/root/jegu/work/research/preds_b2024" if VAL == 2024 else f"/root/jegu/work/research/preds_st{VAL}"
os.makedirs(OUTD, exist_ok=True)
FIN = "/root/jegu/work/tabm_final"
K = 32; HID = 512; BLOCKS = 3; DROP = 0.1; EMB = 16
spec = importlib.util.spec_from_file_location("s53", f"{V53}/script.py"); s53 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s53)
import torch, torch.nn as nn
DEV = "cuda:0"; torch.manual_seed(SEED); np.random.seed(SEED)
T0 = time.time()
def log(m): print(f"{m}  ({time.time()-T0:.0f}s)", flush=True)

bundle = joblib.load(f"{V53}/model/final_model.joblib"); schema = json.load(open(f"{V53}/model/feature_schema.json", encoding="utf-8"))
comp = bundle["components"][0]; CATS = comp["categorical_columns"]
full = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig")
y_all = full["control_success"].to_numpy(int); seasons = full["season"].to_numpy(); row_ids = full["row_id"].to_numpy()
X = s53.component_frame(s53.build_features(full[bundle["input_columns"]].reset_index(drop=True), comp["feature_state"]), comp, schema)
NUMS = [c for c in X.columns if c not in CATS]
itr = np.flatnonzero(seasons < VAL); iva = np.flatnonzero(seasons == VAL)
mt = pd.read_pickle(f"{DATA}/train_matched.pkl").set_index("row_id")
REG = ["rel_speed","spin_rate","induced_vert_break","horz_break","extension","rel_height","rel_side","zone_speed"]
aux_reg = mt[REG].reindex(row_ids).to_numpy("float32")
log(f"X {X.shape} tabm VAL={VAL} seed={SEED} train={len(itr)} val={len(iva)}")

xt = X.iloc[itr]
vocab = {c: {str(v): i+1 for i, v in enumerate(s53.safe_string(xt[c]).unique())} for c in CATS}; card = [len(vocab[c])+1 for c in CATS]
med = xt[NUMS].apply(pd.to_numeric, errors="coerce").median()
def nf(idx): return X.iloc[idx][NUMS].apply(pd.to_numeric, errors="coerce").fillna(med)
trn = nf(itr); mu = trn.mean(); sd = trn.std().replace(0,1)+1e-9
def pn(idx): return ((nf(idx)-mu)/sd).to_numpy("float32")
def pc(idx):
    xv = X.iloc[idx]; return np.stack([s53.safe_string(xv[c]).map(vocab[c]).fillna(0).to_numpy("int64") for c in CATS],1)
ntr, ctr, ytr = pn(itr), pc(itr), y_all[itr].astype("float32")
a = aux_reg[itr]; am = np.isfinite(a[:,0]); a_mu = np.nanmean(a,0); a_sd = np.nanstd(a,0)+1e-6
a_std = np.where(np.isfinite(a), (a-a_mu)/a_sd, 0.0).astype("float32")

class TabM(nn.Module):
    def __init__(self, n_num, card):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, EMB) for c in card])
        d = n_num + EMB*len(card)
        self.R = nn.Parameter(torch.sign(torch.randn(K, d)))
        self.lin0 = nn.Linear(d, HID)
        self.blocks = nn.ModuleList([nn.Linear(HID, HID) for _ in range(BLOCKS-1)])
        self.drop = nn.Dropout(DROP)
        self.head_w = nn.Parameter(torch.randn(K, HID)*0.02); self.head_b = nn.Parameter(torch.zeros(K))
        self.aux_w = nn.Parameter(torch.randn(K, HID, 8)*0.02); self.aux_b = nn.Parameter(torch.zeros(K, 8))
    def forward(self, xn, xc):
        x = torch.cat([xn] + [e(xc[:,i]) for i,e in enumerate(self.embs)], 1)
        h = x.unsqueeze(1) * self.R
        h = self.drop(torch.relu(self.lin0(h)))
        for blk in self.blocks: h = self.drop(torch.relu(blk(h)))
        logits = (h*self.head_w).sum(-1) + self.head_b
        aux = torch.einsum("bkh,khr->bkr", h, self.aux_w) + self.aux_b
        return logits, aux

model = TabM(ntr.shape[1], card).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4); bce = nn.BCEWithLogitsLoss()
n = len(ytr); n_es = n//20; p0 = np.random.default_rng(SEED).permutation(n); esi, tri = p0[:n_es], p0[n_es:]
TN, TC, TY = (torch.tensor(v).to(DEV) for v in (ntr[tri], ctr[tri], ytr[tri]))
TA, TM = torch.tensor(a_std[tri]).to(DEV), torch.tensor(am[tri].astype("float32")).to(DEV)
EN, EC, EY = torch.tensor(ntr[esi]).to(DEV), torch.tensor(ctr[esi]).to(DEV), ytr[esi]
def pred(xn, xc, bs=8192):
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(xn),bs): out.append(torch.sigmoid(model(xn[i:i+bs], xc[i:i+bs])[0]).mean(1).cpu())
    return torch.cat(out).numpy()
BS=4096; best=1e9; bstate=None; pat=0
for ep in range(12):
    model.train(); pm = torch.randperm(len(TY), device=DEV)
    for i in range(0,len(pm),BS):
        idx = pm[i:i+BS]; opt.zero_grad()
        logits, aux = model(TN[idx], TC[idx])
        loss = bce(logits, TY[idx].unsqueeze(1).expand(-1, K))
        m = TM[idx].view(-1,1,1)
        loss = loss + 0.3*((aux - TA[idx].unsqueeze(1))**2 * m).sum()/(m.sum()*K*8+1e-6)
        loss.backward(); opt.step()
    b = float(np.mean((pred(EN,EC)-EY)**2)); log(f"  ep{ep+1} es={b:.5f}")
    if b < best-1e-5: best, bstate, pat = b, {k:v.clone() for k,v in model.state_dict().items()}, 0
    else:
        pat += 1
        if pat >= 2: break
if bstate: model.load_state_dict(bstate)
if len(iva):
    fv = pred(torch.tensor(pn(iva)).to(DEV), torch.tensor(pc(iva)).to(DEV))
    np.savez_compressed(f"{OUTD}/tabmb_s{SEED}.npz", fc=fv, fv=fv)
    log(f"saved {OUTD}/tabmb_s{SEED}.npz")
if SAVE_STATE:
    os.makedirs(FIN, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, f"{FIN}/tabm_s{SEED}.pt")
    if SEED == 0:
        prep = {"NUMS": NUMS, "CATS": list(CATS), "K": K, "HID": HID, "BLOCKS": BLOCKS, "EMB": EMB,
                "vocab": vocab, "med": {c: float(med[c]) for c in NUMS},
                "mu": {c: float(mu[c]) for c in NUMS}, "sd": {c: float(sd[c]) for c in NUMS},
                "card": card}
        json.dump(prep, open(f"{FIN}/prep.json", "w"))
    log(f"saved state {FIN}/tabm_s{SEED}.pt")
