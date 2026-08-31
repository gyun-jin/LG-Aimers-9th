"""TabM(+변형)+GBDT 앙상블 worst-fold 판정. 각 VAL(2022/2023/2024): host(bag, cat3) 단독 vs host+NN w격자.
NN 그룹: tabmb(기존), tabmt(tm2 입력 추가). Platt는 val 고정난수 반분 교차적합. d = 혼합 − host, bootstrap 95% CI."""
import os, glob, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
DATA = "/root/jegu/data"; R = "/root/jegu/work/research"
def brier(y, p): return float(np.mean((p-y)**2))
def z(q): q = np.clip(q, 1e-6, 1-1e-6); return np.log(q/(1-q))
for VAL in (2022, 2023, 2024):
    P = f"{R}/preds_b2024" if VAL == 2024 else f"{R}/preds_st{VAL}"
    hosts = {}
    for h in ("bag", "cat3"):
        f = f"{P}/{h}.npz"
        if os.path.exists(f): hosts[h] = np.load(f)
    nn = {}
    for pref in ("tabmb", "tabmt"):
        fs = sorted(glob.glob(f"{P}/{pref}_s*.npz"))
        if fs: nn[pref] = (np.mean([np.load(f)["fv"] for f in fs], 0), len(fs))
    if not nn or not hosts:
        print(f"[{VAL}] 파일 부족: hosts={list(hosts)} nn={list(nn)}"); continue
    ref = hosts.get("bag", list(hosts.values())[0])
    yv = ref["yv"] if "yv" in ref.files else np.load(f"{P}/base.npz")["yv"]
    half = np.random.default_rng(123).permutation(len(yv)) % 2 == 0
    def xcal(pv):
        out = np.empty_like(pv); sl = []
        for fit, ev in ((half, ~half), (~half, half)):
            m = LogisticRegression(C=1e6, max_iter=500).fit(z(pv[fit]).reshape(-1, 1), yv[fit])
            out[ev] = m.predict_proba(z(pv[ev]).reshape(-1, 1))[:, 1]; sl.append(m.coef_[0][0])
        return np.clip(out, 0, 1), float(np.mean(sl))
    def boot(pb, pn, n=300):
        rng = np.random.default_rng(0); N = len(yv); d = []
        for _ in range(n):
            i = rng.integers(0, N, N); d.append(brier(yv[i], pn[i]) - brier(yv[i], pb[i]))
        d = np.array(d); return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)
    print(f"\n[VAL={VAL}] n={len(yv)}  NN 그룹: " + ", ".join(f"{k}(x{v[1]})" for k, v in nn.items()))
    for name, (tv, k) in nn.items():
        tc, tsl = xcal(tv); print(f"  {name} 단독               Brier={brier(yv,tc):.6f} slope={tsl:.2f}")
    for hname, hd in hosts.items():
        hv = hd["fv"]
        hc, hsl = xcal(hv)
        print(f"  {hname:8s} 단독            Brier={brier(yv,hc):.6f} slope={hsl:.2f}")
        for name, (tv, k) in nn.items():
            for w in (0.05, 0.1, 0.2, 0.3):
                bc, bsl = xcal((1-w)*hv + w*tv)
                m, lo, hi = boot(hc, bc)
                tag = "개선*" if hi < 0 else ("악화*" if lo > 0 else "-")
                print(f"    {hname}+{name} w={w:<4} Brier={brier(yv,bc):.6f} d={m:+.6f} CI[{lo:+.6f},{hi:+.6f}] {tag}")
