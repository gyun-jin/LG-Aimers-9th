"""CatBoost 단독 3-seed bag (v5-3 138피처, LGBM 없음). env: VAL, GPU_ID, RUN(holdout|final).
holdout -> VAL==2024: preds_b2024/cat3.npz, 그 외: preds_st{VAL}/cat3.npz. final -> tabmcat_submit/model/gbdt_cat3.joblib"""
import os, json, time, warnings; warnings.filterwarnings("ignore")
import importlib.util
import numpy as np, pandas as pd, joblib
V53 = "/root/jegu/work/submit_v5_3"; DATA = "/root/jegu/data"
VAL = int(os.environ.get("VAL", "2024")); RUN = os.environ.get("RUN", "holdout")
GPU = os.environ.get("GPU_ID", "0"); os.environ["CUDA_VISIBLE_DEVICES"] = GPU
spec = importlib.util.spec_from_file_location("s53", f"{V53}/script.py"); s53 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s53)
T0 = time.time()
def log(m): print(f"{m}  ({time.time()-T0:.0f}s)", flush=True)
bundle = joblib.load(f"{V53}/model/final_model.joblib"); schema = json.load(open(f"{V53}/model/feature_schema.json", encoding="utf-8"))
comp = bundle["components"][0]; CATS = comp["categorical_columns"]
full = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig")
y_all = full["control_success"].to_numpy(int); seasons = full["season"].to_numpy()
X = s53.component_frame(s53.build_features(full[bundle["input_columns"]].reset_index(drop=True), comp["feature_state"]), comp, schema)
itr = np.flatnonzero(seasons < VAL) if RUN == "holdout" else np.arange(len(X))
log(f"VAL={VAL} RUN={RUN} X {X.shape} train={len(itr)}")
from catboost import CatBoostClassifier, Pool
xt, yt = X.iloc[itr], y_all[itr]; xtc = xt.copy()
for c in CATS: xtc[c] = s53.safe_string(xtc[c])
cats = []
for sd in (42, 43, 44):
    cbm = CatBoostClassifier(loss_function="Logloss", iterations=520, learning_rate=0.03, depth=8, l2_leaf_reg=20.0, random_seed=sd,
                             allow_writing_files=False, verbose=False, task_type="GPU", devices="0")
    cbm.fit(Pool(xtc, yt, cat_features=CATS)); log(f"cat {sd}")
    cats.append(cbm)
if RUN == "holdout":
    iva = np.flatnonzero(seasons == VAL); xv = X.iloc[iva]; xvc = xv.copy()
    for c in CATS: xvc[c] = s53.safe_string(xvc[c])
    p = np.zeros(len(iva))
    for cbm in cats: p += cbm.predict_proba(xvc)[:, 1]
    p /= 3
    OUTD = "/root/jegu/work/research/preds_b2024" if VAL == 2024 else f"/root/jegu/work/research/preds_st{VAL}"
    os.makedirs(OUTD, exist_ok=True)
    np.savez_compressed(f"{OUTD}/cat3.npz", fv=p, yv=y_all[iva])
    log(f"saved {OUTD}/cat3.npz")
else:
    OUT = "/root/jegu/work/tabmcat_submit/model"; os.makedirs(OUT, exist_ok=True)
    joblib.dump({"cats": cats, "feat_cols": list(X.columns), "cat_cols": CATS}, f"{OUT}/gbdt_cat3.joblib", compress=3)
    log("saved gbdt_cat3.joblib")
