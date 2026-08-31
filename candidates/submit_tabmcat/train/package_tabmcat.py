"""7호(tabmcat) 패키징: tabmcat_submit/{script.py, requirements.txt(torch 포함), model/*, submit.zip}
구성: CatBoost 3-seed(0.95) + TabM 3-seed(0.05) -> Platt(2024 홀드아웃 fold 모델 기준)."""
import json, os, re, shutil, zipfile
import numpy as np
from sklearn.linear_model import LogisticRegression

W = "/root/jegu/work"; SUB = f"{W}/tabmcat_submit"; PB = f"{W}/research/preds_b2024"; FIN = f"{W}/tabm_final"
os.makedirs(f"{SUB}/model", exist_ok=True)

base = open(f"{W}/ftt_submit/script.py", encoding="utf-8").read()
marker = "# ============================================================================\n# [FT-T 확장]"
head = base[:base.index(marker)].rstrip() + "\n\n\n"
ext = open(f"{W}/research/ext_tabmcat_snippet.py", encoding="utf-8").read()
open(f"{SUB}/script.py", "w", encoding="utf-8").write(head + ext)
import ast; ast.parse(open(f"{SUB}/script.py", encoding="utf-8").read()); print("script.py OK")

req = open(f"{W}/ftt_submit/requirements.txt", encoding="utf-8").read()
assert re.search(r"^torch==", req, re.M), "ftt requirements에 torch 핀 없음"
open(f"{SUB}/requirements.txt", "w").write(req)
print("requirements:", [r for r in req.splitlines() if r.strip()])

shutil.copy(f"{W}/submit_v5_3/model/final_model.joblib", f"{SUB}/model/final_model.joblib")
shutil.copy(f"{W}/submit_v5_3/model/feature_schema.json", f"{SUB}/model/feature_schema.json")
assert os.path.exists(f"{SUB}/model/gbdt_cat3.joblib"), "gbdt_cat3.joblib 없음 (stage3_cat RUN=final 먼저)"
for s in (0, 1, 2):
    src = f"{FIN}/tabm_s{s}.pt"; assert os.path.exists(src), f"{src} 없음"
    shutil.copy(src, f"{SUB}/model/tabm_s{s}.pt")
assert os.path.exists(f"{FIN}/prep.json"), "prep.json 없음"
shutil.copy(f"{FIN}/prep.json", f"{SUB}/model/tabm_prep.json")

TABM_W = 0.05
cat = np.load(f"{PB}/cat3.npz")
tv = np.mean([np.load(f"{PB}/tabmb_s{s}.npz")["fv"] for s in (0, 1, 2)], 0)
blend = (1 - TABM_W) * cat["fv"] + TABM_W * tv
yv = cat["yv"]
z = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6))).reshape(-1, 1)
calm = LogisticRegression(C=1e6, max_iter=500).fit(z(blend), yv)
p_cal = np.clip(calm.predict_proba(z(blend))[:, 1], 0, 1)
print(f"2024 holdout blend raw Brier={np.mean((blend-yv)**2):.6f} -> platt={np.mean((p_cal-yv)**2):.6f}  coef={calm.coef_[0][0]:.5f} inter={calm.intercept_[0]:.5f}")
json.dump({"platt_coef": float(calm.coef_[0][0]), "platt_intercept": float(calm.intercept_[0]),
           "model": "v5-3 138feats, CatBoost seed(42,43,44) 0.95 + TabM-mini seed(0,1,2) 0.05",
           "calibration": "platt on 2024 holdout (train<=2023 fold models, same blend)"},
          open(f"{SUB}/model/platt_tabmcat.json", "w"), indent=1)

zp = f"{SUB}/submit.zip"
if os.path.exists(zp): os.remove(zp)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(f"{SUB}/script.py", "script.py")
    zf.write(f"{SUB}/requirements.txt", "requirements.txt")
    for fn in sorted(os.listdir(f"{SUB}/model")):
        zf.write(f"{SUB}/model/{fn}", f"model/{fn}")
print("zip:", os.path.getsize(zp), "bytes")
names = sorted(zipfile.ZipFile(zp).namelist()); print(names)
zreq = zipfile.ZipFile(zp).read("requirements.txt").decode()
assert re.search(r"^torch==", zreq, re.M), "zip 실물 requirements에 torch 없음"
zscript = zipfile.ZipFile(zp).read("script.py").decode()
assert "main_tabmcat" in zscript and "tabm_predict" in zscript, "zip 실물 script 확장 누락"
print("zip 실물 검사 OK (torch 핀 + tabmcat 확장 포함)")
