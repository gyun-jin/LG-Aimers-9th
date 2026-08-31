"""7호(tabmcat) 제출물 규칙 검사 (15항목: 행독립 3종 + 원격API + 외부데이터 + zip구성 + 사전학습 + 버전핀 + import커버 + 클린실행/출력 + numpy1.26 재현실행)."""
import importlib.util, json, os, re, shutil, subprocess, sys, tempfile, zipfile
import numpy as np, pandas as pd

SUB = "/root/jegu/work/tabmcat_submit"
DATA = "/root/jegu/data"
results = []
def check(name, ok, detail=""):
    results.append((name, ok)); print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)

script = open(f"{SUB}/script.py", encoding="utf-8").read()
print("[2] 원격 API")
hits = re.findall(r"requests|urllib|http\.client|socket|openai|generativeai|anthropic|huggingface|transformers|torch\.hub|from_pretrained|boto3|gdown|load_state_dict_from_url", script)
check("네트워크·외부API·원격로드 없음", not hits, str(sorted(set(hits))) if hits else "")

print("[3] 외부 데이터")
opened = re.findall(r'(?:read_csv|joblib\.load|np\.load|torch\.load|open)\(\s*([^,)]+)', script)
bad = [o for o in opened if re.search(r"trackman|mapping|kbo|statiz|http", o, re.I)]
check("외부 데이터 참조 없음", not bad, str(bad) if bad else "")
z = zipfile.ZipFile(f"{SUB}/submit.zip"); names = z.namelist()
extra = [n for n in names if not (n in ("script.py","requirements.txt") or n.startswith("model/"))]
check("zip 구성 script.py+requirements+model/", not extra, str(extra) if extra else "")
check("zip forward-slash", all("\\" not in n for n in names))

print("[4][5] requirements")
req = [r.strip() for r in open(f"{SUB}/requirements.txt").read().splitlines() if r.strip()]
pre = [r for r in req if re.match(r"^(transformers|huggingface|timm|torchvision|sentence-transformers|open_clip)", r, re.I)]
check("사전학습 패키지 없음", not pre, str(pre) if pre else "")
pkg_lines = [r for r in req if not r.startswith("--")]
unpinned = [r for r in pkg_lines if "==" not in r]
check("requirements 전 줄 버전 핀", not unpinned, f"미핀={unpinned}" if unpinned else "")
import ast as _ast
tree = _ast.parse(script); imps = set()
for node in _ast.walk(tree):
    if isinstance(node, _ast.Import): imps |= {a.name.split(".")[0] for a in node.names}
    elif isinstance(node, _ast.ImportFrom) and node.module: imps.add(node.module.split(".")[0])
std = {"os","sys","json","time","gc","math","re","warnings","typing","pathlib","functools","itertools","collections"}
third = imps - std
have = {r.split("==")[0].lower() for r in pkg_lines}
alias = {"sklearn": "scikit-learn"}
uncov = [m for m in third if alias.get(m, m).lower() not in have]
check("script import 전부 requirements에 존재", not uncov, f"미포함={sorted(uncov)}" if uncov else f"import={sorted(third)}")

print("[6] 클린 추출 실행 (실측 test 24.5만행)")
tmp = tempfile.mkdtemp(prefix="audit_tabmcat_")
z.extractall(tmp)
os.makedirs(f"{tmp}/open", exist_ok=True)
shutil.copy(f"{DATA}/test.csv", f"{tmp}/open/test.csv")
shutil.copy(f"{DATA}/sample_submission.csv", f"{tmp}/open/sample_submission.csv")
r = subprocess.run([sys.executable, "script.py"], cwd=tmp, capture_output=True, text=True)
check("script.py 정상 종료", r.returncode == 0, (r.stderr or "")[-300:] if r.returncode else "")
out = f"{tmp}/output/submission.csv"
if os.path.exists(out):
    s = pd.read_csv(out); sample = pd.read_csv(f"{DATA}/sample_submission.csv")
    check("출력 컬럼", list(s.columns) == ["row_id","control_success"])
    check("출력 행수", len(s) == len(sample))
    check("NaN 없음", s.control_success.notna().all())
    check("값 범위 [0,1]", ((s.control_success >= 0) & (s.control_success <= 1)).all())

print("[7] 채점환경 재현 실행 (numpy 1.26.4 / pandas 2.0.3 PYTHONPATH)")
tmp2 = tempfile.mkdtemp(prefix="audit_tabmcat_np126_")
z.extractall(tmp2)
os.makedirs(f"{tmp2}/open", exist_ok=True)
df_small = pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig", nrows=2000)
df_small.to_csv(f"{tmp2}/open/test.csv", index=False, encoding="utf-8-sig")
sam = pd.read_csv(f"{DATA}/sample_submission.csv", encoding="utf-8-sig")
sam[sam["row_id"].isin(df_small["row_id"])].to_csv(f"{tmp2}/open/sample_submission.csv", index=False, encoding="utf-8-sig")
env = os.environ.copy(); env["PYTHONPATH"] = "/tmp/pyenv_req"
r2 = subprocess.run([sys.executable, "script.py"], cwd=tmp2, env=env, capture_output=True, text=True)
np_ver = subprocess.run([sys.executable, "-c", "import numpy,pandas;print(numpy.__version__,pandas.__version__)"], env=env, capture_output=True, text=True).stdout.strip()
check(f"numpy1.26 재현환경({np_ver}) 정상 종료", r2.returncode == 0, (r2.stderr or "")[-300:] if r2.returncode else "")
shutil.rmtree(tmp2, ignore_errors=True)

print("[1] 행 독립성")
sys.path.insert(0, tmp); cwd = os.getcwd(); os.chdir(tmp)
spec = importlib.util.spec_from_file_location("scr", f"{tmp}/script.py")
scr = importlib.util.module_from_spec(spec); spec.loader.exec_module(scr)
import joblib as jl
bundle = jl.load("./model/final_model.joblib")
schema = json.load(open("./model/feature_schema.json"))
pc = json.load(open("./model/platt_tabmcat.json"))
prep = json.load(open("./model/tabm_prep.json"))
cat3 = jl.load("./model/gbdt_cat3.joblib")
sps = ["./model/tabm_s0.pt", "./model/tabm_s1.pt", "./model/tabm_s2.pt"]
df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig", nrows=150000)
big = df[bundle["input_columns"] + ["row_id"]].tail(1500).reset_index(drop=True)
def infer(frame):
    fr = frame[bundle["input_columns"]].reset_index(drop=True)
    comp = bundle["components"][0]
    x = scr.component_frame(scr.build_features(fr, comp["feature_state"]), comp, schema)
    x = x[cat3["feat_cols"]]
    xc = x.copy()
    for c in cat3["cat_cols"]: xc[c] = scr.safe_string(xc[c])
    p = np.zeros(len(x))
    for cbm in cat3["cats"]: p += cbm.predict_proba(xc)[:, 1]
    p /= len(cat3["cats"])
    pt = scr.tabm_predict(x, prep, sps)
    return scr.apply_platt_coef((1 - scr.TABM_W) * p + scr.TABM_W * pt, pc["platt_coef"], pc["platt_intercept"])
p_all = infer(big)
pick = sorted(np.random.default_rng(0).choice(len(big), 6, replace=False))
p_alone = np.array([infer(big.iloc[[i]])[0] for i in pick])
p_rev = infer(big.iloc[::-1].reset_index(drop=True))[::-1]
p_half = infer(big.iloc[:700])
d1 = float(np.abs(p_all[pick]-p_alone).max()); d2 = float(np.abs(p_all-p_rev).max()); d3 = float(np.abs(p_all[:700]-p_half).max())
check("단독행==전체", d1 < 1e-9, f"{d1:.1e}")
check("순서뒤집기 불변", d2 < 1e-9, f"{d2:.1e}")
check("부분집합 불변", d3 < 1e-9, f"{d3:.1e}")
os.chdir(cwd); shutil.rmtree(tmp, ignore_errors=True)

fails = [n for n, ok in results if not ok]
print(f"\n=== {'ALL PASS (' + str(len(results)) + '항목)' if not fails else 'FAIL: ' + str(fails)} ===")
sys.exit(1 if fails else 0)
