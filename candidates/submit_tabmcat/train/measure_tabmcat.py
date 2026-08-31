"""7호(tabmcat) 리허설: train 2024시즌 25.4만행을 test 형식으로 변환해 클린 실행, 시간·피크 메모리 측정."""
import os, resource, shutil, subprocess, sys, tempfile, time, zipfile
import pandas as pd
import joblib

SUB = "/root/jegu/work/tabmcat_submit"; DATA = "/root/jegu/data"
tmp = tempfile.mkdtemp(prefix="measure_tabmcat_")
zipfile.ZipFile(f"{SUB}/submit.zip").extractall(tmp)
os.makedirs(f"{tmp}/open", exist_ok=True)
bundle = joblib.load(f"{tmp}/model/final_model.joblib")
df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig")
big = df[df["season"] == 2024].copy()
print("pseudo-test rows:", len(big), flush=True)
big[["row_id"] + bundle["input_columns"]].reset_index(drop=True).to_csv(f"{tmp}/open/test.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"row_id": big["row_id"], "control_success": 0.5}).to_csv(f"{tmp}/open/sample_submission.csv", index=False, encoding="utf-8-sig")
del df, big
t0 = time.time()
r = subprocess.run([sys.executable, "script.py"], cwd=tmp, capture_output=True, text=True)
dt = time.time() - t0
peak_gb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1048576
print(r.stdout[-500:])
if r.returncode != 0:
    print("STDERR:", r.stderr[-800:]); sys.exit(1)
s = pd.read_csv(f"{tmp}/output/submission.csv")
print(f"rows={len(s)} time={dt:.0f}s peak={peak_gb:.1f}GB pred[{s.control_success.min():.4f},{s.control_success.max():.4f}] mean={s.control_success.mean():.4f}")
shutil.rmtree(tmp, ignore_errors=True)
