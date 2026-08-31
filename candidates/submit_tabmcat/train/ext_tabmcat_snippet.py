# ============================================================================
# [tabmcat 확장] v5-3 파이프라인 + CatBoost(3-seed) + TabM-mini(3-seed) 앙상블:
# CatBoost seed 42/43/44 평균 0.95 + TabM seed 0/1/2 평균 0.05 -> Platt(계수 JSON).
# TabM은 float64 CPU 순전파. 행별 독립 연산만 사용(Linear/ReLU/Embedding, 배치통계 없음).
# 추론 입력은 test 컬럼뿐 - 행 독립, 외부 데이터/API 없음.
# ============================================================================
import gc

import torch
import torch.nn as nn

TABM_W = 0.05


def apply_platt_coef(p, coef, intercept):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    zz = np.log(p / (1.0 - p)) * float(coef) + float(intercept)
    return np.clip(np.where(zz >= 0, 1.0 / (1.0 + np.exp(-zz)), np.exp(zz) / (1.0 + np.exp(zz))), 0.0, 1.0)


class TabMNet(nn.Module):
    def __init__(self, n_num, card, k, hid, blocks, emb):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, emb) for c in card])
        d = n_num + emb * len(card)
        self.R = nn.Parameter(torch.zeros(k, d))
        self.lin0 = nn.Linear(d, hid)
        self.blocks = nn.ModuleList([nn.Linear(hid, hid) for _ in range(blocks - 1)])
        self.head_w = nn.Parameter(torch.zeros(k, hid))
        self.head_b = nn.Parameter(torch.zeros(k))
        self.aux_w = nn.Parameter(torch.zeros(k, hid, 8))
        self.aux_b = nn.Parameter(torch.zeros(k, 8))

    def forward(self, xn, xc):
        x = torch.cat([xn] + [e(xc[:, i]) for i, e in enumerate(self.embs)], 1)
        h = x.unsqueeze(1) * self.R
        h = torch.relu(self.lin0(h))
        for blk in self.blocks:
            h = torch.relu(blk(h))
        return (h * self.head_w).sum(-1) + self.head_b


def tabm_predict(x_frame, prep, state_paths, batch_size=8192):
    nums, cats = prep["NUMS"], prep["CATS"]
    med = pd.Series(prep["med"], dtype=float)
    mu = pd.Series(prep["mu"], dtype=float)
    sd = pd.Series(prep["sd"], dtype=float)
    xn_frame = x_frame[nums].apply(pd.to_numeric, errors="coerce").fillna(med)
    xn = ((xn_frame - mu) / sd).to_numpy("float64")
    del xn_frame
    xc = np.stack([safe_string(x_frame[c]).map(prep["vocab"][c]).fillna(0).to_numpy("int64") for c in cats], 1)
    net = TabMNet(len(nums), prep["card"], prep["K"], prep["HID"], prep["BLOCKS"], prep["EMB"]).double().eval()
    total = np.zeros(len(x_frame), dtype=float)
    with torch.no_grad():
        for sp in state_paths:
            state = torch.load(sp, map_location="cpu", weights_only=True)
            net.load_state_dict({k: v.double() for k, v in state.items()})
            for i in range(0, len(xn), batch_size):
                tn = torch.from_numpy(xn[i:i + batch_size])
                tc = torch.from_numpy(xc[i:i + batch_size])
                total[i:i + batch_size] += torch.sigmoid(net(tn, tc)).mean(1).numpy()
    return total / len(state_paths)


def main_tabmcat():
    test_path, sample_path = resolve_paths()
    started = time.perf_counter()
    bundle = joblib.load("./model/final_model.joblib")
    cat3 = joblib.load("./model/gbdt_cat3.joblib")
    with open("./model/feature_schema.json", encoding="utf-8") as stream:
        schema = json.load(stream)
    with open("./model/tabm_prep.json", encoding="utf-8") as stream:
        prep = json.load(stream)
    with open("./model/platt_tabmcat.json", encoding="utf-8") as stream:
        platt_cfg = json.load(stream)
    state_paths = ["./model/tabm_s0.pt", "./model/tabm_s1.pt", "./model/tabm_s2.pt"]
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
    comp = bundle["components"][0]
    features = build_features(test, comp["feature_state"])
    x = component_frame(features, comp, schema)
    del features
    gc.collect()
    if list(x.columns) != cat3["feat_cols"]:
        x = x[cat3["feat_cols"]]
    xc_frame = x.copy()
    for c in cat3["cat_cols"]:
        xc_frame[c] = safe_string(xc_frame[c])
    p_cat = np.zeros(len(x), dtype=float)
    for cbm in cat3["cats"]:
        p_cat += np.asarray(cbm.predict_proba(xc_frame)[:, 1], dtype=float)
    p_cat /= len(cat3["cats"])
    del xc_frame
    gc.collect()
    p_tabm = tabm_predict(x, prep, state_paths)
    del x
    gc.collect()
    blended = (1.0 - TABM_W) * p_cat + TABM_W * p_tabm
    final = apply_platt_coef(blended, platt_cfg["platt_coef"], platt_cfg["platt_intercept"])
    inference_seconds = time.perf_counter() - started
    if not np.isfinite(final).all() or ((final < 0) | (final > 1)).any():
        raise ValueError("prediction is not finite or outside [0, 1]")
    if set(test[ID_COL]) != set(sample[ID_COL]) or len(test) != len(sample):
        raise ValueError("test/sample row_id mismatch")
    pred_map = pd.Series(final, index=test[ID_COL])
    result = sample[[ID_COL]].copy()
    result[TARGET_COL] = pred_map.loc[result[ID_COL]].to_numpy()
    os.makedirs("./output", exist_ok=True)
    result.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"Saved: ./output/submission.csv rows={len(result)} load={load_seconds:.4f}s inference={inference_seconds:.4f}s")


if __name__ == "__main__":
    main_tabmcat()
