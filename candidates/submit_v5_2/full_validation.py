from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data"

spec = importlib.util.spec_from_file_location("task52_train", ROOT / "train.py")
task52 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(task52)
task5 = task52.task5
task51 = task52.task51


FULL_CANDIDATES = [
    "v5_no_trackman_recheck",
    "tm_metadata_physical_core",
    "tm_metadata_physical_pitchmix",
]
CALIBRATION_BY_CANDIDATE = {
    "v5_no_trackman_recheck": "prior_correction",
    "tm_metadata_physical_core": "platt",
    "tm_metadata_physical_pitchmix": "platt",
}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [record["metrics"] for record in records]
    return {
        "folds": metrics,
        "mean_brier": float(np.mean([m["brier"] for m in metrics])),
        "worst_brier": float(np.max([m["brier"] for m in metrics])),
        "mean_metrics": {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]},
    }


def apply_progressive_calibration(kind: str, fold_preds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    previous_y: list[np.ndarray] = []
    previous_p: list[np.ndarray] = []
    for item in fold_preds:
        if previous_y:
            cal = task5.fit_calibration(kind, np.concatenate(previous_y), np.concatenate(previous_p))
        else:
            cal = {"kind": "none"}
        pred = task5.apply_calibration(cal, item["pred"])
        output.append({"season": item["season"], "metrics": task5.metrics(item["y"], pred)})
        previous_y.append(item["y"])
        previous_p.append(item["pred"])
    return output


def main() -> None:
    schema = task5.load_v3_schema()
    task5.INPUT_COLUMNS = schema["input_columns"]
    task5.V3_FEATURE_COLUMNS = schema["feature_columns"]
    task5.V3_CATEGORICAL_COLUMNS = schema["categorical_columns"]
    train = pd.read_csv(DATA_DIR / "train.csv")
    y = train[task5.TARGET_COL].to_numpy(dtype=int)
    tm_state = task51.build_trackman_state(train)

    raw_records: dict[str, list[dict[str, Any]]] = {name: [] for name in FULL_CANDIDATES}
    for valid_season in task52.FOLD_SEASONS:
        seasons = train["season"].to_numpy()
        tr_idx = np.flatnonzero(seasons < valid_season)
        va_idx = np.flatnonzero(seasons == valid_season)
        state = task5.fit_feature_state(train.iloc[tr_idx], y[tr_idx], schema["alpha"])
        base_features = task5.build_v3_features(train, state)
        tm_features = task51.build_trackman_features(train, tm_state, "mapping_all_shrink")
        y_valid = y[va_idx]
        print(f"fold={valid_season} train={len(tr_idx):,} valid={len(va_idx):,}", flush=True)
        for feature_set in FULL_CANDIDATES:
            x_all, cats = task52.prepare_features(base_features, tm_features, feature_set)
            x_train = x_all.iloc[tr_idx]
            x_valid = x_all.iloc[va_idx]
            cat_pred = task52.fit_cat(x_train, y[tr_idx], x_valid, y_valid, cats, task52.CAT_CONFIGS["cat_current"])
            lgb_pred = task52.fit_lgb(x_train, y[tr_idx], x_valid, cats, task52.LGB_CONFIGS["lgb_current"])
            pred = 0.8 * cat_pred + 0.2 * lgb_pred
            metric = task5.metrics(y_valid, pred)
            raw_records[feature_set].append({"season": valid_season, "y": y_valid, "pred": pred, "metrics": metric})
            print(f"  {feature_set} raw_brier={metric['brier']:.8f}", flush=True)

    raw_summary = {name: summarize(records) for name, records in raw_records.items()}
    calibrated_records = {
        name: apply_progressive_calibration(CALIBRATION_BY_CANDIDATE[name], raw_records[name])
        for name in FULL_CANDIDATES
    }
    calibrated_summary = {name: summarize(records) for name, records in calibrated_records.items()}
    output = {
        "references": {"v5_mean_brier": task52.V5_REFERENCE_MEAN, "v5_worst_brier": task52.V5_REFERENCE_WORST, "v5_folds": task52.V5_REFERENCE_FOLDS},
        "calibration_by_candidate": CALIBRATION_BY_CANDIDATE,
        "raw_candidates": raw_summary,
        "calibrated_candidates": calibrated_summary,
    }
    (ROOT / "output").mkdir(exist_ok=True)
    (ROOT / "output" / "full_validation_results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, default=task52.json_default), encoding="utf-8")

    lines = [
        "# 5-2 전체 fold 재검증",
        "",
        "| candidate | calibration | mean Brier | worst Brier | 2022 | 2023 | 2024 | mean AUC | mean_pred | mean_target |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in sorted(calibrated_summary.items(), key=lambda pair: (pair[1]["mean_brier"], pair[1]["worst_brier"])):
        folds = item["folds"]
        avg = item["mean_metrics"]
        lines.append(f"| {name} | {CALIBRATION_BY_CANDIDATE[name]} | {item['mean_brier']:.8f} | {item['worst_brier']:.8f} | {folds[0]['brier']:.8f} | {folds[1]['brier']:.8f} | {folds[2]['brier']:.8f} | {avg['auc']:.6f} | {avg['mean_pred']:.6f} | {avg['mean_target']:.6f} |")
    lines.extend([
        "",
        f"- /5 reference mean/worst: `{task52.V5_REFERENCE_MEAN:.8f}` / `{task52.V5_REFERENCE_WORST:.8f}`",
        "- 전체 fold 기준으로 /5보다 평균과 최악 fold가 모두 낮아야 제출 후보로 본다.",
    ])
    (ROOT / "full_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
