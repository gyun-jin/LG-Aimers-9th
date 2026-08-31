from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"
FOLDS = (2022, 2023, 2024)
EPS = 1e-6


def brier(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((np.asarray(prediction, dtype=np.float64) - y) ** 2))


def logit(prediction: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(prediction, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p)).reshape(-1, 1)


def fit_platt(y: np.ndarray, prediction: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=42)
    model.fit(logit(prediction), y)
    return model


def apply_platt(model: LogisticRegression, prediction: np.ndarray) -> np.ndarray:
    return model.predict_proba(logit(prediction))[:, 1]


def fold_metrics(y: np.ndarray, prediction: np.ndarray, seasons: np.ndarray) -> dict[str, Any]:
    fold_seasons = sorted(np.unique(seasons).astype(int).tolist())
    folds = {
        str(season): brier(y[seasons == season], prediction[seasons == season])
        for season in fold_seasons
    }
    values = list(folds.values())
    return {
        "fold_brier": folds,
        "mean_fold_brier": float(np.mean(values)),
        "worst_fold_brier": float(np.max(values)),
        "row_weighted_brier": brier(y, prediction),
        "mean_prediction": float(np.mean(prediction)),
        "mean_target": float(np.mean(y)),
    }


def progressive_platt(y: np.ndarray, prediction: np.ndarray, seasons: np.ndarray) -> np.ndarray:
    calibrated = np.empty_like(prediction, dtype=np.float64)
    for season in FOLDS:
        valid = seasons == season
        previous = seasons < season
        if previous.any():
            calibrated[valid] = apply_platt(fit_platt(y[previous], prediction[previous]), prediction[valid])
        else:
            calibrated[valid] = prediction[valid]
    return calibrated


def select_grid_weight(
    y: np.ndarray,
    cat: np.ndarray,
    tabm: np.ndarray,
    seasons: np.ndarray,
) -> float:
    candidates = []
    for weight in np.round(np.arange(0.0, 1.0001, 0.05), 2):
        prediction = weight * cat + (1.0 - weight) * tabm
        metrics = fold_metrics(y, prediction, seasons)
        candidates.append(
            (metrics["mean_fold_brier"], metrics["worst_fold_brier"], -weight, float(weight))
        )
    return min(candidates)[-1]


def rolling_weight_and_platt(
    y: np.ndarray,
    cat: np.ndarray,
    tabm: np.ndarray,
    seasons: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    prediction = np.empty_like(cat, dtype=np.float64)
    decisions: dict[str, Any] = {}
    for season in FOLDS:
        valid = seasons == season
        previous = seasons < season
        if not previous.any():
            weight = 1.0
            prediction[valid] = cat[valid]
            decisions[str(season)] = {
                "catboost_weight": weight,
                "calibration": "none (no previous OOF)",
            }
            continue
        weight = select_grid_weight(y[previous], cat[previous], tabm[previous], seasons[previous])
        previous_blend = weight * cat[previous] + (1.0 - weight) * tabm[previous]
        current_blend = weight * cat[valid] + (1.0 - weight) * tabm[valid]
        calibrator = fit_platt(y[previous], previous_blend)
        prediction[valid] = apply_platt(calibrator, current_blend)
        decisions[str(season)] = {
            "catboost_weight": weight,
            "tabm_weight": 1.0 - weight,
            "calibration": "Platt fit only on earlier folds",
        }
    return prediction, decisions


def load_aligned() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    for season in FOLDS:
        cat = np.load(OUTPUT_DIR / f"catboost_oof_{season}.npz")
        tab = np.load(OUTPUT_DIR / f"tabm_oof_{season}.npz")
        cat_ids = cat["row_id"].astype(str)
        tab_ids = tab["row_id"].astype(str)
        if not np.array_equal(cat_ids, tab_ids):
            raise ValueError(f"row_id mismatch in fold {season}")
        if not np.array_equal(cat["y_true"], tab["y_true"]):
            raise ValueError(f"target mismatch in fold {season}")
        rows.append((cat_ids, cat["y_true"], cat["raw_pred"], tab["raw_pred"], cat["season"]))
    return tuple(np.concatenate([row[index] for row in rows]) for index in range(5))  # type: ignore[return-value]


def main() -> None:
    global OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        OUTPUT_DIR = HERE / "output_smoke"
    row_id, y, cat, tabm, seasons = load_aligned()
    y = y.astype(np.int8)
    seasons = seasons.astype(np.int16)
    residual_correlation = {
        "all": float(np.corrcoef(cat - y, tabm - y)[0, 1]),
        **{
            str(season): float(
                np.corrcoef((cat - y)[seasons == season], (tabm - y)[seasons == season])[0, 1]
            )
            for season in FOLDS
        },
    }
    candidates: list[dict[str, Any]] = []
    weights = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    difference = cat.astype(np.float64) - tabm.astype(np.float64)
    denominator = float(np.dot(difference, difference))
    closed_form = float(np.clip(np.dot(difference, y - tabm) / denominator, 0.0, 1.0))
    weights = np.unique(np.append(weights, closed_form))
    for cat_weight in weights:
        raw = cat_weight * cat + (1.0 - cat_weight) * tabm
        progressive = progressive_platt(y, raw, seasons)
        global_model = fit_platt(y, raw)
        global_prediction = apply_platt(global_model, raw)
        candidates.append(
            {
                "catboost_weight": float(cat_weight),
                "tabm_weight": float(1.0 - cat_weight),
                "raw": fold_metrics(y, raw, seasons),
                "progressive_platt": fold_metrics(y, progressive, seasons),
                "global_platt_insample_diagnostic": fold_metrics(y, global_prediction, seasons),
            }
        )
    ranked = sorted(
        candidates,
        key=lambda item: (
            item["progressive_platt"]["mean_fold_brier"],
            item["progressive_platt"]["worst_fold_brier"],
        ),
    )
    selected = ranked[0]
    selected_raw = (
        selected["catboost_weight"] * cat + selected["tabm_weight"] * tabm
    )
    rolling_prediction, rolling_decisions = rolling_weight_and_platt(y, cat, tabm, seasons)
    final_calibrator = fit_platt(y, selected_raw)
    joblib.dump(
        {
            "kind": "platt",
            "model": final_calibrator,
            "catboost_weight": selected["catboost_weight"],
            "tabm_weight": selected["tabm_weight"],
            "folds": list(FOLDS),
        },
        OUTPUT_DIR / "ensemble_calibrator.joblib",
    )
    report = {
        "rows": len(row_id),
        "folds": list(FOLDS),
        "selection_metric": "progressive_platt mean fold Brier, then worst fold Brier",
        "closed_form_row_weighted_raw_catboost_weight": closed_form,
        "residual_correlation": residual_correlation,
        "baselines": {
            "catboost_raw": fold_metrics(y, cat, seasons),
            "catboost_progressive_platt": fold_metrics(y, progressive_platt(y, cat, seasons), seasons),
            "tabm_raw": fold_metrics(y, tabm, seasons),
            "tabm_progressive_platt": fold_metrics(y, progressive_platt(y, tabm, seasons), seasons),
        },
        "selected": selected,
        "rolling_nested": {
            "decisions": rolling_decisions,
            "metrics": fold_metrics(y, rolling_prediction, seasons),
        },
        "candidates": candidates,
        "cautions": [
            "CatBoost OOF uses seed 42 only as a proxy; the submitted v5_8 artifact averages five seeds.",
            "TabM OOF uses one epoch because CPU training takes 7.5-13.8 minutes per fold and epoch.",
            "CatBoost OOF uses feature state stored by the final full-training v5_8 artifact.",
            "CatBoost OOF uses official train.csv; original v5_8 validation removed 116 hand-mismatch rows.",
            "Both base learners use the target fold for early stopping, matching their existing validation style.",
            "Global Platt metrics are in-sample diagnostics and are not used for weight selection.",
        ],
    }
    (OUTPUT_DIR / "blend_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# TabM + CatBoost OOF blend report",
        "",
        f"- Rows: {len(row_id):,}",
        f"- Residual correlation: {residual_correlation['all']:.6f}",
        f"- Selected weights: CatBoost {selected['catboost_weight']:.6f}, TabM {selected['tabm_weight']:.6f}",
        f"- Selected progressive-Platt mean Brier: {selected['progressive_platt']['mean_fold_brier']:.8f}",
        f"- Selected progressive-Platt worst Brier: {selected['progressive_platt']['worst_fold_brier']:.8f}",
        f"- Rolling nested mean Brier: {report['rolling_nested']['metrics']['mean_fold_brier']:.8f}",
        "",
        "## Baselines",
        "",
        "| model | raw mean | raw worst | progressive mean | progressive worst |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("catboost", "tabm"):
        raw = report["baselines"][f"{name}_raw"]
        prog = report["baselines"][f"{name}_progressive_platt"]
        lines.append(
            f"| {name} | {raw['mean_fold_brier']:.8f} | {raw['worst_fold_brier']:.8f} "
            f"| {prog['mean_fold_brier']:.8f} | {prog['worst_fold_brier']:.8f} |"
        )
    lines.extend(
        [
            "",
            "## Rolling nested selection",
            "",
            "| validation season | CatBoost | TabM | Brier |",
            "|---:|---:|---:|---:|",
        ]
    )
    for season in FOLDS:
        decision = report["rolling_nested"]["decisions"][str(season)]
        metric = report["rolling_nested"]["metrics"]["fold_brier"][str(season)]
        lines.append(
            f"| {season} | {decision['catboost_weight']:.2f} "
            f"| {decision.get('tabm_weight', 0.0):.2f} | {metric:.8f} |"
        )
    lines.extend(
        [
            "",
            "## Weight grid",
            "",
            "| CatBoost | TabM | raw mean | raw worst | progressive mean | progressive worst |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in sorted(candidates, key=lambda value: value["catboost_weight"]):
        lines.append(
            f"| {item['catboost_weight']:.6f} | {item['tabm_weight']:.6f} "
            f"| {item['raw']['mean_fold_brier']:.8f} | {item['raw']['worst_fold_brier']:.8f} "
            f"| {item['progressive_platt']['mean_fold_brier']:.8f} "
            f"| {item['progressive_platt']['worst_fold_brier']:.8f} |"
        )
    lines.extend(["", "## Cautions", ""] + [f"- {item}" for item in report["cautions"]])
    (OUTPUT_DIR / "blend_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "residual_correlation": residual_correlation}, indent=2))


if __name__ == "__main__":
    main()
