"""확률 예측 대회의 공식 BSS와 calibration 진단."""

# [추가 구현]
# 목적: Accuracy 대신 Brier Score, BSS, calibration을 일관되게 평가
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score


def brier_score(y_true: np.ndarray, preds: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(preds, dtype=float)
    return float(np.mean((p - y) ** 2))


def brier_skill_score(y_true: np.ndarray, preds: np.ndarray) -> float:
    """대회 공식과 정확히 같은 scale의 Brier Skill Score."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(preds, dtype=float)
    brier = np.mean((p - y) ** 2)
    reference_brier = y.mean() * (1.0 - y.mean())
    if reference_brier <= 0:
        return 0.0
    return float(max(0.0, 100000.0 * (1.0 - brier / reference_brier)))


def calibration_intercept_slope(y_true: np.ndarray, preds: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(preds, dtype=float), 1e-6, 1.0 - 1e-6)
    logit = np.log(p / (1.0 - p))

    def loss(beta: np.ndarray) -> float:
        z = beta[0] + beta[1] * logit
        return float(np.mean(np.logaddexp(0.0, z) - y * z))

    result = minimize(loss, x0=np.array([0.0, 1.0]), method="L-BFGS-B")
    if not result.success:
        return float("nan"), float("nan")
    return float(result.x[0]), float(result.x[1])


def probability_bins(y_true: np.ndarray, preds: np.ndarray, n_bins: int = 10) -> list[dict[str, Any]]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(preds, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    indexes = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for index in range(n_bins):
        mask = indexes == index
        rows.append(
            {
                "interval": f"[{edges[index]:.1f}, {edges[index + 1]:.1f}{']' if index == n_bins - 1 else ')'}",
                "count": int(mask.sum()),
                "prediction_mean": float(p[mask].mean()) if mask.any() else None,
                "actual_rate": float(y[mask].mean()) if mask.any() else None,
            }
        )
    return rows


def probability_metrics(y_true: np.ndarray, preds: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(preds, dtype=float)
    intercept, slope = calibration_intercept_slope(y, p)
    auc = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None
    return {
        "brier_score": brier_score(y, p),
        "brier_skill_score": brier_skill_score(y, p),
        "roc_auc_reference": auc,
        "prediction_mean": float(p.mean()),
        "target_mean": float(y.mean()),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "probability_bins": probability_bins(y, p),
    }
