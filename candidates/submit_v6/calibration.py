"""시간 기반 OOF 확률만 사용하는 보정기 비교와 직렬화 규약."""

# [추가 구현]
# 목적: 없음, Platt, temperature, isotonic을 같은 Brier 함수로 비교
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from metrics import probability_metrics


def _logit(preds: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(preds, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def fit_calibrator(kind: str, preds: np.ndarray, y_true: np.ndarray) -> dict[str, Any]:
    p = np.asarray(preds, dtype=float)
    y = np.asarray(y_true, dtype=int)
    if kind == "none":
        return {"kind": "none"}
    if kind == "platt":
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
        model.fit(_logit(p).reshape(-1, 1), y)
        return {"kind": "platt", "model": model}
    if kind == "temperature":
        logits = _logit(p)

        def objective(log_temperature: float) -> float:
            calibrated = expit(logits / np.exp(log_temperature))
            return float(np.mean((calibrated - y) ** 2))

        result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
        return {"kind": "temperature", "temperature": float(np.exp(result.x))}
    if kind == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(p, y)
        return {"kind": "isotonic", "model": model}
    raise ValueError(f"지원하지 않는 calibrator: {kind}")


def apply_calibrator(calibrator: dict[str, Any], preds: np.ndarray) -> np.ndarray:
    p = np.asarray(preds, dtype=float)
    kind = calibrator["kind"]
    if kind == "none":
        result = p
    elif kind == "platt":
        result = calibrator["model"].predict_proba(_logit(p).reshape(-1, 1))[:, 1]
    elif kind == "temperature":
        result = expit(_logit(p) / float(calibrator["temperature"]))
    elif kind == "isotonic":
        result = calibrator["model"].predict(p)
    else:
        raise ValueError(f"지원하지 않는 calibrator: {kind}")
    return np.clip(np.asarray(result, dtype=float), 0.0, 1.0)


def compare_calibrators(
    train_preds: np.ndarray,
    train_y: np.ndarray,
    eval_preds: np.ndarray,
    eval_y: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    metrics: dict[str, dict[str, Any]] = {}
    fitted: dict[str, dict[str, Any]] = {}
    for kind in ["none", "platt", "temperature", "isotonic"]:
        calibrator = fit_calibrator(kind, train_preds, train_y)
        calibrated = apply_calibrator(calibrator, eval_preds)
        metrics[kind] = probability_metrics(eval_y, calibrated)
        fitted[kind] = calibrator
    return metrics, fitted
