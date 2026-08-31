"""로컬 최종 추론 검증용 유틸리티."""

# [추가 구현]
# 목적: 학습 bundle을 test 내부 통계 없이 로드·예측하고 제출 계약을 강제
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from calibration import apply_calibrator
from feature_engineering import ID_COL, TARGET_COL, build_features, prepare_catboost
from validation import apply_category_encoder


def predict_bundle(bundle: dict[str, Any], test: pd.DataFrame) -> np.ndarray:
    expected = bundle["input_columns"]
    actual = [c for c in test.columns if c != ID_COL]
    if actual != expected:
        raise ValueError(f"입력 컬럼/순서 불일치: expected={expected}, actual={actual}")
    predictions = np.zeros(len(test), dtype=float)
    for component in bundle["components"]:
        name = component["name"]
        if name == "random_forest":
            x = test.loc[:, component["feature_columns"]].copy()
        else:
            x = build_features(test, component["feature_state"])
            if list(x.columns) != component["feature_columns"]:
                raise ValueError(f"{name} 파생 피처 schema 불일치")
            if name == "catboost":
                x = prepare_catboost(x, component["categorical_columns"])
            else:
                x = apply_category_encoder(x, component["encoder"])
        part = component["model"].predict_proba(x)[:, 1]
        predictions += float(component["weight"]) * np.asarray(part, dtype=float)
    return predictions


def build_submission(
    test: pd.DataFrame,
    sample: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    if list(sample.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(f"submission 컬럼 불일치: {list(sample.columns)}")
    if test[ID_COL].duplicated().any() or sample[ID_COL].duplicated().any():
        raise ValueError("중복 row_id")
    test_ids, sample_ids = set(test[ID_COL]), set(sample[ID_COL])
    if test_ids != sample_ids:
        raise ValueError(
            f"ID 집합 불일치: missing={sample_ids - test_ids}, extra={test_ids - sample_ids}"
        )
    p = np.asarray(predictions, dtype=float)
    if len(p) != len(test) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("예측 행 수, 유한성 또는 확률 범위 오류")
    pred_map = pd.Series(p, index=test[ID_COL])
    if pred_map.index.duplicated().any():
        raise ValueError("예측 row_id 중복")
    result = sample[[ID_COL]].copy()
    result[TARGET_COL] = pred_map.loc[result[ID_COL]].to_numpy()
    if result[ID_COL].tolist() != sample[ID_COL].tolist():
        raise ValueError("출력 row_id 순서 오류")
    return result


def calibrated_predictions(
    bundle: dict[str, Any], calibrator: dict[str, Any], test: pd.DataFrame
) -> np.ndarray:
    return apply_calibrator(calibrator, predict_bundle(bundle, test))
