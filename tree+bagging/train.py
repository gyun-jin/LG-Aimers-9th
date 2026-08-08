"""Train the submission-ready Tree + Bagging model.

Validation is strictly chronological (2019-2023 -> 2024).  Feature engineering
uses only information available immediately before each pitch.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data"
MODEL_DIR = ROOT / "model"
TARGET = "control_success"
ID_COLUMN = "row_id"
RANDOM_STATE = 42

# Selected from temporal validation and the fitted baseline's feature importance.
# IDs and weak hand-crafted game-state interactions are intentionally excluded.
FEATURE_COLUMNS = [
    "season",
    "game_month",
    "game_type",
    "li",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    "success_recent_1_5_gap",
    "success_delta_3",
    "middle_delta_3",
    "pitcher_log_n",
    "pitchmix_log_n",
]

DERIVED_SOURCES = [
    "asof_pitcher_n",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_success_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_middle_rate",
]

TREE_PARAMS = {
    "max_depth": 7,
    "min_samples_leaf": 200,
    "max_features": 0.8,
}
BAGGING_PARAMS = {
    "n_estimators": 100,
    "max_samples": 0.8,
    "bootstrap": True,
    "n_jobs": -1,
    "verbose": 1,
}


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the finalized, leakage-safe feature set."""
    required = set(FEATURE_COLUMNS) | set(DERIVED_SOURCES)
    required -= {
        "success_recent_1_5_gap",
        "success_delta_3",
        "middle_delta_3",
        "pitcher_log_n",
        "pitchmix_log_n",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    features = frame.copy()
    features["success_recent_1_5_gap"] = (
        features["asof_pitcher_prev1_game_success_rate"]
        - features["asof_pitcher_prev5_game_success_rate"]
    )
    features["success_delta_3"] = (
        features["asof_pitcher_prev3_game_success_rate"]
        - features["asof_pitcher_success_rate"]
    )
    features["middle_delta_3"] = (
        features["asof_pitcher_prev3_game_middle_rate"]
        - features["asof_pitcher_middle_rate"]
    )
    features["pitcher_log_n"] = np.log1p(features["asof_pitcher_n"].clip(lower=0))
    features["pitchmix_log_n"] = np.log1p(
        features["asof_pitcher_pitchmix_n"].clip(lower=0)
    )
    return features[FEATURE_COLUMNS]


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical = features.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric = [column for column in FEATURE_COLUMNS if column not in categorical]
    return ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )


def make_model(features: pd.DataFrame) -> Pipeline:
    tree = DecisionTreeClassifier(random_state=RANDOM_STATE, **TREE_PARAMS)
    bagging = BaggingClassifier(estimator=tree, random_state=RANDOM_STATE, **BAGGING_PARAMS)
    return Pipeline([("preprocess", make_preprocessor(features)), ("model", bagging)])


def temporal_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_part = frame.loc[frame["season"].between(2019, 2023)]
    valid_part = frame.loc[frame["season"] == 2024]
    if train_part.empty or valid_part.empty:
        raise ValueError("Expected 2019-2023 training rows and 2024 validation rows.")
    return train_part, valid_part


def evaluate(y_true: pd.Series, probability: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, probability >= 0.5)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, probability)), 6),
        "log_loss": round(float(log_loss(y_true, probability)), 6),
        "brier_score": round(float(brier_score_loss(y_true, probability)), 6),
    }


def save_tree_plot(model: Pipeline) -> None:
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    first_tree = model.named_steps["model"].estimators_[0]
    plt.figure(figsize=(32, 14))
    plot_tree(
        first_tree,
        feature_names=feature_names,
        class_names=["failure (0)", "success (1)"],
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        max_depth=3,
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(ROOT / "tree_preview.png", dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    print("[1/5] Reading train.csv ...", flush=True)
    frame = pd.read_csv(DATA_DIR / "train.csv")
    train_part, valid_part = temporal_split(frame)
    x_train, y_train = make_features(train_part), train_part[TARGET].astype(int)
    x_valid, y_valid = make_features(valid_part), valid_part[TARGET].astype(int)

    print(
        f"[2/5] Temporal validation: {len(x_train):,} train / {len(x_valid):,} valid ...",
        flush=True,
    )
    validation_model = make_model(x_train)
    validation_model.fit(x_train, y_train)
    raw_probability = validation_model.predict_proba(x_valid)[:, 1]

    # A sigmoid map corrects leaf probabilities without changing their ranking.
    print("[3/5] Fitting sigmoid probability calibration ...", flush=True)
    calibrator = LogisticRegression(random_state=RANDOM_STATE)
    calibrator.fit(raw_probability.reshape(-1, 1), y_valid)
    calibrated_probability = calibrator.predict_proba(raw_probability.reshape(-1, 1))[:, 1]
    metrics = {
        "train_seasons": "2019-2023",
        "validation_season": 2024,
        "train_rows": len(train_part),
        "validation_rows": len(valid_part),
        "n_features": len(FEATURE_COLUMNS),
        "raw": evaluate(y_valid, raw_probability),
        "calibrated": evaluate(y_valid, calibrated_probability),
    }
    print(json.dumps(metrics, indent=2), flush=True)

    print(f"[4/5] Refitting final model on {len(frame):,} rows ...", flush=True)
    x_all, y_all = make_features(frame), frame[TARGET].astype(int)
    final_model = make_model(x_all)
    final_model.fit(x_all, y_all)

    MODEL_DIR.mkdir(exist_ok=True)
    artifact = {
        "model": final_model,
        "calibrator": calibrator,
        "feature_columns": FEATURE_COLUMNS,
    }
    with (MODEL_DIR / "model.pkl").open("wb") as model_file:
        pickle.dump(artifact, model_file)
    (ROOT / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_tree_plot(final_model)
    print("[5/5] Export complete.", flush=True)
    print(f"Saved model to {MODEL_DIR / 'model.pkl'}")
    print(f"Saved metrics to {ROOT / 'validation_metrics.json'}")
    print(f"Saved tree visualization to {ROOT / 'tree_preview.png'}")


if __name__ == "__main__":
    main()
