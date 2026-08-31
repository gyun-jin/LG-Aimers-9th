from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"

# IDs and state variables are categorical. Historical rates/counts stay numerical.
CATEGORICAL_COLUMNS = [
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "hand_matchup",
    "count_state",
    "runner_out_state",
    "pressure_state",
]


def _as_string(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("__MISSING__").astype(str)


def build_features(frame: pd.DataFrame, input_columns: list[str]) -> pd.DataFrame:
    """Create only row-local/as-of features; never aggregate over the test set."""
    missing = [column for column in input_columns if column not in frame.columns]
    extras = [
        column
        for column in frame.columns
        if column not in set(input_columns) | {ID_COL, TARGET_COL}
    ]
    if missing or extras:
        raise ValueError(f"Input schema mismatch: missing={missing}, extras={extras}")

    x = frame.loc[:, input_columns].copy()
    x["hand_matchup"] = _as_string(x["pitcher_hand"]) + "_" + _as_string(x["batter_hand"])
    x["count_state"] = _as_string(x["balls_before"]) + "-" + _as_string(x["strikes_before"])
    x["runner_out_state"] = _as_string(x["base_state"]) + "_o" + _as_string(x["outs_before"])

    close_game = (pd.to_numeric(x["score_diff_pitcher_team"], errors="coerce").abs() <= 2)
    late_inning = pd.to_numeric(x["inning"], errors="coerce") >= 7
    high_li = pd.to_numeric(x["li"], errors="coerce") >= 1.5
    x["pressure_state"] = (
        "late" + late_inning.astype("int8").astype(str)
        + "_close" + close_game.astype("int8").astype(str)
        + "_li" + high_li.astype("int8").astype(str)
    )

    x["score_abs"] = pd.to_numeric(x["score_diff_pitcher_team"], errors="coerce").abs()
    x["close_game"] = close_game.astype("int8")
    x["late_inning"] = late_inning.astype("int8")
    x["log1p_asof_pitcher_n"] = np.log1p(
        pd.to_numeric(x["asof_pitcher_n"], errors="coerce").clip(lower=0)
    )
    x["log1p_asof_batter_n"] = np.log1p(
        pd.to_numeric(x["asof_batter_n"], errors="coerce").clip(lower=0)
    )

    pitcher_success = pd.to_numeric(x["asof_pitcher_success_rate"], errors="coerce")
    batter_success = pd.to_numeric(x["asof_batter_success_rate"], errors="coerce")
    middle = pd.to_numeric(x["asof_pitcher_middle_rate"], errors="coerce")
    reverse = pd.to_numeric(x["asof_pitcher_reverse_rate"], errors="coerce")
    ball = pd.to_numeric(x["asof_pitcher_ball_rate"], errors="coerce")
    breaking = pd.to_numeric(x["asof_pitcher_breaking_rate"], errors="coerce")
    offspeed = pd.to_numeric(x["asof_pitcher_offspeed_rate"], errors="coerce")
    two_strike = (pd.to_numeric(x["strikes_before"], errors="coerce") == 2).astype("int8")
    risp = (
        (pd.to_numeric(x["runner_on_2b"], errors="coerce") == 1)
        | (pd.to_numeric(x["runner_on_3b"], errors="coerce") == 1)
    ).astype("int8")
    outs = pd.to_numeric(x["outs_before"], errors="coerce")

    # Stable v4/v6 row-local interactions used by the tree candidates.
    x["success_gap_pitcher_batter"] = pitcher_success - batter_success
    x["fail_prone"] = middle + reverse + ball
    x["middle_matchup"] = middle * pd.to_numeric(x["asof_batter_middle_rate"], errors="coerce")
    x["offspeed_x_2strike"] = offspeed * two_strike
    x["breaking_x_2strike"] = breaking * two_strike
    x["nonfastball_x_risp"] = (breaking + offspeed) * risp
    x["offspeed_x_outs"] = offspeed * outs
    return x


@dataclass
class Preprocessor:
    input_columns: list[str]
    numerical_columns: list[str]
    categorical_columns: list[str]
    missing_indicator_columns: list[str]
    medians: list[float]
    means: list[float]
    stds: list[float]
    categories: dict[str, list[str]]

    @classmethod
    def fit(cls, frame: pd.DataFrame, input_columns: list[str]) -> "Preprocessor":
        x = build_features(frame, input_columns)
        categorical = [column for column in CATEGORICAL_COLUMNS if column in x.columns]
        numerical_base = [column for column in x.columns if column not in categorical]
        bad_objects = [column for column in numerical_base if x[column].dtype == object]
        if bad_objects:
            raise TypeError(f"Unclassified object columns: {bad_objects}")

        numeric = x[numerical_base].apply(pd.to_numeric, errors="coerce")
        missing_indicators = [column for column in numerical_base if numeric[column].isna().any()]
        for column in missing_indicators:
            numeric[f"{column}__missing"] = numeric[column].isna().astype("float32")
        numerical = list(numeric.columns)

        medians_s = numeric.median(axis=0).fillna(0.0)
        filled = numeric.fillna(medians_s)
        means_s = filled.mean(axis=0)
        stds_s = filled.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        categories = {
            column: sorted(_as_string(x[column]).unique().tolist())
            for column in categorical
        }
        return cls(
            input_columns=list(input_columns),
            numerical_columns=numerical,
            categorical_columns=categorical,
            missing_indicator_columns=missing_indicators,
            medians=medians_s.to_numpy(dtype=np.float64).tolist(),
            means=means_s.to_numpy(dtype=np.float64).tolist(),
            stds=stds_s.to_numpy(dtype=np.float64).tolist(),
            categories=categories,
        )

    @property
    def cat_cardinalities(self) -> list[int]:
        # One explicit slot is reserved for unseen categories.
        return [len(self.categories[column]) + 1 for column in self.categorical_columns]

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = build_features(frame, self.input_columns)
        base_num = [
            column for column in self.numerical_columns if not column.endswith("__missing")
        ]
        numeric = x[base_num].apply(pd.to_numeric, errors="coerce")
        for column in self.missing_indicator_columns:
            numeric[f"{column}__missing"] = numeric[column].isna().astype("float32")
        numeric = numeric.loc[:, self.numerical_columns]
        median = np.asarray(self.medians, dtype=np.float64)
        mean = np.asarray(self.means, dtype=np.float64)
        std = np.asarray(self.stds, dtype=np.float64)
        x_num = numeric.to_numpy(dtype=np.float64, copy=True)
        nan_rows, nan_cols = np.where(~np.isfinite(x_num))
        x_num[nan_rows, nan_cols] = median[nan_cols]
        x_num = ((x_num - mean) / std).astype(np.float32)
        x_num = np.clip(x_num, -12.0, 12.0, out=x_num)

        encoded = []
        for column in self.categorical_columns:
            mapping = {value: index for index, value in enumerate(self.categories[column])}
            unknown = len(mapping)
            values = _as_string(x[column]).map(mapping).fillna(unknown).to_numpy(dtype=np.int64)
            encoded.append(values)
        x_cat = np.column_stack(encoded).astype(np.int64, copy=False)
        return np.ascontiguousarray(x_num), np.ascontiguousarray(x_cat)

    def to_dict(self) -> dict:
        return {
            "input_columns": self.input_columns,
            "numerical_columns": self.numerical_columns,
            "categorical_columns": self.categorical_columns,
            "missing_indicator_columns": self.missing_indicator_columns,
            "medians": self.medians,
            "means": self.means,
            "stds": self.stds,
            "categories": self.categories,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Preprocessor":
        return cls(**payload)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Preprocessor":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
