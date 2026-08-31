"""전체 배포 데이터의 구조, 누수 위험, 중복 관계 및 ID 매핑 가능성 감사."""

# [추가 구현]
# 목적: 모델 학습 전 요구된 데이터 검사를 재현 가능하게 JSON으로 저장
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"
DESIGNATED_CATEGORICAL = [
    "top_bottom",
    "game_type",
    "base_state",
    "game_dayofweek",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]


def _python_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _frame_schema(df: pd.DataFrame) -> dict[str, Any]:
    numeric = df.select_dtypes(include=[np.number])
    ranges = {
        col: {
            "min": _python_value(numeric[col].min()),
            "max": _python_value(numeric[col].max()),
        }
        for col in numeric.columns
    }
    missing = {
        col: {
            "count": int(df[col].isna().sum()),
            "ratio": float(df[col].isna().mean()),
        }
        for col in df.columns
    }
    categorical = {}
    for col in df.select_dtypes(exclude=[np.number]).columns:
        uniques = df[col].dropna().unique()
        categorical[col] = {
            "nunique": int(len(uniques)),
            "sample_values": [_python_value(x) for x in uniques[:30]],
        }
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing": missing,
        "infinite_counts": {
            col: int(np.isinf(numeric[col].to_numpy(dtype=float, na_value=np.nan)).sum())
            for col in numeric.columns
        },
        "numeric_ranges": ranges,
        "categorical_values": categorical,
    }


def _new_entity_stats(train: pd.DataFrame, id_col: str) -> dict[str, Any]:
    old_ids = set(train.loc[train["season"] < 2024, id_col].dropna().unique())
    rows_2024 = train.loc[train["season"] == 2024, id_col]
    ids_2024 = set(rows_2024.dropna().unique())
    new_ids = ids_2024 - old_ids
    return {
        "old_unique": len(old_ids),
        "season_2024_unique": len(ids_2024),
        "new_2024_unique": len(new_ids),
        "new_2024_unique_ratio": len(new_ids) / max(1, len(ids_2024)),
        "new_2024_row_ratio": float(rows_2024.isin(new_ids).mean()),
    }


def run_audit(data_dir: Path, output_path: Path) -> dict[str, Any]:
    train = pd.read_csv(data_dir / "train.csv", encoding="utf-8-sig")
    test = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig")
    sample = pd.read_csv(data_dir / "sample_submission.csv", encoding="utf-8-sig")

    input_train_cols = [c for c in train.columns if c != TARGET_COL]
    test_cols = list(test.columns)
    train_ids = set(train["pitcher_id"].dropna().unique())
    train_batters = set(train["batter_id"].dropna().unique())
    test_pitchers = set(test["pitcher_id"].dropna().unique())
    test_batters = set(test["batter_id"].dropna().unique())

    track = pd.read_csv(
        data_dir / "trackman_history.csv",
        encoding="utf-8-sig",
        usecols=["season", "pitcher_trackman_id", "batter_trackman_id"],
    )
    track_pitchers = set(track["pitcher_trackman_id"].dropna().unique())
    track_batters = set(track["batter_trackman_id"].dropna().unique())

    runners_sum = train[["runner_on_1b", "runner_on_2b", "runner_on_3b"]].sum(axis=1)
    pitchmix_sum = train[
        [
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ]
    ].sum(axis=1, min_count=3)
    numeric_corr = train.select_dtypes(include=[np.number]).corr()
    high_corr_pairs = []
    cols = list(numeric_corr.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            corr = numeric_corr.loc[left, right]
            if pd.notna(corr) and abs(corr) >= 0.995:
                high_corr_pairs.append({"left": left, "right": right, "correlation": float(corr)})

    relations = {
        "pitcher_n_equals_pitchmix_n_ratio": float(
            (train["asof_pitcher_n"] == train["asof_pitcher_pitchmix_n"]).mean()
        ),
        "run_total_equals_components_ratio": float(
            (train["run_total_before"] == train["run_top_before"] + train["run_bot_before"]).mean()
        ),
        "runner_count_equals_flags_ratio": float((train["num_runners_on"] == runners_sum).mean()),
        "home_plus_away_expectancy": {
            "min": float((train["home_win_expectancy"] + train["away_win_expectancy"]).min()),
            "max": float((train["home_win_expectancy"] + train["away_win_expectancy"]).max()),
            "equals_100_ratio": float(
                np.isclose(
                    train["home_win_expectancy"] + train["away_win_expectancy"], 100.0
                ).mean()
            ),
        },
        "pitchmix_rate_sum": {
            "nonmissing_rows": int(pitchmix_sum.notna().sum()),
            "min": _python_value(pitchmix_sum.min()),
            "max": _python_value(pitchmix_sum.max()),
            "equals_1_ratio_among_nonmissing": float(
                np.isclose(pitchmix_sum.dropna(), 1.0, atol=1e-5).mean()
            ),
        },
        "prev3_prev5_success_correlation": float(
            train[
                [
                    "asof_pitcher_prev3_game_success_rate",
                    "asof_pitcher_prev5_game_success_rate",
                ]
            ].corr().iloc[0, 1]
        ),
        "prev3_prev5_middle_correlation": float(
            train[
                [
                    "asof_pitcher_prev3_game_middle_rate",
                    "asof_pitcher_prev5_game_middle_rate",
                ]
            ].corr().iloc[0, 1]
        ),
        "numeric_pairs_abs_corr_ge_0_995": high_corr_pairs,
    }

    audit = {
        "train": _frame_schema(train),
        "test": _frame_schema(test),
        "sample_submission": _frame_schema(sample),
        "input_column_contract": {
            "train_input_equals_test_order": input_train_cols == test_cols,
            "missing_from_test": sorted(set(input_train_cols) - set(test_cols)),
            "extra_in_test": sorted(set(test_cols) - set(input_train_cols)),
        },
        "duplicate_row_id": {
            "train": int(train[ID_COL].duplicated().sum()),
            "test": int(test[ID_COL].duplicated().sum()),
            "sample_submission": int(sample[ID_COL].duplicated().sum()),
        },
        "sample_id_contract": {
            "test_equals_submission_set": set(test[ID_COL]) == set(sample[ID_COL]),
            "test_equals_submission_order": test[ID_COL].tolist() == sample[ID_COL].tolist(),
        },
        "target": {
            "counts": {str(k): int(v) for k, v in train[TARGET_COL].value_counts().sort_index().items()},
            "mean": float(train[TARGET_COL].mean()),
        },
        "seasons": {
            str(season): {
                "rows": int(len(group)),
                "target_mean": float(group[TARGET_COL].mean()),
            }
            for season, group in train.groupby("season", sort=True)
        },
        "designated_categorical": {
            col: {
                "storage_dtype": str(train[col].dtype),
                "train_nunique": int(train[col].nunique(dropna=True)),
                "test_nunique": int(test[col].nunique(dropna=True)),
            }
            for col in DESIGNATED_CATEGORICAL
        },
        "entity_cardinality": {
            col: int(train[col].nunique(dropna=True))
            for col in ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"]
        },
        "new_2024": {
            "pitcher": _new_entity_stats(train, "pitcher_id"),
            "batter": _new_entity_stats(train, "batter_id"),
        },
        "duplicate_relations": relations,
        "trackman": {
            "rows": int(len(track)),
            "season_min": int(track["season"].min()),
            "season_max": int(track["season"].max()),
            "main_train_pitcher_intersection": len(train_ids & track_pitchers),
            "main_test_pitcher_intersection": len(test_pitchers & track_pitchers),
            "main_train_batter_intersection": len(train_batters & track_batters),
            "main_test_batter_intersection": len(test_batters & track_batters),
            "mapping_decision": "exclude: anonymous main IDs do not directly map to Trackman IDs",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("solution/output/data_audit.json"))
    args = parser.parse_args()
    audit = run_audit(args.data_dir, args.output)
    summary = {
        "train_shape": audit["train"]["shape"],
        "test_shape": audit["test"]["shape"],
        "target": audit["target"],
        "seasons": audit["seasons"],
        "new_2024": audit["new_2024"],
        "duplicate_relations": audit["duplicate_relations"],
        "trackman": audit["trackman"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
