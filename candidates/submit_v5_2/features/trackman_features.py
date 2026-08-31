from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT if (ROOT / "5-1" / "train.py").is_file() else ROOT.parent
DEFAULT_DATA_DIR = ROOT if (ROOT / "trackman_history.csv").is_file() else PROJECT_ROOT / "data"
DEFAULT_STRATEGY = "mapping_all_shrink"

TM_METADATA_STABLE = [
    "tm_has_mapping",
    "tm_has_pitcher_history",
    "tm_is_low_history",
    "tm_is_rookie_or_no_history",
    "tm_hist_pitch_count",
    "tm_hist_game_count",
    "tm_hist_season_count",
    "tm_mapping_exact_game_count",
    "tm_mapping_margin",
    "tm_mapping_ratio",
    "tm_mapping_n_candidates",
    "tm_mapping_confidence",
    "tm_mapping_confidence_bucket",
    "tm_mapping_is_high_confidence",
    "tm_mapping_selected",
]
TM_PHYSICAL_STABLE = [
    "tm_hist_release_speed_mean_shrunk",
    "tm_hist_release_speed_std_shrunk",
    "tm_hist_spin_rate_mean_shrunk",
    "tm_hist_spin_rate_std_shrunk",
    "tm_hist_release_pos_x_mean_shrunk",
    "tm_hist_release_pos_x_std_shrunk",
    "tm_hist_release_pos_z_mean_shrunk",
    "tm_hist_release_pos_z_std_shrunk",
    "tm_hist_extension_mean_shrunk",
    "tm_hist_extension_std_shrunk",
]
TM_PITCH_STABLE = [
    "tm_hist_pitch_group_fastball_shrunk",
    "tm_hist_pitch_group_breaking_shrunk",
    "tm_hist_pitch_group_offspeed_shrunk",
    "tm_hist_pitch_type_entropy",
]

FEATURE_SETS = {
    "metadata_only": TM_METADATA_STABLE,
    "metadata_pitchmix": TM_METADATA_STABLE + TM_PITCH_STABLE,
    "metadata_physical_core": TM_METADATA_STABLE + TM_PHYSICAL_STABLE,
    "metadata_physical_pitchmix": TM_METADATA_STABLE + TM_PHYSICAL_STABLE + TM_PITCH_STABLE,
    "5-2_server_957": TM_METADATA_STABLE + TM_PHYSICAL_STABLE + TM_PITCH_STABLE,
}


def _load_task51_module():
    path = PROJECT_ROOT / "5-1" / "train.py"
    spec = importlib.util.spec_from_file_location("task51_trackman_train", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


_task51 = _load_task51_module()

TM_FEATURE_COLUMNS = list(_task51.TM_FEATURE_COLUMNS)


def build_trackman_state(train: pd.DataFrame, data_dir: str | Path | None = None) -> dict[str, Any]:
    """Build prior-season Trackman summary state from official train rows.

    The state contains mapping, prior-season pitcher summaries, and hand-based
    priors. It does not use target values and does not join current-pitch
    Trackman rows.
    """
    if data_dir is not None:
        _task51.DATA_DIR = Path(data_dir)
    else:
        _task51.DATA_DIR = DEFAULT_DATA_DIR
    return _task51.build_trackman_state(train)


def build_trackman_features(
    frame: pd.DataFrame,
    state: dict[str, Any],
    strategy: str = DEFAULT_STRATEGY,
    feature_set: str = "5-2_server_957",
) -> pd.DataFrame:
    """Return selected Trackman features aligned to `frame` row order."""
    if feature_set == "all":
        selected = TM_FEATURE_COLUMNS
    elif feature_set in FEATURE_SETS:
        selected = FEATURE_SETS[feature_set]
    else:
        raise ValueError(f"unknown trackman feature_set: {feature_set}")
    features = _task51.build_trackman_features(frame, state, strategy)
    features.index = frame.index
    return features.loc[:, selected].copy()


def append_trackman_features(
    base_features: pd.DataFrame,
    raw_rows: pd.DataFrame,
    state: dict[str, Any],
    strategy: str = DEFAULT_STRATEGY,
    feature_set: str = "5-2_server_957",
) -> pd.DataFrame:
    """Append Trackman features to an existing model feature matrix."""
    tm_features = build_trackman_features(raw_rows, state, strategy=strategy, feature_set=feature_set)
    tm_features.index = base_features.index
    return pd.concat([base_features, tm_features], axis=1)


def categorical_columns(feature_columns: list[str] | None = None) -> list[str]:
    """Trackman categorical columns that should be passed as categorical."""
    cols = ["tm_mapping_confidence_bucket"]
    if feature_columns is None:
        return cols
    selected = set(feature_columns)
    return [col for col in cols if col in selected]
