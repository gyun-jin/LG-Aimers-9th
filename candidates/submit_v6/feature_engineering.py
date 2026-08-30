"""누수 없는 fold-fit prior와 행 단위 피처 엔지니어링."""

# [추가 구현]
# 목적: 학습·검증·추론에서 같은 입력 스키마와 같은 행 단위 피처를 생성
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"

BASE_CATEGORICAL = [
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
DERIVED_CATEGORICAL = [
    "hand_matchup", "count_state", "runner_out_state", "pressure_state",
    "runners_count_state", "runners_out_count_state", "outs_count_state",
    "high_li_count", "pressure_count", "late_close_count", 
    "risp_count_state", "loaded_count_state"
]

RATE_COUNT_MAP = {
    "asof_pitcher_success_rate": "asof_pitcher_n",
    "asof_pitcher_reverse_rate": "asof_pitcher_n",
    "asof_pitcher_middle_rate": "asof_pitcher_n",
    "asof_pitcher_ball_rate": "asof_pitcher_n",
    "asof_pitcher_strike_rate": "asof_pitcher_n",
    "asof_batter_success_rate": "asof_batter_n",
    "asof_batter_middle_rate": "asof_batter_n",
    "asof_pitcher_fastball_rate": "asof_pitcher_pitchmix_n",
    "asof_pitcher_breaking_rate": "asof_pitcher_pitchmix_n",
    "asof_pitcher_offspeed_rate": "asof_pitcher_pitchmix_n",
}

REDUNDANT_COLUMNS = [
    "asof_pitcher_pitchmix_n",  # asof_pitcher_n과 완전 동일
    "run_total_before",  # run_top_before + run_bot_before와 완전 동일
    "num_runners_on",  # 세 주자 플래그 합과 완전 동일
    "away_win_expectancy",  # home_win_expectancy와 거의 완전한 보수 관계
    "asof_pitcher_offspeed_rate",  # 세 구종군 비율의 합이 1
]


@dataclass(frozen=True)
class FeatureState:
    input_columns: list[str]
    alpha: float
    target_prior: float
    rate_priors: dict[str, float]
    missing_columns: list[str]
    drop_redundant: bool
    feature_version: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FeatureState":
        # 기존 submit_v6 모델에는 feature_version이 없다. 그 모델은 저장 당시의
        # 131개 피처를 그대로 재현하고, 새로 fit한 state부터 정리된 v2 피처를 쓴다.
        normalized = dict(value)
        normalized.setdefault("feature_version", 1)
        return cls(**normalized)


def fit_feature_state(
    frame: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    input_columns: list[str] | None = None,
    alpha: float = 50.0,
    drop_redundant: bool = True,
) -> FeatureState:
    """각 학습 fold에서만 smoothing prior와 결측 schema를 fit한다."""
    if input_columns is None:
        input_columns = [c for c in frame.columns if c not in {ID_COL, TARGET_COL}]
    missing = sorted(c for c in input_columns if frame[c].isna().any())
    target_prior = float(np.mean(np.asarray(y, dtype=float)))
    rate_priors: dict[str, float] = {}
    for rate_col, count_col in RATE_COUNT_MAP.items():
        rate = pd.to_numeric(frame[rate_col], errors="coerce").to_numpy(dtype=float)
        count = pd.to_numeric(frame[count_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(rate) & np.isfinite(count) & (count > 0)
        if rate_col in {"asof_pitcher_success_rate", "asof_batter_success_rate"}:
            prior = target_prior
        elif valid.any() and count[valid].sum() > 0:
            prior = float(np.sum(rate[valid] * count[valid]) / np.sum(count[valid]))
        else:
            prior = float(np.nanmean(rate)) if np.isfinite(rate).any() else 0.0
        rate_priors[rate_col] = prior
    return FeatureState(
        input_columns=list(input_columns),
        alpha=float(alpha),
        target_prior=target_prior,
        rate_priors=rate_priors,
        missing_columns=missing,
        drop_redundant=bool(drop_redundant),
        feature_version=2,
    )


def _safe_string(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("__MISSING__").astype(str)


def build_features(frame: pd.DataFrame, state: FeatureState | dict[str, Any]) -> pd.DataFrame:
    """test 내부 집계 없이 한 행과 fold-fit 상수만으로 피처를 만든다."""
    if isinstance(state, dict):
        state = FeatureState.from_dict(state)
    # v3는 CatBoost가 잘 활용하던 legacy 131-column layout을 유지하면서
    # v6 pitchmix 상호작용의 결측 처리만 smoothed prior 방식으로 교체한다.
    legacy_layout = state.feature_version in (1, 3)
    smoothed_pitchmix = state.feature_version >= 2
    missing = [c for c in state.input_columns if c not in frame.columns]
    extra = [c for c in frame.columns if c not in state.input_columns + [ID_COL, TARGET_COL]]
    if missing or extra:
        raise ValueError(f"입력 schema 불일치: missing={missing}, extra={extra}")
    x = frame.loc[:, state.input_columns].copy()

    # 행 하나의 투구 직전 상황만 사용한다.
    x["hand_matchup"] = _safe_string(x["pitcher_hand"]) + "_" + _safe_string(x["batter_hand"])
    x["count_state"] = _safe_string(x["balls_before"]) + "-" + _safe_string(x["strikes_before"])
    x["runner_out_state"] = _safe_string(x["base_state"]) + "_o" + _safe_string(x["outs_before"])
    x["close_game"] = (x["score_diff_pitcher_team"].abs() <= 2).astype("int8")
    x["late_inning"] = (x["inning"] >= 7).astype("int8")
    high_li = (x["li"] >= 1.5).astype("int8")
    x["pressure_state"] = (
        "late" + x["late_inning"].astype(str)
        + "_close" + x["close_game"].astype(str)
        + "_li" + high_li.astype(str)
    )
    x["log1p_asof_pitcher_n"] = np.log1p(x["asof_pitcher_n"].clip(lower=0))
    x["log1p_asof_batter_n"] = np.log1p(x["asof_batter_n"].clip(lower=0))

    for rate_col, count_col in RATE_COUNT_MAP.items():
        count = pd.to_numeric(x[count_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        rate = pd.to_numeric(x[rate_col], errors="coerce")
        prior = state.rate_priors[rate_col]
        observed = rate.fillna(prior)
        x[f"{rate_col}_smoothed"] = (
            count * observed + state.alpha * prior
        ) / (count + state.alpha)

    recent_success_cols = [
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    ]
    recent_middle_cols = [
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
    ]
    success = x[recent_success_cols]
    middle = x[recent_middle_cols]
    x["recent_success_mean_1_3_5"] = success.mean(axis=1, skipna=True)
    x["recent_success_range_1_3_5"] = success.max(axis=1, skipna=True) - success.min(axis=1, skipna=True)
    x["recent_success_std_1_3_5"] = success.std(axis=1, skipna=True, ddof=0)
    for window, col in zip((1, 3, 5), recent_success_cols):
        x[f"recent_gap_{window}"] = x[col] - x["asof_pitcher_success_rate_smoothed"]
    x["recent_middle_mean_1_3_5"] = middle.mean(axis=1, skipna=True)
    x["recent_middle_range_1_3_5"] = middle.max(axis=1, skipna=True) - middle.min(axis=1, skipna=True)
    x["recent_middle_vs_cumulative"] = (
        x["recent_middle_mean_1_3_5"] - x["asof_pitcher_middle_rate_smoothed"]
    )

    if legacy_layout:
        for col in state.missing_columns:
            x[f"{col}__missing"] = x[col].isna().astype("int8")
        x["is_pitcher_cold_start"] = (x["asof_pitcher_n"].fillna(0) <= 0).astype("int8")
        x["is_batter_cold_start"] = (x["asof_batter_n"].fillna(0) <= 0).astype("int8")
    else:
        # 원본 데이터에서 같은 시점에 함께 비는 rate들의 중복 indicator를 하나로 축약한다.
        x["pitcher_rate_missing"] = x[
            [
                "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
                "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
            ]
        ].isna().any(axis=1).astype("int8")
        x["batter_rate_missing"] = x[
            ["asof_batter_success_rate", "asof_batter_middle_rate"]
        ].isna().any(axis=1).astype("int8")
        x["recent_form_missing"] = x[recent_success_cols + recent_middle_cols].isna().any(axis=1).astype("int8")
        x["pitchmix_available"] = x[
            [
                "asof_pitcher_fastball_rate",
                "asof_pitcher_breaking_rate",
                "asof_pitcher_offspeed_rate",
            ]
        ].notna().all(axis=1).astype("int8")

        pitcher_count = pd.to_numeric(x["asof_pitcher_n"], errors="coerce").fillna(0.0).clip(lower=0.0)
        batter_count = pd.to_numeric(x["asof_batter_n"], errors="coerce").fillna(0.0).clip(lower=0.0)
        pitchmix_count = pd.to_numeric(x["asof_pitcher_pitchmix_n"], errors="coerce").fillna(0.0).clip(lower=0.0)
        x["pitcher_rate_reliability"] = pitcher_count / (pitcher_count + state.alpha)
        x["batter_rate_reliability"] = batter_count / (batter_count + state.alpha)

        def posterior_sd(smoothed_col: str, count: pd.Series) -> pd.Series:
            probability = x[smoothed_col].clip(lower=0.0, upper=1.0)
            return np.sqrt(probability * (1.0 - probability) / (count + state.alpha + 1.0))

        x["pitcher_success_posterior_sd"] = posterior_sd(
            "asof_pitcher_success_rate_smoothed", pitcher_count
        )
        x["batter_success_posterior_sd"] = posterior_sd(
            "asof_batter_success_rate_smoothed", batter_count
        )
        for pitch_name in ("fastball", "breaking", "offspeed"):
            x[f"pitchmix_{pitch_name}_posterior_sd"] = posterior_sd(
                f"asof_pitcher_{pitch_name}_rate_smoothed", pitchmix_count
            )
    mix_cols = [
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ]
    mix = x[mix_cols].clip(lower=0.0, upper=1.0)
    x["pitchmix_entropy"] = -(mix * np.log(mix.clip(lower=1e-12))).sum(axis=1, min_count=1)
    x["score_abs"] = x["score_diff_pitcher_team"].abs()
    x["is_pitcher_team_leading"] = (x["score_diff_pitcher_team"] > 0).astype("int8")
    x["is_pitcher_team_trailing"] = (x["score_diff_pitcher_team"] < 0).astype("int8")

    # 1. Count features
    if legacy_layout:
        x['count_advantage'] = x['strikes_before'] - x['balls_before']
    x['count_pressure'] = x['balls_before'] - x['strikes_before']
    x['is_hitter_count'] = (x['balls_before'] > x['strikes_before']).astype("int8")
    x['is_pitcher_count'] = (x['strikes_before'] > x['balls_before']).astype("int8")
    x['is_even_count'] = (x['balls_before'] == x['strikes_before']).astype("int8")
    x['is_three_ball'] = (x['balls_before'] == 3).astype("int8")
    x['is_two_strike'] = (x['strikes_before'] == 2).astype("int8")
    x['is_full_count'] = ((x['balls_before'] == 3) & (x['strikes_before'] == 2)).astype("int8")
    # x['count_state'] = ... (already done)

    # 2. Runner features
    x['risp'] = ((x['runner_on_2b'] == 1) | (x['runner_on_3b'] == 1)).astype("int8")
    x['loaded_base'] = (x['base_state'] == "123").astype("int8")
    x['any_runner'] = (x['num_runners_on'] > 0).astype("int8")
    x['runner_pressure'] = x['num_runners_on'] * x['li']
    x['scoring_runner_pressure'] = x['risp'] * x['li']
    x['loaded_base_pressure'] = x['loaded_base'] * x['li']

    # 3. Out/Runner interaction
    # runner_out_state already defined
    x['runners_count_state'] = _safe_string(x['base_state']) + "_" + x['count_state']
    x['runners_out_count_state'] = x['runner_out_state'] + "_" + x['count_state']
    x['outs_count_state'] = _safe_string(x['outs_before']) + "_" + x['count_state']

    # 4. Inning/Score
    # score_abs, close_game, late_inning, high_li, pressure_state already defined above
    x['tie_game'] = (x['score_diff_pitcher_team'] == 0).astype("int8")
    if legacy_layout:
        x['pitcher_leading'] = (x['score_diff_pitcher_team'] > 0).astype("int8")
        x['pitcher_trailing'] = (x['score_diff_pitcher_team'] < 0).astype("int8")
    x['extra_inning'] = (x['inning'] >= 10).astype("int8")
    x['very_high_li'] = (x['li'] >= 2.0).astype("int8")
    x['late_close'] = (x['late_inning'] & x['close_game']).astype("int8")
    
    # 5. Interactions
    x['high_li_count'] = _safe_string(high_li) + "_" + x['count_state']
    x['pressure_count'] = x['pressure_state'] + "_" + x['count_state']
    x['late_runner_pressure'] = x['late_inning'] * x['num_runners_on'] * x['li']
    x['close_count_pressure'] = x['close_game'] * x['count_pressure']
    x['late_close_count'] = _safe_string(x['late_close']) + "_" + x['count_state']
    x['risp_count_state'] = _safe_string(x['risp']) + "_" + x['count_state']
    x['loaded_count_state'] = _safe_string(x['loaded_base']) + "_" + x['count_state']
    # pitcher_hand_batter_hand (already defined as hand_matchup)

    # 6. v4 추가: 제구 실패 정의(가운데/반대/볼) 직결 상호작용. 모두 smoothed 값 사용.
    v4_pm = x['asof_pitcher_middle_rate_smoothed']
    v4_pr = x['asof_pitcher_reverse_rate_smoothed']
    v4_pb = x['asof_pitcher_ball_rate_smoothed']
    x['v4_fail_prone'] = v4_pm + v4_pr + v4_pb
    x['v4_middle_matchup'] = v4_pm * x['asof_batter_middle_rate_smoothed']
    x['v4_succ_gap'] = x['asof_pitcher_success_rate_smoothed'] - x['asof_batter_success_rate_smoothed']
    x['v4_reverse_x_2strike'] = v4_pr * x['is_two_strike']
    x['v4_ball_x_3ball'] = v4_pb * x['is_three_ball']
    x['v4_middle_x_risp'] = v4_pm * x['risp']

    # 7. v6 추가: 구종 운영 x 상황 (조선미 2023, 상황별 구종 예측). 한 행 내 계산.
    if not smoothed_pitchmix:
        # 저장된 v6 모델의 schema/예측을 보존한다.
        v6_off = x['asof_pitcher_offspeed_rate'].fillna(0.0)
        v6_brk = x['asof_pitcher_breaking_rate'].fillna(0.0)
    else:
        # 신규/콜드스타트 투수를 구종 미사용으로 오해하지 않도록 fold-fit prior로 평활한다.
        v6_off = x['asof_pitcher_offspeed_rate_smoothed']
        v6_brk = x['asof_pitcher_breaking_rate_smoothed']
    x['v6_offspeed_x_2strike'] = v6_off * x['is_two_strike']
    x['v6_breaking_x_2strike'] = v6_brk * x['is_two_strike']
    x['v6_nonfastball_x_risp'] = (v6_off + v6_brk) * x['risp']
    x['v6_offspeed_x_outs'] = v6_off * x['outs_before']

    if state.drop_redundant:
        x = x.drop(columns=[c for c in REDUNDANT_COLUMNS if c in x.columns])
    return x


def categorical_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in BASE_CATEGORICAL + DERIVED_CATEGORICAL if c in frame.columns]


def prepare_catboost(frame: pd.DataFrame, cat_columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for col in cat_columns:
        result[col] = _safe_string(result[col])
    return result
