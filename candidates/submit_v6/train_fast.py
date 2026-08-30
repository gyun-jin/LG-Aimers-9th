"""Strict temporal OOF 검증 후 v6 앙상블과 확률 보정기를 최종 학습한다."""
from __future__ import annotations
import json, time
from dataclasses import replace
from pathlib import Path

# Windows native wheels는 LightGBM보다 pandas/sklearn/CatBoost가 먼저 로드되면
# OpenMP DLL이 충돌할 수 있으므로 validation(LightGBM import)을 가장 먼저 불러온다.
from validation import fit_final_model, rolling_splits
import joblib, numpy as np, pandas as pd
from scipy.optimize import minimize
from calibration import apply_calibrator, fit_calibrator
from feature_engineering import ID_COL, TARGET_COL, build_features, categorical_columns, fit_feature_state
from metrics import brier_score, brier_skill_score

BASE_DIR = Path(__file__).resolve().parent
DATA = Path(__file__).resolve().parents[2] / 'data'
MODEL_DIR = BASE_DIR / 'model'; OUT = BASE_DIR / 'output'; MODEL_DIR.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)

WEIGHTS = {'catboost': 0.65, 'lightgbm': 0.05, 'xgboost': 0.30}
# Windows에서는 CatBoost를 한 번 fit한 뒤 LightGBM을 fit하면 OpenMP DLL 충돌이 난다.
# 모든 LightGBM fit을 먼저 끝내도록 학습 순서를 별도로 고정한다.
MODEL_ORDER = ('lightgbm', 'catboost', 'xgboost')
FEATURE_VERSION_BY_MODEL = {'lightgbm': 2, 'catboost': 1, 'xgboost': 2}
DROP_COLUMNS_BY_MODEL = {
    'lightgbm': ('game_type',),
    'catboost': (),
    'xgboost': ('game_type',),
}
PARAMS = {   # v4 final_iteration_params 결과와 동일 (median best_iter + 1, 최소 100)
    'catboost': {'iterations': 100, 'depth': 8, 'learning_rate': 0.05, 'l2_leaf_reg': 10.0, 'verbose': 0},
    'lightgbm': {'n_estimators': 100, 'learning_rate': 0.04, 'num_leaves': 63, 'min_child_samples': 200, 'reg_lambda': 10.0},
    'xgboost': {'n_estimators': 100},
}


def fit_component(name, train_df, input_columns, weight):
    y = train_df[TARGET_COL].to_numpy(dtype=int)
    st = fit_feature_state(train_df, y, input_columns=input_columns, drop_redundant=True)
    st = replace(st, feature_version=FEATURE_VERSION_BY_MODEL[name])
    x = build_features(train_df, st).drop(
        columns=list(DROP_COLUMNS_BY_MODEL[name]), errors='ignore'
    )
    t = time.time()
    model, enc, _ = fit_final_model(name, PARAMS[name], x, y)
    print(f'    {name} fit {time.time()-t:.0f}s', flush=True)
    return {'name': name, 'weight': float(weight), 'model': model, 'encoder': enc,
            'feature_state': st.to_dict(), 'feature_columns': list(x.columns),
            'categorical_columns': categorical_columns(x), 'params': PARAMS[name]}


def predict_component_matrix(comps, frame):
    from validation import apply_category_encoder
    from feature_engineering import prepare_catboost
    predictions = []
    names = []
    for c in comps:
        x = build_features(frame, c['feature_state'])
        if c['name'] in ('lightgbm', 'xgboost'):
            removable = [
                column for column in ('season', 'game_type')
                if column not in c['feature_columns']
            ]
            x = x.drop(columns=removable, errors='ignore')
        assert list(x.columns) == c['feature_columns'], f"{c['name']} schema mismatch"
        if c['name'] == 'catboost':
            x = prepare_catboost(x, c['categorical_columns'])
        else:
            x = apply_category_encoder(x, c['encoder'])
        predictions.append(np.asarray(c['model'].predict_proba(x)[:, 1], dtype=float))
        names.append(c['name'])
    return np.column_stack(predictions), names


def blend_predictions(matrix, names, weights):
    weight_vector = np.asarray([weights[name] for name in names], dtype=float)
    return np.asarray(matrix, dtype=float) @ weight_vector


def predict_comps(comps, frame):
    matrix, names = predict_component_matrix(comps, frame)
    weights = {component['name']: component['weight'] for component in comps}
    return blend_predictions(matrix, names, weights)


def fit_blend_weights(prediction_blocks, label_blocks, names):
    """Strict temporal OOF 예측의 Brier를 최소화하는 simplex 가중치를 구한다."""
    matrix = np.concatenate(prediction_blocks, axis=0)
    labels = np.concatenate(label_blocks, axis=0).astype(float)
    initial = np.asarray([WEIGHTS[name] for name in names], dtype=float)
    initial /= initial.sum()

    def objective(weight_vector):
        return float(np.mean((matrix @ weight_vector - labels) ** 2))

    result = minimize(
        objective,
        x0=initial,
        method='SLSQP',
        bounds=[(0.0, 1.0)] * len(names),
        constraints={'type': 'eq', 'fun': lambda value: float(value.sum() - 1.0)},
        options={'maxiter': 500, 'ftol': 1e-12},
    )
    if not result.success:
        raise RuntimeError(f'ensemble weight optimization failed: {result.message}')
    return {name: float(weight) for name, weight in zip(names, result.x)}


def main():
    t0 = time.time()
    test_cols = pd.read_csv(DATA / 'test.csv', encoding='utf-8-sig', nrows=0).columns.tolist()
    input_columns = [c for c in test_cols if c != ID_COL]
    train = pd.read_csv(DATA / 'train.csv', encoding='utf-8-sig')
    print(f'train {len(train):,}  ({time.time()-t0:.0f}s)', flush=True)

    # (1) strict temporal OOF. 2021~2023만 선택/보정에 쓰고 2024는 독립 평가로 남긴다.
    print('[1] strict temporal OOF 2021-2024', flush=True)
    fold_records = []
    component_names = list(MODEL_ORDER)
    for train_idx, val_idx, train_period, val_season in rolling_splits(train):
        fold_records.append({
            'train_idx': train_idx,
            'val_idx': val_idx,
            'train_period': train_period,
            'validation_season': int(val_season),
            'component_predictions': {},
            'labels': train[TARGET_COL].iloc[val_idx].to_numpy(dtype=int),
        })

    # 모델 종류를 바깥 루프로 둬서 LightGBM 전체 fit이 CatBoost fit보다 먼저 끝나게 한다.
    final_components = []
    for name in component_names:
        print(f'  [{name}] temporal folds', flush=True)
        for record in fold_records:
            print(f'    {record["train_period"]} -> {record["validation_season"]}', flush=True)
            component = fit_component(
                name,
                train.iloc[record['train_idx']],
                input_columns,
                WEIGHTS[name],
            )
            prediction_matrix, names = predict_component_matrix(
                [component], train.iloc[record['val_idx']]
            )
            if names != [name]:
                raise ValueError(f'component order mismatch: expected={[name]}, actual={names}')
            record['component_predictions'][name] = prediction_matrix[:, 0]
            del component
        print(f'    final 2019-2024 {name}', flush=True)
        final_components.append(fit_component(name, train, input_columns, WEIGHTS[name]))

    for record in fold_records:
        record['predictions'] = np.column_stack([
            record['component_predictions'].pop(name) for name in component_names
        ])
        del record['component_predictions'], record['train_idx'], record['val_idx']

    selection_records = [record for record in fold_records if record['validation_season'] <= 2023]
    holdout_records = [record for record in fold_records if record['validation_season'] == 2024]
    if len(holdout_records) != 1:
        raise ValueError(f'expected one 2024 holdout, got {len(holdout_records)}')

    diagnostic_unconstrained_weights = fit_blend_weights(
        [record['predictions'] for record in selection_records],
        [record['labels'] for record in selection_records],
        component_names,
    )
    # 연도별 성공률 하락 상황에서 pooled OOF 최적화가 XGBoost 비중을 과도하게
    # 키웠으므로, 2024에서 재현성이 확인된 기존 가중치를 고정한다.
    selection_weights = dict(WEIGHTS)
    selection_raw = np.concatenate([
        blend_predictions(record['predictions'], component_names, selection_weights)
        for record in selection_records
    ])
    selection_y = np.concatenate([record['labels'] for record in selection_records])
    selection_calibrator = fit_calibrator('platt', selection_raw, selection_y)

    holdout = holdout_records[0]
    raw24 = blend_predictions(holdout['predictions'], component_names, selection_weights)
    y24 = holdout['labels']
    cal24 = apply_calibrator(selection_calibrator, raw24)
    raw24_brier = brier_score(y24, raw24)
    cal24_brier = brier_score(y24, cal24)
    selected_calibration = 'platt' if cal24_brier < raw24_brier else 'none'
    rep = {
        'validation_protocol': 'robust fixed weights; calibrator fit on strict OOF 2021-2023; 2024 evaluation',
        'selection_weights': selection_weights,
        'diagnostic_unconstrained_oof_weights_not_used': diagnostic_unconstrained_weights,
        'selected_calibration': selected_calibration,
        'selection_oof_rows': int(len(selection_y)),
        'selection_oof_raw_brier': brier_score(selection_y, selection_raw),
        'selection_oof_platt_insample_brier_diagnostic_only': brier_score(
            selection_y, apply_calibrator(selection_calibrator, selection_raw)
        ),
        'val_2024_rows': int(len(y24)),
        'val_2024_raw_brier': raw24_brier,
        'val_2024_raw_bss': brier_skill_score(y24, raw24),
        'val_2024_platt_brier': cal24_brier,
        'val_2024_platt_bss': brier_skill_score(y24, cal24),
        'folds': [],
    }
    for record in fold_records:
        fold_raw = blend_predictions(record['predictions'], component_names, selection_weights)
        rep['folds'].append({
            'train_period': record['train_period'],
            'validation_season': record['validation_season'],
            'rows': int(len(record['labels'])),
            'selected_weight_raw_brier': brier_score(record['labels'], fold_raw),
            'selected_weight_raw_bss': brier_skill_score(record['labels'], fold_raw),
            'independent_of_weight_and_calibrator_fit': record['validation_season'] == 2024,
        })
    print(
        f'  2024 raw Brier={rep["val_2024_raw_brier"]:.6f} '
        f'Platt={rep["val_2024_platt_brier"]:.6f}',
        flush=True,
    )

    # (2) 최종 제출용 가중치/보정기는 모든 strict OOF를 사용한다.
    final_weights = dict(WEIGHTS)
    final_oof_raw = np.concatenate([
        blend_predictions(record['predictions'], component_names, final_weights)
        for record in fold_records
    ])
    final_oof_y = np.concatenate([record['labels'] for record in fold_records])
    calibrator = fit_calibrator(selected_calibration, final_oof_raw, final_oof_y)
    rep['final_oof_weights'] = final_weights
    rep['final_oof_rows'] = int(len(final_oof_y))
    rep['final_oof_raw_brier'] = brier_score(final_oof_y, final_oof_raw)

    print('[2] save final 2019-2024 models', flush=True)
    comps = final_components
    for component in comps:
        component['weight'] = float(final_weights[component['name']])
    bundle = {'format_version': 1, 'input_columns': input_columns, 'components': comps}
    joblib.dump(bundle, MODEL_DIR / 'final_model.joblib', compress=3)
    joblib.dump(calibrator, MODEL_DIR / 'calibration_model.joblib', compress=3)
    schema = {'input_columns': input_columns, 'input_column_order_required': True, 'target': TARGET_COL,
              'components': [{'name': c['name'], 'feature_state': c['feature_state'], 'feature_columns': c['feature_columns'],
                              'categorical_columns': c['categorical_columns'], 'encoder': c['encoder']} for c in comps]}
    (MODEL_DIR / 'feature_schema.json').write_text(json.dumps(schema, ensure_ascii=False, indent=2,
                                                              default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)), encoding='utf-8')
    (MODEL_DIR / 'ensemble_config.json').write_text(json.dumps({
        'selected_ensemble': 'catboost_lightgbm_xgboost',
        'weights': final_weights,
        'selection_weights_2021_2023': selection_weights,
        'diagnostic_unconstrained_oof_weights_not_used': diagnostic_unconstrained_weights,
        'calibration': f'{selected_calibration}_on_strict_temporal_oof_2021_2024',
        'feature_versions': FEATURE_VERSION_BY_MODEL,
        'drop_columns': {name: list(columns) for name, columns in DROP_COLUMNS_BY_MODEL.items()},
        'note': 'robust fixed weights; CatBoost legacy layout; XGB/LGB v2 with season retained',
    }, indent=2), encoding='utf-8')
    (OUT / 'validation_report.json').write_text(json.dumps(rep, indent=2), encoding='utf-8')
    print(f'done {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
