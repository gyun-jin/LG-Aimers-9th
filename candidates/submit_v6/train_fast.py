"""v6 빠른 최종 학습: v4가 확정한 앙상블 구성/파라미터를 그대로 쓰고
rolling 검증은 생략. (1) 2019~2023 학습 -> 2024 예측으로 Platt 보정기 fit + 검증 리포트
(2) 2019~2024 전체 재학습 -> model/ 저장. v4 train.py의 최종 단계와 동일한 산출물 형식."""
from __future__ import annotations
import json, time
from pathlib import Path
import joblib, numpy as np, pandas as pd

from calibration import apply_calibrator, fit_calibrator
from feature_engineering import ID_COL, TARGET_COL, build_features, categorical_columns, fit_feature_state
from validation import fit_final_model
from metrics import brier_score, brier_skill_score

DATA = Path(r'C:\Users\김아영\Desktop\open\data')
MODEL_DIR = Path('model'); OUT = Path('output'); MODEL_DIR.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)

WEIGHTS = {'catboost': 0.65, 'lightgbm': 0.05, 'xgboost': 0.30}
PARAMS = {   # v4 final_iteration_params 결과와 동일 (median best_iter + 1, 최소 100)
    'catboost': {'iterations': 100, 'depth': 8, 'learning_rate': 0.05, 'l2_leaf_reg': 10.0, 'verbose': 0},
    'lightgbm': {'n_estimators': 100, 'learning_rate': 0.04, 'num_leaves': 63, 'min_child_samples': 200, 'reg_lambda': 10.0},
    'xgboost': {'n_estimators': 100},
}


def fit_ensemble(train_df, input_columns):
    y = train_df[TARGET_COL].to_numpy(dtype=int)
    comps = []
    for name, w in WEIGHTS.items():
        st = fit_feature_state(train_df, y, input_columns=input_columns, drop_redundant=True)
        x = build_features(train_df, st)
        if name in ('lightgbm', 'xgboost'):
            x = x.drop(columns=['season', 'game_type'], errors='ignore')
        t = time.time()
        model, enc, _ = fit_final_model(name, PARAMS[name], x, y)
        print(f'  {name} fit {time.time()-t:.0f}s', flush=True)
        comps.append({'name': name, 'weight': float(w), 'model': model, 'encoder': enc,
                      'feature_state': st.to_dict(), 'feature_columns': list(x.columns),
                      'categorical_columns': categorical_columns(x), 'params': PARAMS[name]})
    return comps


def predict_comps(comps, frame):
    from validation import apply_category_encoder
    from feature_engineering import prepare_catboost
    p = np.zeros(len(frame))
    for c in comps:
        x = build_features(frame, c['feature_state'])
        if c['name'] in ('lightgbm', 'xgboost'):
            x = x.drop(columns=['season', 'game_type'], errors='ignore')
        assert list(x.columns) == c['feature_columns'], f"{c['name']} schema mismatch"
        if c['name'] == 'catboost':
            x = prepare_catboost(x, c['categorical_columns'])
        else:
            x = apply_category_encoder(x, c['encoder'])
        p += c['weight'] * c['model'].predict_proba(x)[:, 1]
    return p


def main():
    t0 = time.time()
    test_cols = pd.read_csv(DATA / 'test.csv', encoding='utf-8-sig', nrows=0).columns.tolist()
    input_columns = [c for c in test_cols if c != ID_COL]
    train = pd.read_csv(DATA / 'train.csv', encoding='utf-8-sig')
    print(f'train {len(train):,}  ({time.time()-t0:.0f}s)', flush=True)

    # (1) 홀드아웃: 2019~2023 -> 2024
    print('[1] holdout fit 2019-2023', flush=True)
    hold = train[train.season <= 2023]; val = train[train.season == 2024]
    comps_h = fit_ensemble(hold, input_columns)
    raw24 = predict_comps(comps_h, val)
    y24 = val[TARGET_COL].to_numpy(dtype=int)
    calibrator = fit_calibrator('platt', raw24, y24)
    cal24 = apply_calibrator(calibrator, raw24)
    rep = {'val_2024_raw_brier': brier_score(y24, raw24), 'val_2024_raw_bss': brier_skill_score(y24, raw24),
           'val_2024_platt_insample_brier': brier_score(y24, cal24)}
    print(f'  2024 raw Brier={rep["val_2024_raw_brier"]:.6f} BSS={rep["val_2024_raw_bss"]:.1f}', flush=True)
    del comps_h

    # (2) 전체 재학습 2019~2024
    print('[2] final fit 2019-2024', flush=True)
    comps = fit_ensemble(train, input_columns)
    bundle = {'format_version': 1, 'input_columns': input_columns, 'components': comps}
    joblib.dump(bundle, MODEL_DIR / 'final_model.joblib', compress=3)
    joblib.dump(calibrator, MODEL_DIR / 'calibration_model.joblib', compress=3)
    schema = {'input_columns': input_columns, 'input_column_order_required': True, 'target': TARGET_COL,
              'components': [{'name': c['name'], 'feature_state': c['feature_state'], 'feature_columns': c['feature_columns'],
                              'categorical_columns': c['categorical_columns'], 'encoder': c['encoder']} for c in comps]}
    (MODEL_DIR / 'feature_schema.json').write_text(json.dumps(schema, ensure_ascii=False, indent=2,
                                                              default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)), encoding='utf-8')
    (MODEL_DIR / 'ensemble_config.json').write_text(json.dumps({'selected_ensemble': 'catboost_lightgbm_xgboost', 'weights': WEIGHTS,
                                                                'calibration': 'platt', 'note': 'v6 = v4 config + 4 pitchmix x situation features'}, indent=2), encoding='utf-8')
    (OUT / 'validation_report.json').write_text(json.dumps(rep, indent=2), encoding='utf-8')
    print(f'done {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
