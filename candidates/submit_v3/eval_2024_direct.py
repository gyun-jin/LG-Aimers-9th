import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import feature_engineering as fe

print("1. 데이터 로드 및 전처리 진행 중...")
df = pd.read_csv('../../data/train.csv')

if hasattr(fe, 'preprocess_data'):
    df = fe.preprocess_data(df)

# 타겟 컬럼 확인
target_col = 'target' if 'target' in df.columns else 'is_strike'

# 학습 / 검증 분리 (2023년까지 학습, 2024년 검증)
tr = df[df['season'] < 2024].copy()
val = df[df['season'] == 2024].copy()

print(f"   - 학습 데이터 (2019~2023): {len(tr):,}건")
print(f"   - 검증 데이터 (2024년):     {len(val):,}건")

# 피처 추출 (비피처 컬럼 제외)
ignore_cols = ['season', 'game_date', 'date', 'target', 'is_strike', 'id', 'pitch_id']
features = [c for c in tr.columns if c not in ignore_cols and not tr[c].dtype == 'object']

X_tr, y_tr = tr[features], tr[target_col]
X_val, y_val = val[features], val[target_col]

print("\n2. 모델 학습 및 2024년 예측 수행 중...")

# 1) CatBoost
cb = CatBoostClassifier(iterations=1000, depth=8, learning_rate=0.05, verbose=0, random_seed=42)
cb.fit(X_tr, y_tr)
p_cb = cb.predict_proba(X_val)[:, 1]

# 2) LightGBM
lgb = LGBMClassifier(n_estimators=1000, learning_rate=0.04, num_leaves=63, random_state=42, verbose=-1)
lgb.fit(X_tr, y_tr)
p_lgb = lgb.predict_proba(X_val)[:, 1]

# 3) XGBoost
xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, eval_metric='logloss')
xgb.fit(X_tr, y_tr)
p_xgb = xgb.predict_proba(X_val)[:, 1]

# 3. 앙상블 (이전 최적 가중치 적용: CatBoost 0.65 / XGBoost 0.30 / LightGBM 0.05)
p_ensemble = (p_cb * 0.65) + (p_xgb * 0.30) + (p_lgb * 0.05)

# 4. Brier Score 계산
score_cb = brier_score_loss(y_val, p_cb)
score_lgb = brier_score_loss(y_val, p_lgb)
score_xgb = brier_score_loss(y_val, p_xgb)
score_ens = brier_score_loss(y_val, p_ensemble)

print("\n" + "="*50)
print(f"📊 [2024년 단독 검증 Brier Score]")
print(f"   • CatBoost Brier Score : {score_cb:.6f}")
print(f"   • LightGBM Brier Score : {score_lgb:.6f}")
print(f"   • XGBoost Brier Score  : {score_xgb:.6f}")
print(f"   🎯 앙상블 최종 Brier Score: {score_ens:.6f}")
print("="*50)
