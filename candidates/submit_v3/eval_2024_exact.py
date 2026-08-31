import pandas as pd
import numpy as np
from metrics import brier_score
import feature_engineering as fe
import validation

print("1. 데이터 로드 및 전처리 중...")
df = pd.read_csv('../../data/train.csv')
if hasattr(fe, 'preprocess_data'):
    df = fe.preprocess_data(df)

# 타겟 컬럼 자동 찾기
target_col = 'is_strike' if 'is_strike' in df.columns else 'target'

# 2. 2019-2023 (학습) vs 2024 (검증) 데이터 분리
train_mask = (df['season'] >= 2019) & (df['season'] <= 2023)
val_mask = (df['season'] == 2024)

x_train = df[train_mask].drop(columns=[target_col], errors='ignore')
y_train = df[train_mask][target_col].values

x_val = df[val_mask].drop(columns=[target_col], errors='ignore')
y_val = df[val_mask][target_col].values

print(f"   - 학습 데이터 (2019~2023): {len(x_train):,}건")
print(f"   - 검증 데이터 (2024년):     {len(x_val):,}건\n")

print("2. validation.py의 fit_predict_model 함수로 모델별 학습 진행...")

# 1) CatBoost
print("   [1/3] CatBoost 학습 중...")
_, _, p_cb, _, _, _ = validation.fit_predict_model('catboost', {}, x_train, y_train, x_val, y_val)

# 2) LightGBM
print("   [2/3] LightGBM 학습 중...")
_, _, p_lgb, _, _, _ = validation.fit_predict_model('lightgbm', {}, x_train, y_train, x_val, y_val)

# 3) XGBoost
print("   [3/3] XGBoost 학습 중...")
_, _, p_xgb, _, _, _ = validation.fit_predict_model('xgboost', {}, x_train, y_train, x_val, y_val)

# 앙상블 (가중치: CatBoost 0.65 / XGBoost 0.30 / LightGBM 0.05)
p_ens = (0.65 * p_cb) + (0.30 * p_xgb) + (0.05 * p_lgb)

# 점수 산출
score_cb = brier_score(y_val, p_cb)
score_lgb = brier_score(y_val, p_lgb)
score_xgb = brier_score(y_val, p_xgb)
score_ens = brier_score(y_val, p_ens)

print("\n" + "="*50)
print("📊 [2024년 단독 검증 Brier Score]")
print(f"   • CatBoost Brier Score : {score_cb:.8f}")
print(f"   • LightGBM Brier Score : {score_lgb:.8f}")
print(f"   • XGBoost Brier Score  : {score_xgb:.8f}")
print(f"   🎯 앙상블 최종 Brier Score: {score_ens:.8f}")
print("="*50)
