import pandas as pd
import numpy as np
import feature_engineering as fe

# 1. 데이터 로드 및 전처리
df = pd.read_csv('../../data/train.csv')

if hasattr(fe, 'preprocess_data'):
    df = fe.preprocess_data(df)

# 2. season 컬럼 기준으로 데이터 분리 (2024년 이전 vs 2024년)
tr = df[df['season'] < 2024].copy()
val = df[df['season'] == 2024].copy()

print(f"[데이터 수] 학습(2019-2023): {len(tr):,}건 | 검증(2024): {len(val):,}건")

# 3. 모델 학습 및 2024년 Brier Score 검증
import validation

if hasattr(validation, 'train_and_eval_single_fold'):
    res = validation.train_and_eval_single_fold(tr, val)
    score = res['brier_score'] if isinstance(res, dict) else res
    print("\n" + "="*50)
    print(f"🎯 2024년 검증 단독 Brier Score: {score:.8f}")
    print("="*50)
elif hasattr(validation, 'run_validation'):
    res = validation.run_validation(tr, val)
    print(f"\n🎯 2024년 검증 결과: {res}")
else:
    print("\n[알림] validation.py 모듈을 읽었으나 함수 매칭이 필요합니다.")
