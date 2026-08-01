"""
src/pipeline/train.py

Reproduces the final chosen model (XGBoost, with tradeline features,
2007-2015 train / 2016 val / 2017-2018 test) end-to-end from the raw
CSV. Shares preprocessing logic with predict.py via preprocess.py —
this file only handles training-specific steps: loading raw data,
building the target, splitting by year, training, and evaluating.

 To Run from the project root: python -m src.pipeline.train
"""

import os
import gc
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.stats import ks_2samp

from src.pipeline.preprocess import (
    load_artifacts, convert_date_columns, build_target, run_feature_engineering,
    apply_caps, apply_imputation, apply_encoding, select_tree_features,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'src', 'data', 'raw', 'accepted_2007_to_2018Q4.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'src', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Must match the sampling used during feature engineering, so results reproduce
CSV_CHUNK_SIZE = 50_000
SAMPLE_FRAC = 0.75
RANDOM_SEED = 42


# =========================================================
# LOAD RAW DATA
# =========================================================
def load_raw_data():
    """Reads the huge raw CSV in chunks, keeping a 75% random sample of each."""
    print("Loading raw CSV in chunks...")
    sampled_chunks = []
    for i, chunk in enumerate(pd.read_csv(DATA_PATH, chunksize=CSV_CHUNK_SIZE, low_memory=False)):
        sampled_chunks.append(chunk.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED))
        if i % 10 == 0:
            print(f"  processed chunk {i}")
    df = pd.concat(sampled_chunks, ignore_index=True)
    del sampled_chunks
    gc.collect()
    print(f"Raw sampled shape: {df.shape}")
    return df

# =========================================================
# tEMPORAL SPLIT
# =========================================================
def temporal_split(df, artifacts):
    """
    Splits by year, not randomly — train on the past, test on genuinely
    unseen future years. This matches how the model would actually be
    used: scoring new loans it's never seen before.
    """
    cfg = artifacts['fe_config']
    train_df = df[df['issue_year'] <= cfg['train_years_max']].copy()   # 2007-2015
    val_df   = df[df['issue_year'] == cfg['val_year']].copy()          # 2016
    test_df  = df[df['issue_year'] >= cfg['test_years_min']].copy()    # 2017-2018

    X_train, y_train = train_df.drop(columns=['target']), train_df['target']
    X_val, y_val     = val_df.drop(columns=['target']), val_df['target']
    X_test, y_test   = test_df.drop(columns=['target']), test_df['target']

    del df, train_df, val_df, test_df
    gc.collect()
    return X_train, y_train, X_val, y_val, X_test, y_test

# =========================================================
# EVALUATE MODEL
# =========================================================
def evaluate_credit_model(y_true, y_proba, model_name="model", n_deciles=10):
    """Prints ROC-AUC, KS, Brier score, and a decile lift table for one split."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)
    # Biggest gap between defaulters' and non-defaulters' predicted scores
    ks_stat = ks_2samp(y_proba[y_true == 1], y_proba[y_true == 0]).statistic

    # Split predictions into 10 risk buckets (D1 = safest, D10 = riskiest)
    # and check the real default rate in each bucket
    df = pd.DataFrame({'y_true': y_true, 'y_proba': y_proba})
    df['decile'] = pd.qcut(
        df['y_proba'].rank(method='first'), n_deciles,
        labels=[f"D{i}" for i in range(1, n_deciles + 1)]
    )
    lift_table = df.groupby('decile', observed=True).agg(
        n=('y_true', 'size'), default_rate=('y_true', 'mean')
    ).reindex([f"D{i}" for i in range(1, n_deciles + 1)])
    overall_rate = y_true.mean()
    lift_table['lift'] = lift_table['default_rate'] / overall_rate
    top10_default_rate = lift_table.loc['D10', 'default_rate']
    top10_lift = lift_table.loc['D10', 'lift']

    print(f"=== {model_name} ===")
    print(f"ROC-AUC:      {roc_auc:.4f}")
    print(f"PR-AUC:       {pr_auc:.4f}")
    print(f"KS statistic: {ks_stat:.4f}")
    print(f"Brier score:  {brier:.4f}")
    print(f"Top 10% (D10) default rate: {top10_default_rate:.2%}  (lift: {top10_lift:.2f}x)")

    return {
        'roc_auc': roc_auc, 'pr_auc': pr_auc, 'ks_stat': ks_stat, 'brier': brier,
        'top10_default_rate': top10_default_rate, 'top10_lift': top10_lift,
        'lift_table': lift_table,
    }

# =========================================================
# TRAIN MODEL
# =========================================================
def train_model(X_train, y_train, X_val, y_val):
    """
    Trains the final XGBoost model. Shallow trees + slow learning rate
    for better generalization, row/feature subsampling to reduce
    overfitting, scale_pos_weight to handle the class imbalance
    (~79% non-default, ~21% default), and early stopping once
    validation AUC stops improving.
    """
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=1500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',
        eval_metric='auc',
        early_stopping_rounds=50,
        n_jobs=4,
        random_state=42,
    )
    model.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    print(f"Best iteration: {model.best_iteration}")
    return model

# =========================================================
# MAIN Function 
# =========================================================
def main():
    # Load every fitted object needed for preprocessing (nothing gets re-fit)
    artifacts = load_artifacts()

    df = load_raw_data()
    df = convert_date_columns(df)                # turn issue_d into a real date FIRST
    df = build_target(df, artifacts)             # target needs issue_d as a real date
    df = run_feature_engineering(df, artifacts)  # clean data, engineer features, drop issue_d

    #Temporal Split
    X_train, y_train, X_val, y_val, X_test, y_test = temporal_split(df, artifacts)

    # Apply the same caps/encoding/imputation to each split separately
    X_train = apply_caps(X_train, artifacts)
    X_val   = apply_caps(X_val, artifacts)
    X_test  = apply_caps(X_test, artifacts)

    X_train = apply_imputation(X_train, artifacts)
    X_val   = apply_imputation(X_val, artifacts)
    X_test  = apply_imputation(X_test, artifacts)

    X_train = apply_encoding(X_train, artifacts)
    X_val   = apply_encoding(X_val, artifacts)
    X_test  = apply_encoding(X_test, artifacts)

    X_train = select_tree_features(X_train, artifacts)
    X_val   = select_tree_features(X_val, artifacts)
    X_test  = select_tree_features(X_test, artifacts)

    print(f"Final shapes — train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    model = train_model(X_train, y_train, X_val, y_val)

    # Check performance on validation (used for early stopping) and test (final, unbiased check)
    val_proba = model.predict_proba(X_val)[:, 1]
    evaluate_credit_model(y_val, val_proba, 'XGBoost - Validation')

    test_proba = model.predict_proba(X_test)[:, 1]
    evaluate_credit_model(y_test, test_proba, 'XGBoost - Test')

    # Save the trained model and its exact feature list
    joblib.dump(model, os.path.join(MODEL_DIR, 'xgb_with_tradeline.pkl'))
    joblib.dump(artifacts['tree_feature_cols'], os.path.join(MODEL_DIR, 'xgb_with_tradeline_features.pkl'))
    print(f"\nModel saved to: {MODEL_DIR}")

    xgb_with_tradeline_features = joblib.load(os.path.join(MODEL_DIR, 'xgb_with_tradeline_features.pkl'))

    assert list(X_train.columns) == artifacts['tree_feature_cols']
    assert list(X_val.columns) == artifacts['tree_feature_cols']
    assert list(X_test.columns) == artifacts['tree_feature_cols']

    print(f"Columns before tree feature selection: {X_train.shape[1]}")

    expected_after_drop = set(artifacts['tree_feature_cols'])
    actual_cols = set(X_train.columns) - set(artifacts['tree_drop_cols'])

    extra_cols = actual_cols - expected_after_drop
    missing_cols = expected_after_drop - actual_cols

    print(f"Extra columns (unexpected, not in the trained model): {extra_cols}")
    print(f"Missing columns (expected, but not present): {missing_cols}")

if __name__ == '__main__':
    main()