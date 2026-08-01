"""
tests/test_predict.py

Quick smoke test for predict.py — pulls a few real rows from the raw
CSV (with loan_status removed, simulating genuine new applications),
runs them through predict_default_probability(), and checks the
output looks sane.

Run from the project root:
    python -m tests.test_predict
"""

import os
import pandas as pd
from src.pipeline.predict import predict_default_probability

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'src', 'data', 'raw', 'accepted_2007_to_2018Q4.csv')


def main():
    # --- Step 1: build a small, realistic test sample ---
    print("Loading sample rows from raw CSV...")
    sample_df = pd.read_csv(DATA_PATH, nrows=10, low_memory=False)
    print(f"Sample shape: {sample_df.shape}")

    # Simulate a real new application — no outcome known yet
    sample_df_no_target = sample_df.drop(columns=['loan_status'], errors='ignore')
    print(f"Has loan_status removed: {'loan_status' not in sample_df_no_target.columns}")

    # --- Step 2: run it through predict.py ---
    print("\nScoring sample...")
    result = predict_default_probability(sample_df_no_target)

    print("\nResult:")
    print(result)

    # --- Step 3: basic sanity checks ---
    assert result['default_probability'].between(0, 1).all(), "Probabilities out of [0, 1] range!"
    assert result['decision'].isin(['approve', 'reject']).all(), "Unexpected decision value!"
    print("\n All sanity checks passed.")


if __name__ == '__main__':
    main()