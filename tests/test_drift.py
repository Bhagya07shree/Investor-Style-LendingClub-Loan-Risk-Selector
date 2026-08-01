"""
tests/test_drift.py

Quick smoke test for monitor_drift.py — pulls a batch of real rows
from the raw CSV to simulate a "recent incoming loans" file, and runs
it through check_drift() to confirm PSI calculation works end-to-end.
"""

import os
import pandas as pd
from src.pipeline.monitor_drift import check_drift

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'src', 'data', 'raw', 'accepted_2007_to_2018Q4.csv')


def main():
    # Pull a reasonably large sample — PSI needs enough rows per
    # bucket to be meaningful, unlike predict.py's 10-row test
    print("Loading sample rows from raw CSV...")
    sample_df = pd.read_csv(DATA_PATH, nrows=2000, low_memory=False)
    print(f"Sample shape: {sample_df.shape}")

    sample_df_no_target = sample_df.drop(columns=['loan_status'], errors='ignore')

    print("\nChecking for drift against training baseline...")
    report = check_drift(sample_df_no_target)

    print("\nTop 10 features by PSI:")
    print(report.head(20).to_string(index=False))

    alerts = report[report['status'].str.contains('ALERT')]
    warnings = report[report['status'].str.contains('WARNING')]
    print(f"\nFeatures with major drift (ALERT): {len(alerts)}")
    print(f"Features with moderate drift (WARNING): {len(warnings)}")
    print(f"Stable features: {len(report) - len(alerts) - len(warnings)}")


if __name__ == '__main__':
    main()