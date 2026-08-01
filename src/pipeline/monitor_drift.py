"""
src/pipeline/monitor_drift.py

Compares a batch of new (recent) loan data against the original
training data's feature distributions, using PSI (Population Stability
Index)

Usage:
    python -m src.pipeline.monitor_drift --input recent_loans.csv
"""

import os
import argparse
import numpy as np
import pandas as pd

from src.pipeline.preprocess import load_artifacts, transform

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PARQUET_DIR = os.path.join(PROJECT_ROOT, 'src', 'data', 'processed', 'parquet_chunks')

# The FULL feature-engineered training data (150 cols, before the
# tree-specific drop) - saved from the FE notebook.
BASELINE_FULL_PATH = os.path.join(PARQUET_DIR, 'X_train_fe.parquet')


def calculate_psi(baseline_series, new_series, n_bins=10):
    """Same PSI logic used in the drift analysis notebook."""
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = baseline_series.quantile(quantiles).values
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_edges = np.unique(bin_edges)

    baseline_counts = pd.cut(baseline_series, bins=bin_edges).value_counts(normalize=True).sort_index()
    new_counts = pd.cut(new_series, bins=bin_edges).value_counts(normalize=True).sort_index()

    baseline_pct = baseline_counts.replace(0, 0.0001)
    new_pct = new_counts.replace(0, 0.0001)

    return np.sum((new_pct - baseline_pct) * np.log(new_pct / baseline_pct))


def load_baseline(artifacts):
    """
    Loads the full FE training data, then applies the same tree-drop
    step used for the production model, so the baseline's columns
    match what predict.py actually uses.
    """
    baseline_full = pd.read_parquet(BASELINE_FULL_PATH)
    baseline_tree = baseline_full.drop(columns=artifacts['tree_drop_cols'], errors='ignore')
    return baseline_tree


def check_drift(new_raw_df, psi_warning_threshold=0.10, psi_alert_threshold=0.25):
    """
    Compares new incoming loan data against the training baseline,
    feature by feature, and flags anything showing meaningful drift.
    """
    artifacts = load_artifacts()
    baseline_df = load_baseline(artifacts)

    # Run new data through the same preprocessing as training,
    new_df = transform(new_raw_df, artifacts=artifacts)

    numeric_cols = baseline_df.select_dtypes(include=[np.number]).columns
    # Skip binary/flag columns — PSI's quantile-bucketing breaks down
    # for them, as found during the original drift analysis
    continuous_cols = [c for c in numeric_cols if baseline_df[c].nunique() > 2 and c in new_df.columns]

    results = []
    for col in continuous_cols:
        psi = calculate_psi(baseline_df[col].dropna(), new_df[col].dropna())
        if psi < psi_warning_threshold:
            status = 'stable'
        elif psi < psi_alert_threshold:
            status = 'WARNING - moderate shift'
        else:
            status = 'ALERT - major shift'
        results.append({'feature': col, 'psi': round(psi, 4), 'status': status})

    report = pd.DataFrame(results).sort_values('psi', ascending=False)
    return report


def main():
    parser = argparse.ArgumentParser(description="Check new loan data for feature drift vs training baseline.")
    parser.add_argument('--input', required=True, help="Path to a CSV of recent loan applications.")
    args = parser.parse_args()

    new_raw_df = pd.read_csv(args.input, low_memory=False)
    report = check_drift(new_raw_df)

    print(report.to_string(index=False))

    alerts = report[report['status'].str.contains('ALERT')]
    if len(alerts) > 0:
        print(f"\n {len(alerts)} feature(s) showing major drift — model may need review/retraining.")
    else:
        print("\n  No major drift detected.")


if __name__ == '__main__':



## "C:\Users\bhagyashree.s\anaconda3\python.exe" -m streamlit run app.py
    main()