"""
src/pipeline/predict.py

Loads the final trained model and scores new, unseen loan applications.
Uses the SAME preprocess.py transform() function that train.py uses
internally, so predictions are guaranteed to use identical preprocessing
to what the model was trained on.

Uses the CALIBRATED model (not the raw one) for scoring, since the
approve/reject threshold below was chosen based on calibrated
probabilities. The raw model's probabilities run much higher than true
default rates (a known effect of training with scale_pos_weight), so
using the raw model here would make the threshold meaningless.

Usage as a script (CLI):
    python -m src.pipeline.predict --input new_loans.csv --output predictions.csv

Usage as an import (e.g. from a Streamlit app):
    from src.pipeline.predict import predict_default_probability
    result = predict_default_probability(new_loans_df)
"""

import os
import argparse
import joblib
import pandas as pd

from src.pipeline.preprocess import load_artifacts, transform

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'src', 'models')

# Use the CALIBRATED model — its probability outputs actually reflect
# real default rates, unlike the raw model's inflated probabilities.
MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_with_tradeline_calibrated.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'xgb_with_tradeline_features.pkl')

# Business-chosen risk threshold: reject any loan predicted at or above
# this default probability. Chosen from a cost/benefit analysis
# (reports/business_metrics.md) using calibrated probabilities — net
# estimated profit stayed positive up to around this point and turned
# negative beyond it. This is NOT the technical default of 0.5, which
# the same analysis showed would actually be the LEAST profitable
# choice available.
DEFAULT_RISK_THRESHOLD = 0.20


def load_model():
    """
    Loads two things from disk:
    1. The trained (calibrated) model itself.
    2. The list of column names the model expects, in the exact order
       it expects them. Used later to double-check the data is shaped
       correctly before we ask the model to predict.
    """
    model = joblib.load(MODEL_PATH)
    expected_features = joblib.load(FEATURES_PATH)
    return model, expected_features


def predict_default_probability(raw_df, model=None, expected_features=None,
                                  artifacts=None, threshold=DEFAULT_RISK_THRESHOLD):
    """
    Main function: takes raw loan data and returns both a default-risk
    probability AND an approve/reject decision for each row.

    raw_df must have the same columns as the original LendingClub CSV.
    It should NOT include a loan_status column, since new applications
    don't have one yet.

    Returns a dataframe with two columns:
    - default_probability: a number between 0 and 1 (e.g. 0.18 means
      an estimated 18% chance this loan defaults)
    - decision: "approve" or "reject", based on the threshold

    Raises ValueError if the preprocessed data's columns don't exactly
    match what the model expects. This should never happen in normal
    use — it means the model and the preprocessing artifacts were saved
    from different, inconsistent runs. Rather than silently guessing
    (e.g. filling a missing column with 0), this fails loudly, since a
    silent mismatch could produce a confident-looking but wrong
    prediction for a real lending decision.
    """
    if model is None or expected_features is None:
        model, expected_features = load_model()
    if artifacts is None:
        artifacts = load_artifacts()

    # Run the raw data through the exact same cleaning, feature
    # engineering, and encoding steps used during training.
    X = transform(raw_df, artifacts=artifacts)

    # transform() already reindexes to artifacts['tree_feature_cols'].
    # If that doesn't match the model's own expected feature list, the
    # model and artifacts are out of sync — 
    missing = set(expected_features) - set(X.columns)
    extra = set(X.columns) - set(expected_features)
    if missing or extra:
        raise ValueError(
            f"Feature mismatch between model and preprocessing artifacts.\n"
            f"Missing: {missing}\nUnexpected: {extra}\n"
            f"This likely means the model and artifacts were saved from "
            f"different runs — retrain via train.py to regenerate both consistently."
        )

    X = X[expected_features]

    # predict_proba gives two probabilities per row: chance of "not
    # default" and chance of "default". We only want the second one.
    proba = model.predict_proba(X)[:, 1]

    decision = pd.Series(proba, index=raw_df.index).apply(
        lambda p: 'reject' if p >= threshold else 'approve'
    )

    return pd.DataFrame({
        'default_probability': proba,
        'decision': decision,
    }, index=raw_df.index)


def score_csv(input_path, output_path, threshold=DEFAULT_RISK_THRESHOLD):
    """
    Reads a CSV file of new loan applications, scores every row, and
    writes a new CSV with two extra columns: default_probability and decision.
    """
    print(f"Loading input data from: {input_path}")
    raw_df = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded {len(raw_df)} rows")

    model, expected_features = load_model()
    artifacts = load_artifacts()

    print(f"Scoring, using risk threshold: {threshold}")
    scored = predict_default_probability(raw_df, model, expected_features, artifacts, threshold)

    result = raw_df.copy()
    result['default_probability'] = scored['default_probability']
    result['decision'] = scored['decision']

    approve_count = (result['decision'] == 'approve').sum()
    print(f"Approved: {approve_count} / {len(result)} ({approve_count / len(result):.1%})")

    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} scored rows to: {output_path}")


def main():
    """
     this file can be run directly from the terminal, like:
    python -m src.pipeline.predict --input new_loans.csv --output predictions.csv --threshold 0.20
    """
    parser = argparse.ArgumentParser(description="Score loan applications for default risk.")
    parser.add_argument('--input', required=True, help="Path to a CSV of new loan applications.")
    parser.add_argument('--output', required=True, help="Path to save the scored CSV to.")
    parser.add_argument('--threshold', type=float, default=DEFAULT_RISK_THRESHOLD,
                         help=f"Risk threshold for reject decision (default: {DEFAULT_RISK_THRESHOLD}).")
    args = parser.parse_args()

    score_csv(args.input, args.output, args.threshold)


if __name__ == '__main__':
    main()