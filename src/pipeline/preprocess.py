"""
src/pipeline/preprocess.py

Single source of truth for turning raw LendingClub-schema data into
model-ready features. Used by BOTH train.py (to build train/val/test)
and predict.py (to score new, unseen loan applications) — so training
and inference are guaranteed to apply identical preprocessing.

All fitted objects (imputers, encoder, caps, mappings) are loaded from
src/artifacts/ and only ever APPLIED here, never re-fit.
"""

import os
import pickle
import numpy as np
import pandas as pd

# Folder where all fitted preprocessing objects live
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')

# Every artifact file needed for preprocessing
ARTIFACT_NAMES = [
    'median_imputer', 'cat_imputer', 'ohe_encoder', 'winsor_caps', 'rare_states',
    'num_impute_cols', 'cat_impute_cols', 'ohe_cols', 'flag_map', 'binary_flags',
    'sentinel_cols', 'tradeline_cols', 'business_caps', 'subgrade_mapping',
    'verification_mapping', 'initial_list_map', 'home_ownership_merge',
    'final_drop_list', 'drop_after_eda', 'drop_replaced_source_cols',
    'tree_drop_flags', 'tree_drop_cols', 'default_statuses', 'non_default_statuses',
    'tree_feature_cols', 'fe_config',
]


# =========================================================
# ARTIFACT LOADING
# =========================================================

def load_artifact(name):
    """Loads one pickled artifact by name from src/artifacts/."""
    with open(os.path.join(ARTIFACT_DIR, f"{name}.pkl"), 'rb') as f:
        return pickle.load(f)


def load_artifacts():
    """Loads every artifact into one dict, keyed by name."""
    return {name: load_artifact(name) for name in ARTIFACT_NAMES}


# =========================================================
# HELPERS
# =========================================================

def clean_emp_length(col):
    """Turns emp_length text ('< 1 year', '10+ years') into plain numbers."""
    col = col.astype(str)
    col = col.str.replace(r'< 1 year', '0', regex=False)   # "< 1 year" -> "0"
    col = col.str.replace(r'\+ years', '', regex=True)      # "10+ years" -> "10"
    col = col.str.replace(r' years', '', regex=True)        # "2 years" -> "2"
    col = col.str.replace(r' year', '', regex=True)         # "1 year" -> "1"
    return pd.to_numeric(col, errors='coerce')

# =========================================================
# DateTime Conversion
# =========================================================

def convert_date_columns(df):
    """Converts issue_d and earliest_cr_line from text to real dates."""
    for col in ['issue_d', 'earliest_cr_line']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%b-%Y', errors='coerce')
    return df


# =========================================================
# TARGET CONSTRUCTION
# Training/backtesting only — never called at inference time,
# since new loan applications have no loan_status yet.
# =========================================================

def build_target(df, artifacts):
    """
    Turns loan_status into a 0/1 default label.
    - Fully Paid, Charged Off, etc. get a direct label.
    - 'Current' loans only count as non-default if they've already run
      their FULL term (otherwise the outcome isn't known yet, so the
      row gets dropped instead of guessed at).
    """
    cfg = artifacts['fe_config']
    snapshot_date = pd.Timestamp(cfg['snapshot_date'])
    maturity_pct = cfg['maturity_threshold_pct']

    # How many months have passed since the loan was issued
    months_since_issue = (
        (snapshot_date.year - df['issue_d'].dt.year) * 12 +
        (snapshot_date.month - df['issue_d'].dt.month)
    )
    # Loan term in months (36 or 60), pulled out of the "term" text
    term_months = df['term'].astype(str).str.extract(r'(\d+)').astype(float)
    term_months = term_months.fillna(term_months.median())
    pct_of_term_elapsed = months_since_issue / term_months.iloc[:, 0]

    # A "Current" loan only counts as trustworthy non-default once it's
    # fully run its course (100% of term elapsed)
    mask_mature_current = (df['loan_status'] == 'Current') & (pct_of_term_elapsed >= maturity_pct)

    # Keep rows with a known outcome, or a mature "Current" loan.
    # Drop everything else (outcome still uncertain).
    mask_keep = (
        df['loan_status'].isin(artifacts['default_statuses'] + artifacts['non_default_statuses'])
        | mask_mature_current
    )
    df = df[mask_keep].copy()

    # 1 = defaulted, 0 = paid off successfully
    df['target'] = df['loan_status'].isin(artifacts['default_statuses']).astype('int8')
    return df.drop(columns=['loan_status'])


# =========================================================
# FEATURE ENGINEERING 
# =========================================================

def run_feature_engineering(df, artifacts):
    """
    Cleans raw data and builds new features: drops leakage columns,
    fixes bad values, adds missingness flags, and engineers ratios
    like payment-to-income.

    NOTE: grade/application_type/disbursement_method/emp_title are NOT
    dropped here anymore — that now happens in apply_encoding(), since
    the categorical imputer needs these columns to still exist as text.
    """
    cfg = artifacts['fe_config']

    # --- Remove columns that leak future info or add no value ---
    df = df.drop(columns=artifacts['final_drop_list'], errors='ignore')

    if 'emp_length' in df.columns:
        df['emp_length'] = clean_emp_length(df['emp_length'])

    # --- Fix known bad/placeholder values ---
    if 'annual_inc' in df.columns:
        df['annual_inc'] = df['annual_inc'].replace(0, np.nan)  # 0 income = missing, not real
        # Very low income + a much bigger loan = probably bad data
        suspicious = (
            (df['annual_inc'] < cfg['low_income_threshold']) &
            (df['loan_amnt'] / df['annual_inc'] > cfg['loan_to_income_ratio'])
        )
        df.loc[suspicious, 'annual_inc'] = np.nan

    if 'dti' in df.columns:
        df['dti'] = df['dti'].replace(-1, np.nan)  # -1 is a known placeholder, not a real DTI

    for col in ['tot_hi_cred_lim', 'total_rev_hi_lim']:
        if col in df.columns:
            df[col] = df[col].replace(cfg['credit_limit_sentinel'], np.nan)  # 9,999,999 = placeholder

    # --- Flags for "did this ever happen", BEFORE filling missing values ---
    # Missing here usually means the event never happened, not bad data.
    for flag_name, source_col in artifacts['flag_map'].items():
        df[flag_name] = df[source_col].notna().astype('int8') if source_col in df.columns else 0

    # --- New features built from existing columns ---
    if {'fico_range_low', 'fico_range_high'}.issubset(df.columns):
        df['fico_score'] = ((df['fico_range_low'] + df['fico_range_high']) / 2).astype('float32')  # one clean FICO score

    if {'installment', 'annual_inc'}.issubset(df.columns):
        df['pti_ratio'] = (df['installment'] / (df['annual_inc'] / 12)).astype('float32')  # payment as % of monthly income

    if {'loan_amnt', 'annual_inc'}.issubset(df.columns):
        df['loan_to_income'] = (df['loan_amnt'] / df['annual_inc']).astype('float32')  # loan size vs income

    if {'tot_cur_bal', 'tot_hi_cred_lim'}.issubset(df.columns):
        df['overall_util'] = (df['tot_cur_bal'] / (df['tot_hi_cred_lim'] + 1)).astype('float32')  # how much credit is used

    if {'issue_d', 'earliest_cr_line'}.issubset(df.columns):
        df['credit_age_yrs'] = ((df['issue_d'] - df['earliest_cr_line']).dt.days / 365).astype('float32')  # years of credit history

    if 'issue_d' in df.columns:
        df['issue_year'] = df['issue_d'].dt.year.astype('Int16')  # used only to split by year, dropped later

    # Any divide-by-near-zero above can create +/- infinity — treat as missing
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    # --- Drop columns replaced by the new features above ---
    # (drop_after_eda — grade, application_type, disbursement_method,
    #  emp_title — is handled later in apply_encoding(), NOT here,
    #  since the categorical imputer needs those columns to still exist)
    df = df.drop(columns=artifacts['drop_replaced_source_cols'], errors='ignore')
    df = df.drop(columns=['issue_d'], errors='ignore')  # no longer needed after issue_year/credit_age_yrs are built

    return df

# =========================================================
# Winsorization(Capping)
# =========================================================
def apply_caps(df, artifacts):
    """
    Caps extreme values and applies log transforms. Runs BEFORE
    imputation and encoding — works purely on numeric columns, so it
    doesn't matter whether categorical columns are still text yet.
    All caps here were learned from training data earlier; this
    function only applies them, never re-learns them.
    """
    cfg = artifacts['fe_config']

    if 'dti' in df.columns:
        df['dti'] = df['dti'].replace(999, np.nan)
        df['dti'] = df['dti'].mask(df['dti'] > cfg['dti_upper_cap'], np.nan)

    for col, cap in artifacts['winsor_caps'].items():
        if col in df.columns:
            df[col] = df[col].clip(upper=cap)

    for col, cap in artifacts['business_caps'].items():
        if col in df.columns:
            df[col] = df[col].clip(upper=cap)

    for col in cfg['log1p_cols']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # forces None/bad values to real NaN
            df[col] = np.log1p(df[col].clip(lower=0))

    return df

# =========================================================
# Imputation
# =========================================================
def apply_imputation(df, artifacts):
    """
    Fills in missing values. Numbers get the training median, categories
    get the training mode. A few special "months since X" columns get
    filled with 999 instead, meaning "this never happened."
    """
    num_cols = [c for c in artifacts['num_impute_cols'] if c in df.columns]
    if num_cols:
        df[num_cols] = artifacts['median_imputer'].transform(df[num_cols])

    cat_cols = [c for c in artifacts['cat_impute_cols'] if c in df.columns]
    if cat_cols:
        df[cat_cols] = artifacts['cat_imputer'].transform(df[cat_cols])

    for col in artifacts['sentinel_cols']:
        if col in df.columns:
            df[col] = df[col].fillna(999).astype('float32')  # 999 = event never happened

    return df

# =========================================================
# Encoding
# =========================================================
def apply_encoding(df, artifacts):
    """
    Turns text categories into numbers. Runs AFTER imputation, since
    the categorical imputer (mode-fill) was fit expecting these columns
    to still be raw text — encoding them earlier would break it.
    """
    # Drop zero-info / redundant columns found during EDA — done here
    # (not earlier) since they're plain drops, unrelated to imputation timing
    df = df.drop(columns=artifacts['drop_after_eda'], errors='ignore')

    if 'term' in df.columns:
        df['term'] = df['term'].astype(str).str.extract(r'(\d+)').astype('int8')  # "36 months" -> 36

    if 'initial_list_status' in df.columns:
        df['initial_list_status'] = df['initial_list_status'].map(artifacts['initial_list_map']).astype('int8')  # f/w -> 0/1

    if 'sub_grade' in df.columns:
        df['sub_grade'] = df['sub_grade'].astype(str).map(artifacts['subgrade_mapping']).astype('int8')  # A1-G5 -> 1-35

    if 'verification_status' in df.columns:
        # Ordered by actual default rate found in the data, not alphabetically
        df['verification_status'] = df['verification_status'].astype(str).map(artifacts['verification_mapping']).astype('int8')

    if 'home_ownership' in df.columns:
        df['home_ownership'] = df['home_ownership'].replace(artifacts['home_ownership_merge'])  # rare categories -> OTHER

    if 'addr_state' in df.columns:
        # States with very few training rows get grouped into one bucket
        df['addr_state'] = df['addr_state'].replace({s: 'OTHER_STATE' for s in artifacts['rare_states']})

    # One-hot encode using the encoder already fit on training data
    ohe = artifacts['ohe_encoder']
    ohe_cols = artifacts['ohe_cols']
    ohe_array = ohe.transform(df[ohe_cols])
    ohe_df = pd.DataFrame(ohe_array, columns=ohe.get_feature_names_out(ohe_cols), index=df.index)
    df = pd.concat([df.drop(columns=ohe_cols), ohe_df], axis=1)

    return df

def select_tree_features(df, artifacts):
    """
    Picks the final columns for the chosen model (xgb_with_tradeline):
    drops the 5 redundant flag columns, issue_year, and delinq_amnt.
    Keeps all tradeline (bureau) columns. Column order is forced to
    match exactly what the model was trained on.
    """
    df = df.drop(columns=artifacts['tree_drop_cols'], errors='ignore')
    feature_cols = artifacts['tree_feature_cols']
    return df[feature_cols]  # exact column order the model expects


# =========================================================
# MAIN Function — what predict.py calls
# =========================================================

def transform(df, artifacts=None):
    """
    Takes raw, new loan application data and turns it into the exact
    feature matrix the model expects, ready for model.predict_proba().

    Order matters: caps -> imputation -> encoding. The categorical
    imputer expects raw text columns, so encoding must happen last.
    """
    if artifacts is None:
        artifacts = load_artifacts()

    df = df.copy()
    df = convert_date_columns(df) 
    df = run_feature_engineering(df, artifacts)
    df = apply_caps(df, artifacts)
    df = apply_imputation(df, artifacts)
    df = apply_encoding(df, artifacts)
    df = select_tree_features(df, artifacts)
    return df