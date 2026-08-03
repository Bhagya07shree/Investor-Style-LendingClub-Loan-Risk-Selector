"""
app.py — Investor-Style LendingClub Loan Risk Selector

Run from the project root:
    streamlit run app.py

Overview
--------
This Streamlit application lets a user select borrower and loan attributes,
run the trained default-risk model, and review an investor-style risk summary.

"""

from __future__ import annotations

import datetime
import os
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.pipeline.predict import MODEL_DIR, predict_default_probability
from src.pipeline.preprocess import load_artifacts, transform

from feature_explanations import FEATURE_EXPLANATIONS
from available_fields_and_field_categories import AVAILABLE_FIELDS, FIELD_CATEGORIES


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Investor-Style LendingClub Loan Risk Selector",
    page_icon="🏦",
    layout="wide",
)

# =========================================================
# CONSTANTS
# =========================================================
DEFAULT_RISK_THRESHOLD = 0.20
LGD_ASSUMPTION = 0.60

EXAMPLE_PROFILES: Dict[str, Dict[str, Any]] = {
    "Low Risk Case": {
        "values": {
            "fico_range_low": 760,
            "fico_range_high": 764,
            "dti": 9.5,
            "annual_inc": 95000,
            "revol_util": 22,
            "delinq_2yrs": 0,
            "loan_amnt": 8000,
            "term": "36 months",
            "sub_grade": "A2",
            "int_rate": 7.5,
        },
        "description": "Strong credit score, low debt burden, healthy income, modest utilization, and top-tier grade.",
        "icon": "🟢",
    },
    "Balanced Yield Case": {
        "values": {
            "fico_range_low": 680,
            "fico_range_high": 684,
            "dti": 21.0,
            "annual_inc": 55000,
            "revol_util": 58,
            "delinq_2yrs": 1,
            "loan_amnt": 15000,
            "term": "36 months",
            "sub_grade": "C3",
            "int_rate": 15.0,
        },
        "description": "Moderate borrower profile with average credit quality, some leverage pressure, and mid-range pricing.",
        "icon": "🟡",
    },
    "Aggressive Risk Case": {
        "values": {
            "fico_range_low": 640,
            "fico_range_high": 644,
            "dti": 34.5,
            "annual_inc": 32000,
            "revol_util": 95,
            "delinq_2yrs": 3,
            "loan_amnt": 30000,
            "term": "60 months",
            "sub_grade": "F2",
            "int_rate": 26.0,
        },
        "description": "Weaker borrower profile with elevated leverage, high utilization, multiple delinquencies, and long duration risk.",
        "icon": "🔴",
    },
}

ALL_LABEL_TO_KEY = {meta["label"]: key for key, meta in AVAILABLE_FIELDS.items()}


# =========================================================
# SESSION STATE
# =========================================================
def initialize_session_state() -> None:
    """
    Initialize all application-level session state values exactly once.
    """
    defaults = {
        "selected_keys": [],
        "field_picker_labels": [],
        "user_inputs": {},
        "editing_keys": set(),     # fields the user has reopened for editing
        "confirmed_keys": set(),   # fields the user has explicitly saved
        "summary_editing_keys": set(),  # fields being edited INLINE in the summary table
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def load_example_profile(profile_values: Dict[str, Any]) -> None:
    """
    Preload widgets and selected fields from one example borrower profile.

    This function updates session state first, then the caller should trigger
    st.rerun() so the widgets are rebuilt with the example values.
    """
    st.session_state.selected_keys = list(profile_values.keys())

    for key, value in profile_values.items():
        field = AVAILABLE_FIELDS[key]

        if field["type"] == "number_optional":
            is_blank = value is None or pd.isna(value)
            st.session_state[f"blank_{key}"] = is_blank
            st.session_state[f"input_{key}"] = field.get("default", 0) if is_blank else value
        else:
            st.session_state[f"input_{key}"] = value

    st.session_state.user_inputs = profile_values.copy()


def search_fields_and_categories(
    query: str,
    available_fields: Dict[str, Dict[str, Any]],
    field_categories: Dict[str, List[str]],
) -> List[str]:
    """
    Search fields by category name, field name, or display label.
    Returns ordered field keys without duplicates.
    """
    query = query.strip().lower()
    if not query:
        return list(available_fields.keys())

    tokens = query.split()
    matched_keys: List[str] = []

    for category, keys in field_categories.items():
        if all(token in category.lower() for token in tokens):
            matched_keys.extend(keys)

    for key, meta in available_fields.items():
        searchable_text = f"{key} {meta['label']}".lower()
        if all(token in searchable_text for token in tokens):
            matched_keys.append(key)

    seen = set()
    ordered = []
    for key in matched_keys:
        if key not in seen:
            seen.add(key)
            ordered.append(key)

    return ordered

def add_selected_fields_from_picker() -> None:
    """
    Add only newly selected fields from the picker.
    Already-saved fields stay saved and should not reopen in the input form.
    Newly added fields open in edit mode.
    """
    current_selected = list(st.session_state.selected_keys)
    current_set = set(current_selected)

    new_keys = []
    for label in st.session_state.field_picker_labels:
        key = ALL_LABEL_TO_KEY.get(label)
        if key is not None and key not in current_set:
            new_keys.append(key)

    st.session_state.selected_keys = current_selected + new_keys

    for key in new_keys:
        st.session_state.editing_keys.add(key)
        st.session_state.confirmed_keys.discard(key)

    st.session_state.field_picker_labels = []

def clear_all_selected_fields() -> None:
    """
    Reset all selected fields and their associated session state.
    """
    for key in AVAILABLE_FIELDS:
        st.session_state.pop(f"input_{key}", None)
        st.session_state.pop(f"blank_{key}", None)

    st.session_state.selected_keys = []
    st.session_state.field_picker_labels = []
    st.session_state.user_inputs = {}
    st.session_state.confirmed_keys = set()
    st.session_state.editing_keys = set()


def gather_user_inputs(selected_keys: List[str]) -> Dict[str, Any]:
    """
    Read saved summary values first, otherwise read widget values.
    """
    user_inputs: Dict[str, Any] = {}

    for key in selected_keys:
        field = AVAILABLE_FIELDS[key]

        if key in st.session_state.user_inputs:
            user_inputs[key] = st.session_state.user_inputs[key]
            continue

        if field["type"] == "number_optional":
            if st.session_state.get(f"blank_{key}", True):
                user_inputs[key] = None
            else:
                user_inputs[key] = st.session_state.get(f"input_{key}")
        else:
            user_inputs[key] = st.session_state.get(f"input_{key}")

    return user_inputs


def build_raw_row(user_inputs: Dict[str, Any]) -> pd.DataFrame:
    """
    Build the single-row raw input DataFrame expected by the preprocessing pipeline.
    Fields not selected by the user are left as NaN and handled downstream.
    """
    row = {key: user_inputs.get(key, np.nan) for key in AVAILABLE_FIELDS}

    row["issue_d"] = datetime.date.today().strftime("%b-%Y")
    row["delinq_amnt"] = 0
    row["application_type"] = "Individual"
    row["disbursement_method"] = "Cash"

    sub_grade = user_inputs.get("sub_grade")
    row["grade"] = sub_grade[0] if sub_grade else "C"

    return pd.DataFrame([row])


def get_risk_band(probability: float) -> Tuple[str, str]:
    """
    Convert predicted default probability into a user-friendly risk band.
    """
    if probability < 0.15:
        return "Low", "🟢"
    if probability < 0.25:
        return "Moderate", "🟡"
    if probability < 0.40:
        return "High", "🟠"
    return "Very High", "🔴"


def get_investor_decision_text(probability: float, threshold: float) -> str:
    """
    Compare predicted default probability with investor risk preference.
    """
    if probability <= threshold:
        return "Eligible for Consideration"
    return "Outside Investment Preference"


def get_threshold_guidance(threshold: float) -> tuple[str, str, str]:
    """
    Return the investment strategy, explanation, and alert type
    for the selected maximum default risk.
    """
    if threshold <= 0.15:
        return (
            "🟢 Very Cautious",
            "Only borrowers with a low chance of default will be recommended. This approach prioritizes safety over the number of investment opportunities.",
            "success",
        )

    elif threshold <= 0.25:
        return (
            "🟡 Balanced",
            "A balance between safety and investment opportunities. Borrowers with a moderate chance of default may still be considered.",
            "info",
        )

    elif threshold <= 0.40:
        return (
            "🟠 Growth-Focused",
            "More borrowers will qualify, creating additional investment opportunities, but the risk of financial loss also increases.",
            "warning",
        )

    return (
        "🔴 Aggressive",
        "Many borrowers will qualify, but there is a significantly higher chance of defaults and financial losses.",
        "error",
    )


def get_case_result_message(
    probability: float,
    threshold: float,
    risk_band: str,
) -> tuple[str, str]:
    """
    Return a dynamic message based on the investor's selected risk threshold.
    """
    probability_pct = probability * 100
    threshold_pct = threshold * 100
    difference_pct = abs(probability_pct - threshold_pct)

    if probability <= threshold:
        return (
            "success",
            f"This borrower is classified as {risk_band.lower()} risk with a predicted default probability of "
            f"{probability_pct:.1f}%. This is {difference_pct:.1f} percentage points below your selected "
            f"maximum risk threshold of {threshold_pct:.0f}%, so the application is within your investment preference."
        )

    return (
        "warning",
        f"This borrower is classified as {risk_band.lower()} risk with a predicted default probability of "
        f"{probability_pct:.1f}%. This exceeds your selected maximum risk threshold of "
        f"{threshold_pct:.0f}% by {difference_pct:.1f} percentage points, so it falls outside your current investment preference."
    )


def estimate_profit_loss(
    user_inputs: Dict[str, Any],
    default_probability: float,
    lgd: float = LGD_ASSUMPTION,
) -> Dict[str, Any]:
    """
    Estimate a simple expected-value view using:
    - loan amount
    - stated interest rate
    - predicted probability of default
    - assumed loss given default
    """
    loan_amnt = float(user_inputs.get("loan_amnt", 0) or 0)
    int_rate = float(user_inputs.get("int_rate", 0) or 0) / 100.0

    performing_probability = 1.0 - default_probability
    expected_interest_income = loan_amnt * int_rate * performing_probability
    expected_credit_loss = loan_amnt * default_probability * lgd

    display_interest_income = round(expected_interest_income)
    display_credit_loss = round(expected_credit_loss)

    return {
        "expected_interest_income": expected_interest_income,
        "expected_credit_loss": expected_credit_loss,
        "display_interest_income": display_interest_income,
        "display_credit_loss": display_credit_loss,
        "display_net": display_interest_income - display_credit_loss,
        "lgd": lgd,
    }


@st.cache_resource(show_spinner=False)
def load_shap_explainer():
    """
    Load and cache the tree explainer once for the current Streamlit session.
    """
    model_path = os.path.join(MODEL_DIR, "xgb_with_tradeline.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"SHAP model file not found: {model_path}"
        )

    raw_model = joblib.load(model_path)

    return shap.TreeExplainer(raw_model)


def get_shap_explanation(
    X_transformed: pd.DataFrame,
    explainer,
    top_n: int = 8,
) -> pd.DataFrame:
    """
    Compute SHAP values and return the top features ranked by absolute impact.
    """
    shap_exp = explainer(X_transformed)

    shap_df = pd.DataFrame(
        {
            "feature": X_transformed.columns,
            "shap_value": shap_exp.values[0],
            "feature_value": X_transformed.iloc[0].values,
        }
    )
    shap_df["abs_shap"] = shap_df["shap_value"].abs()
    shap_df = shap_df.sort_values("abs_shap", ascending=False).head(top_n)

    return shap_df.sort_values("shap_value")


def plot_shap_bar(shap_df: pd.DataFrame):
    """
    Plot a horizontal SHAP contribution chart.
    Positive SHAP values increase predicted default risk.
    Negative SHAP values reduce predicted default risk.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d62728" if value > 0 else "#2ca02c" for value in shap_df["shap_value"]]

    ax.barh(shap_df["feature"], shap_df["shap_value"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Impact on predicted default risk")
    ax.set_title("Top factors influencing this prediction")
    plt.tight_layout()

    return fig


def get_top_drivers_rule_based(user_inputs: Dict[str, Any]) -> List[str]:
    """
    Fallback explanation when SHAP is unavailable.
    Returns simple, investor-friendly explanations of the main risk drivers.
    """
    drivers: List[str] = []

    def has_value(key: str) -> bool:
        value = user_inputs.get(key)
        return value is not None and not pd.isna(value)

    if has_value("dti") and user_inputs["dti"] > 25:
        drivers.append(
            f"**Debt-to-Income Ratio ({user_inputs['dti']:.1f}%)**  \n"
            "The borrower is using a relatively large portion of their income to repay existing debt. "
            "A higher debt burden may make it more difficult to manage additional loan repayments, "
            "which can increase the likelihood of default."
        )

    if user_inputs.get("sub_grade") and user_inputs["sub_grade"][0] in ("E", "F", "G"):
        drivers.append(
            f"**Loan Sub-Grade ({user_inputs['sub_grade']})**  \n"
            "This loan belongs to a lower credit-quality grade. Borrowers in these grades have historically "
            "shown a higher probability of missing repayments, so this increases the predicted risk."
        )

    if has_value("revol_util") and user_inputs["revol_util"] > 75:
        drivers.append(
            f"**Credit Utilization ({user_inputs['revol_util']:.1f}%)**  \n"
            "The borrower is using a large percentage of their available revolving credit. "
            "High utilization can indicate financial pressure and is generally associated with higher credit risk."
        )

    if has_value("delinq_2yrs") and user_inputs["delinq_2yrs"] > 0:
        drivers.append(
            f"**Recent Delinquencies ({int(user_inputs['delinq_2yrs'])})**  \n"
            "The borrower has had one or more delinquent payments during the last two years. "
            "Past repayment problems are an important indicator of future repayment risk."
        )

    if has_value("fico_range_low") and user_inputs["fico_range_low"] < 660:
        drivers.append(
            f"**FICO Credit Score ({int(user_inputs['fico_range_low'])}-{int(user_inputs.get('fico_range_high', user_inputs['fico_range_low']))})**  \n"
            "The borrower's credit score is below the range typically associated with lower-risk borrowers. "
            "Lower credit scores generally indicate a greater likelihood of default."
        )

    if not drivers:
        drivers.append(
            "**Overall Assessment**  \n"
            "No major individual risk factor stands out based on the information provided. "
            "The prediction is influenced by the combined effect of all borrower and loan characteristics "
            "rather than a single dominant factor."
        )

    return drivers


def build_diagnostics_table(
    user_inputs: Dict[str, Any],
    threshold: float,
    probability: float,
    investor_decision: str,
    risk_band: str,
) -> pd.DataFrame:
    """
    Build a small diagnostics table for transparency and debugging.
    """
    return pd.DataFrame(
        [
            {"Metric": "Selected Input Fields", "Value": len(user_inputs)},
            {"Metric": "Decision Threshold", "Value": f"{threshold:.0%}"},
            {"Metric": "Predicted Default Probability", "Value": f"{probability:.2%}"},
            {"Metric": "Risk Band", "Value": risk_band},
            {"Metric": "Investor Decision", "Value": investor_decision},
        ]
    )


def format_feature_help(raw_help) -> str:
    """
    Convert feature explanation content into a Streamlit-safe string.
    """
    if raw_help is None:
        return "No explanation available yet."

    if isinstance(raw_help, str):
        return raw_help

    if isinstance(raw_help, dict):
        lines = []
        for k, v in raw_help.items():
            if v is not None:
                lines.append(f"{k}: {v}")
        return "\n".join(lines) if lines else "No explanation available yet."

    if isinstance(raw_help, (list, tuple)):
        return "\n".join(str(x) for x in raw_help if x is not None)

    return str(raw_help)


# =========================================================
# HEADER
# =========================================================
st.title("🏦 Investor-Style LendingClub Loan Risk Selector")
st.markdown(
    "Evaluate borrower information, estimate loan risk, understand the key factors "
    "behind the recommendation, and review the potential investment outcome."
)

st.caption(
    "🔍 An end-to-end machine learning workflow for evaluating loan applications, "
    "estimating default risk, and supporting investment decisions."
)

st.divider()


# =========================================================
# EXAMPLE PROFILES
# =========================================================
with st.container(border=True):
    st.subheader("🚀 Try an Example Borrower Profile")
    st.caption("Load a sample profile to quickly see how the model behaves across different risk levels.")

    profile_cols = st.columns(len(EXAMPLE_PROFILES))
    for col, (profile_name, profile_data) in zip(profile_cols, EXAMPLE_PROFILES.items()):
        with col:
            with st.container(border=True):
                st.markdown(f"#### {profile_data['icon']} {profile_name}")
                st.caption(profile_data["description"])
                if st.button("Load this profile", key=f"load_{profile_name}", use_container_width=True):
                    load_example_profile(profile_data["values"])
                    st.rerun()

st.write("")


# =========================================================
# FIELD SELECTION
# =========================================================
with st.container(border=True):
    st.subheader("🧩 Select Information to Provide")

    search_query = st.text_input(
        "Search by field name or category",
        placeholder="e.g. Loan, Income & Employment, Credit, Delinquency, Recent, Months...",
    )

    matched_keys = search_fields_and_categories(search_query, AVAILABLE_FIELDS, FIELD_CATEGORIES)
    matched_labels = [AVAILABLE_FIELDS[key]["label"] for key in matched_keys]

    st.session_state.field_picker_labels = [
        label for label in st.session_state.field_picker_labels if label in matched_labels
    ]

    st.multiselect(
        "Available fields",
        options=matched_labels,
        key="field_picker_labels",
        help="Choose one or more fields, then click Add fields.",
    )

    selector_col1, selector_col2 = st.columns(2)
    with selector_col1:
        st.button("➕ Add fields", on_click=add_selected_fields_from_picker, use_container_width=True)

    with selector_col2:
        st.button("🗑️ Clear all fields", on_click=clear_all_selected_fields, use_container_width=True)

    selected_keys = st.session_state.selected_keys

    if not selected_keys:
        st.info("💡 Select fields above to begin building a borrower and loan profile.")
    else:
        st.caption(f"✅ {len(selected_keys)} field(s) selected so far.")

st.write("")


# =========================================================
# DYNAMIC INPUT FORM
# =========================================================
# (unchanged — exactly as you had it)

def confirm_field(key: str) -> None:
    """Mark a field as saved."""
    st.session_state.confirmed_keys.add(key)
    st.session_state.editing_keys.discard(key)


def start_editing(key: str) -> None:
    """Move a saved field back into edit mode."""
    st.session_state.editing_keys.add(key)


def is_actually_filled(key: str) -> bool:
    field = AVAILABLE_FIELDS[key]
    if field["type"] == "number_optional":
        return f"blank_{key}" in st.session_state
    return f"input_{key}" in st.session_state


saved_keys = [
    key
    for key in selected_keys
    if key in st.session_state.confirmed_keys
    and key not in st.session_state.editing_keys
    and is_actually_filled(key)
]

editing_keys = [
    key
    for key in selected_keys
    if key not in st.session_state.confirmed_keys
    or key in st.session_state.editing_keys
]

if selected_keys:
    with st.container(border=True):
        st.subheader("✍️ Enter Borrower & Loan Information")

        if editing_keys:
            progress_pct = len(saved_keys) / len(selected_keys) if selected_keys else 0
            st.progress(progress_pct, text=f"{len(saved_keys)} of {len(selected_keys)} fields saved")

            input_columns = st.columns(3)

            for i, key in enumerate(editing_keys):
                field = AVAILABLE_FIELDS[key]
                prefill = st.session_state.user_inputs.get(key, field.get("default"))
                help_text = format_feature_help(
                    FEATURE_EXPLANATIONS.get(key, "No explanation available yet.")
                )

                with input_columns[i % 3]:
                    if field["type"] == "number":
                        default_value = (
                            float(prefill)
                            if prefill is not None and not pd.isna(prefill)
                            else float(field["default"])
                        )
                        st.number_input(
                            field["label"],
                            min_value=float(field["min"]),
                            max_value=float(field["max"]),
                            value=default_value,
                            step=1.0,
                            key=f"input_{key}",
                            help=help_text,
                        )

                    elif field["type"] == "select":
                        options = field["options"]
                        index = options.index(prefill) if prefill in options else 0
                        st.selectbox(
                            field["label"],
                            options=options,
                            index=index,
                            key=f"input_{key}",
                            help=help_text,
                        )

                    elif field["type"] == "number_optional":
                        blank_key = f"blank_{key}"
                        input_key = f"input_{key}"

                        if blank_key not in st.session_state:
                            st.session_state[blank_key] = prefill is None

                        left_blank = st.checkbox(
                            f"{field['label']} — never happened / unknown",
                            key=blank_key,
                            help=help_text,
                        )

                        if not left_blank:
                            numeric_value = (
                                prefill
                                if prefill is not None and not pd.isna(prefill)
                                else field.get("default", field["min"])
                            )
                            st.number_input(
                                field["label"],
                                min_value=float(field["min"]),
                                max_value=float(field["max"]),
                                value=float(numeric_value),
                                step=1.0,
                                key=input_key,
                                help=help_text,
                            )

            st.write("")
            if st.button("💾 Save Inputs", use_container_width=True, type="primary"):
                for key in editing_keys:
                    confirm_field(key)
                st.session_state.user_inputs = gather_user_inputs(selected_keys)
                st.rerun()
        else:
            st.success("✅ All selected fields are saved. Review them below or run the evaluation.")


# =========================================================
# CURRENT INPUT SUMMARY
# =========================================================
confirmed_keys_list = [
    key for key in selected_keys
    if key in st.session_state.confirmed_keys and is_actually_filled(key)
]

current_inputs = gather_user_inputs(selected_keys)

if confirmed_keys_list:
    st.write("")
    with st.container(border=True):
        st.subheader("📋 Current Input Summary")

        header_col1, header_col2, header_col3 = st.columns([2, 2, 1])
        header_col1.markdown("**Field**")
        header_col2.markdown("**Value**")
        header_col3.markdown("**Action**")
        st.divider()

        for key in confirmed_keys_list:
            field = AVAILABLE_FIELDS[key]

            if key in st.session_state.summary_editing_keys:  
                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    st.write(field["label"])

                with col2:
                    if field["type"] == "number":
                        current_value = current_inputs.get(key)
                        prefill = float(current_value) if current_value is not None else float(field["default"])
                        st.number_input(
                            field["label"],
                            min_value=float(field["min"]),
                            max_value=float(field["max"]),
                            value=prefill,
                            key=f"summary_edit_{key}",
                            label_visibility="collapsed",
                        )
                    elif field["type"] == "select":
                        current_value = current_inputs.get(key)
                        options = field["options"]
                        index = options.index(current_value) if current_value in options else 0
                        st.selectbox(
                            field["label"],
                            options=options,
                            index=index,
                            key=f"summary_edit_{key}",
                            label_visibility="collapsed",
                        )
                    elif field["type"] == "number_optional":                 # <-- ADDED: was missing
                        current_value = current_inputs.get(key)
                        blank_now = st.checkbox(
                            "Never happened / unknown",
                            value=(current_value is None),
                            key=f"summary_edit_blank_{key}",
                        )
                        if not blank_now:
                            prefill = float(current_value) if current_value is not None else float(field["min"])
                            st.number_input(
                                field["label"],
                                min_value=float(field["min"]),
                                max_value=float(field["max"]),
                                value=prefill,
                                key=f"summary_edit_{key}",
                                label_visibility="collapsed",
                            )

                with col3:
                    if st.button("💾 Save", key=f"save_summary_{key}", use_container_width=True):
                        if field["type"] == "number_optional" and st.session_state.get(f"summary_edit_blank_{key}"):
                            new_value = None
                        else:
                            new_value = st.session_state.get(f"summary_edit_{key}")
                        st.session_state.user_inputs[key] = new_value
                        st.session_state.summary_editing_keys.discard(key)   
                        st.rerun()

            else:
                value = current_inputs.get(key)

                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.write(field["label"])
                with col2:
                    st.write("—" if value is None else value)
                with col3:
                    if st.button("✏️ Edit", key=f"edit_summary_{key}", use_container_width=True):
                        st.session_state.summary_editing_keys.add(key)   
                        st.rerun()

        st.divider()
        st.markdown(
            f"**📊 {len(selected_keys)} of {len(AVAILABLE_FIELDS)} fields selected so far "
    f"({len(selected_keys)/len(AVAILABLE_FIELDS):.0%} of full profile)**"
)

st.write("")

# =========================================================
# THRESHOLD SECTION (INVESTOR PREFERENCE)
# =========================================================
with st.container(border=True):
    st.subheader("🎚️ Set Your Investment Safety Preference")

    threshold = st.slider(
        "How much borrower risk are you comfortable accepting?",
        min_value=0.00,
        max_value=0.50,
        value=DEFAULT_RISK_THRESHOLD,
        step=0.01,
    )
    strategy, message, message_type = get_threshold_guidance(threshold)

    st.caption(f"Current setting: **{strategy}** ({threshold:.0%} maximum predicted chance of default)")

    if message_type == "success":
        st.success(message)
    elif message_type == "info":
        st.info(message)
    elif message_type == "warning":
        st.warning(message)
    else:
        st.error(message)

st.write("")


# =========================================================
# APP CONTROLS
# =========================================================
with st.sidebar:
    st.header("⚙️ Application Settings")
    if st.button("🔄 Reset app state", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.divider()
    show_debug = st.checkbox("Show debug details", value=False)
    show_diagnostics = st.checkbox("Show diagnostics", value=True)


# =========================================================
# PREDICTION
# =========================================================
st.divider()
st.subheader("🚦 Run Risk Evaluation")

run_clicked = st.button(
    "🔍 Analyze Loan Application",
    disabled=(len(selected_keys) == 0),
    use_container_width=True,
    type="primary",
)

if run_clicked:
    user_inputs = gather_user_inputs(selected_keys)
    raw_row = build_raw_row(user_inputs)

    if show_debug:
        with st.expander("🛠️ Debug: Raw row sent to the model"):
            st.write(raw_row)

    try:
        result = predict_default_probability(raw_row, threshold=threshold)
        probability = result["default_probability"].iloc[0]

        risk_band, risk_emoji = get_risk_band(probability)
        investor_decision = get_investor_decision_text(probability, threshold)

        profit_info = estimate_profit_loss(user_inputs, probability)
        message_type, case_result_message = get_case_result_message(probability, threshold, risk_band)

        st.write("")

        # =====================================================
        # PRIMARY RESULT METRICS
        # =====================================================
        with st.container(border=True):
            st.subheader("📊 Risk Evaluation Results")

            st.progress(
                min(probability, 1.0),
                text=f"Predicted default probability: {probability:.1%}",
            )

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Predicted Default Probability", f"{probability:.1%}")
            with metric_col2:
                st.metric("Estimated Borrower Risk", f"{risk_emoji} {risk_band}")
            with metric_col3:
                st.metric("Investor Decision", investor_decision)

            if message_type == "success":
                st.success(case_result_message)
            else:
                st.warning(case_result_message)

        st.write("")

        # =====================================================
        # EXPECTED VALUE SECTION
        # =====================================================
        with st.container(border=True):
            st.subheader("💰 Estimated Financial Outcome")

            ev_col1, ev_col2, ev_col3 = st.columns(3)
            ev_col1.metric("Expected Interest Income", f"${profit_info['display_interest_income']:,.0f}")
            ev_col2.metric("Expected Credit Loss", f"${profit_info['display_credit_loss']:,.0f}")
            ev_col3.metric(
                "Expected Net Value",
                f"${profit_info['display_net']:,.0f}",
                delta=f"${profit_info['display_net']:,.0f}",
            )

            st.caption(
                f"These estimates are based on the entered loan amount, interest rate, predicted default "
                f"probability, and an assumed {profit_info['lgd']:.0%} loss if the borrower defaults."
            )

        st.write("")

        # =====================================================
        # SHAP / INTERPRETABILITY
        # =====================================================
        with st.container(border=True):
            st.subheader("🧠 Key Factors Influencing This Prediction")

            artifacts = load_artifacts()
            X_transformed = transform(raw_row, artifacts=artifacts)

            try:
                explainer = load_shap_explainer()
                shap_df = get_shap_explanation(X_transformed, explainer)

                st.caption(
                    "🔴 Red bars increase loan risk · 🟢 Green bars reduce loan risk"
                )

                fig = plot_shap_bar(shap_df)
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)

                with st.expander("📎 View exact feature impacts"):
                    display_df = shap_df.sort_values("abs_shap", ascending=False)[
                        ["feature", "feature_value", "shap_value"]
                    ].copy()
                    display_df.columns = ["Feature", "Entered Value", "SHAP Impact"]
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

            except Exception as exc:
                st.info("ℹ️ Detailed SHAP output is unavailable. Showing rule-based explanation instead.")

                if show_debug:
                    import traceback
                    st.warning(f"SHAP Debug: {exc}")
                    st.code(traceback.format_exc())

                for driver in get_top_drivers_rule_based(user_inputs):
                    st.markdown(f"- {driver}")

        st.write("")

        # =====================================================
        # DIAGNOSTICS
        # =====================================================
        if show_diagnostics:
            with st.container(border=True):
                st.subheader("🔎 Prediction Details")

                diagnostics_df = build_diagnostics_table(
                    user_inputs=user_inputs,
                    threshold=threshold,
                    probability=probability,
                    investor_decision=investor_decision,
                    risk_band=risk_band,
                )
                st.dataframe(diagnostics_df, use_container_width=True, hide_index=True)

    except ValueError as exc:
        st.error(f"⚠️ Prediction error: {exc}")
    except Exception as exc:
        st.error(f"⚠️ Unexpected application error: {exc}")
