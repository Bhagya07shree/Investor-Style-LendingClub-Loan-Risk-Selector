# Deployment — Pipeline Architecture, App, and Known Limitations

## 1. Pipeline Architecture

The trained model is served through a single, reproducible preprocessing pipeline shared between training and inference — this is deliberate, and central to avoiding train/serve mismatch.

| File | Role |
|---|---|
| `src/pipeline/preprocess.py` | Single source of truth for every transformation (date parsing, feature engineering, capping, imputation, encoding). Used identically by both `train.py` and `predict.py`. |
| `src/pipeline/train.py` | Reproduces the final model end-to-end from raw CSV with one command. |
| `src/pipeline/predict.py` | Inference entry point — loads the saved model and artifacts, scores new applications using the exact same preprocessing logic as training. |
| `src/pipeline/monitor_drift.py` | PSI-based drift monitoring (see `05_feature_drift_analysis.md`). |
| `src/artifacts/` | All fitted preprocessing objects — imputers, encoder, winsorization caps, category mappings — fit once on training data and only ever *applied*, never re-fit, at inference time. |

**Why this matters:** because `preprocess.py` is imported by both training and inference, there is no risk of the two silently drifting apart over time (e.g., a median calculated one way during training and a slightly different way during inference). Any change to preprocessing logic automatically applies to both.

---

## 2. The Application (`app.py`)

A Streamlit app (*"Investor-Style LendingClub Loan Risk Selector"*) that lets a user simulate evaluating a real loan application, using any subset of ~79 available borrower/loan attributes.

**Core workflow:**
1. Search and select which fields to provide (not all ~79 are required)
2. Enter and save values field by field, with the ability to revisit and edit any saved field
3. Set a personal maximum-risk threshold via a slider
4. Run the evaluation to get:
   - Calibrated default probability and risk band (Low / Moderate / High / Very High)
   - An investment recommendation relative to the chosen threshold
   - An estimated financial outcome (expected interest income vs. expected credit loss, using a configurable Loss Given Default assumption)
   - A SHAP-based explanation of the specific features driving that prediction, with a rule-based fallback if SHAP is unavailable

**Example profiles** (Low Risk, Balanced Yield, Aggressive Risk) are included so the model's behavior can be explored quickly without manually entering every field.

---

## 3. Notable Engineering Issues Resolved During Deployment

Documented here because each reflects a real class of bug worth knowing about if extending the app:

- **Stale multiselect state:** narrowing the field search box while items were selected but not yet confirmed could crash the app (`StreamlitAPIException`) — Streamlit doesn't allow a widget's stored selection to fall outside its current options list. Fixed by filtering stale selections out before the widget renders.
- **Fields silently becoming uneditable:** an earlier version of the Save/Edit logic excluded any field already present in `user_inputs` from getting an input widget — which meant fields pre-filled by an example profile lost their editable widgets entirely. Fixed by always rendering a widget for every currently selected field, and gating "saved" status on an explicit user action (clicking Save) rather than incidental session-state presence.
- **`TypeError: unhashable type: 'list'`:** a batch "Save Inputs" button attempted to pass a list of keys into a function designed to confirm one key at a time. Fixed by looping over the list and calling the single-key function per iteration.
- **SHAP/XGBoost version incompatibility:** `shap.TreeExplainer` failed with `ValueError: could not convert string to float: '[5E-1]'`. Root cause: newer XGBoost versions can serialize the model's `base_score` as a bracketed string (`"[5E-1]"`), which older `shap` versions can't parse. Resolved by upgrading `shap` (preferred), with a manual config-patching workaround available as a fallback if the version can't be changed.


---

## 4. Known Limitations

See [`know-limitations.md`](known-limitations.md) for the full, consolidated list. Most relevant to deployment specifically:

- The model's **raw, uncalibrated probabilities should never be used directly** — only the calibrated output is meaningful as an actual probability (see `04_calibration.md`).
- **~0.71–0.73 AUC is the practical ceiling** for this dataset across every configuration tested — the app should be framed as a decision-support tool, not a precise risk score.
- **Bureau tradeline fields carry no real signal for loans issued before ~2015** due to LendingClub's own reporting history (see `05_feature_drift_analysis.md`) — a live application wouldn't have this limitation, but it's a structural artifact of the historical training data.
