# LendingClub Default Risk Prediction — Project Overview

## Objective

Predict borrower default risk at loan origination, using only information available at the time of application — no post-loan data, no leakage. Built as an end-to-end pipeline, not a notebook-only project:

**raw data → feature engineering → model training → evaluation → calibration → drift monitoring → business-impact analysis → interactive deployment**

---

## Dataset

- LendingClub accepted loans, 2007–2018 (~2.26M rows, 151 raw columns)
- Feature engineering used a 75% random sample for development speed
- Verified this sampling has no meaningful effect on results (0.0005 AUC difference vs. the full population)

---

## Approach

### Temporal validation, not random splits

Trained on 2007–2015, validated on 2016, tested on 2017–2018. A random split was deliberately avoided — borrower behavior and underwriting standards shifted substantially over this period (yearly default rate climbed from 13.75% in 2009 to 26.60% in 2017), and a random split would leak future information into training.

### Two training strategies compared head-to-head

| | Training Data | Notes |
|---|---|---|
| **Experiment 1** | 2007–2015 (9 years) | Includes the 2008 financial crisis. Bureau/tradeline fields dropped — structurally absent before 2015. |
| **Experiment 2** | 2016 only | Bureau tradeline fields fully populated. Tests whether a narrower, richer dataset beats a broader one. |

### Four models compared

Logistic Regression (interpretable baseline, VIF-cleaned), Random Forest, XGBoost, and LightGBM — all evaluated on the same held-out test set.

---

## Result

**Final model: XGBoost, trained on 2007–2015, with tradeline features included.**

| Metric | Score |
|---|---|
| Test AUC | 0.7106 |
| Test KS | 0.3047 |

Notably, Experiment 2 (2016-only training) scored *higher* on every test metric (AUC 0.7251) — but was **not** selected. The deciding factor was pipeline robustness, not raw test performance:

- Experiment 1's entire preprocessing pipeline (every imputation median, winsorization cap, category encoding) is fit once across nine years of history, spanning multiple credit cycles.
- Experiment 2's pipeline reflects only a single year's borrower population — a narrower, more fragile foundation for scoring future, unseen loan vintages.

This was a deliberate, documented tradeoff, not an oversight. Full reasoning in [`03_final_model_selection.md`](03_final_model_selection.md).

---

## Beyond the model: what makes this a full pipeline

- **Calibration** — the raw model's probabilities ran nearly double the true default rate (46% predicted vs. 26% actual). Isotonic regression corrected this, improving the Brier score from 0.2171 to 0.1734, with no change to the model's risk ranking (AUC unchanged).
- **Feature drift analysis** — systematic PSI analysis across all 75 continuous features uncovered two distinct structural data-introduction events in LendingClub's own reporting history (~2011 and ~2015, affecting 22 and 14 features respectively) — one-time schema changes, not gradual drift. Found and root-caused with hard evidence.
- **Business-impact analysis** — translated the model's AUC into an actual lending decision by simulating approve/reject outcomes across a range of risk thresholds, with dollar-estimated net impact. The "obvious" default threshold (0.5) would have been the *least* profitable choice available; a ~20% threshold was recommended instead.
- **Reproducible pipeline, not notebook artifacts** — `src/pipeline/` contains:
  - `preprocess.py` — single source of truth for all transformations, shared between training and inference
  - `train.py` — reproduces the final model end-to-end from raw CSV with one command
  - `predict.py` — scores new applications using the same preprocessing as training, eliminating train/serve mismatch by construction
- **Interactive deployment** — a Streamlit app where a user can enter any subset of ~79 loan attributes and get a calibrated risk score, an adjustable-threshold approve/reject recommendation, an estimated profit/loss impact, and a SHAP-based explanation of the features driving that specific prediction.

---

## Full Reports

| Report | Contents |
|---|---|
| [01_feature_engineering.md](01_feature_engineering.md) | Leakage checks, target construction, sentinel handling |
| [02_model_comparison.md](02_model_comparison.md) | LR vs. RF vs. XGBoost vs. LightGBM, Experiment 1 vs. Experiment 2 |
| [03_final_model_selection.md](03_final_model_selection.md) | Test results and rationale for selecting `xgb_with_tradeline` |
| [04_calibration.md](04_calibration.md) | Raw vs. calibrated Brier score and AUC |
| [05_feature_drift_analysis.md](05_feature_drift_analysis.md) | PSI findings and the two structural data events |
| [06_business_metrics.md](06_business_metrics.md) | Threshold analysis and profit/loss tradeoffs |
| [07_deployment.md](07_deployment.md) | Pipeline architecture, `app.py`, known limitations |
| [known_limitations.md](known_limitations.md) | Consolidated, single source of truth for all caveats |

---

## Known Limitations

Full detail in [`known_limitations.md`](known_limitations.md). Most notably:

- ~0.71–0.73 AUC appears to be the practical ceiling for this dataset — consistent across every configuration tested, and in line with published benchmarks
- The model's raw (uncalibrated) probabilities should never be used directly — use the calibrated output
- Roughly 15–20% of features carry no real signal for loans issued before ~2012, due to LendingClub's own reporting history

---

## Project Structure

```
src/
├── artifacts/            # fitted preprocessing objects (imputers, encoders, caps)
├── models/                # trained models (raw, calibrated) + their feature lists
├── pipeline/
│   ├── preprocess.py      # shared transformation logic
│   ├── train.py           # reproducible end-to-end training script
│   ├── predict.py         # inference entry point
│   └── monitor_drift.py   # PSI-based drift monitoring
tests/                     # smoke tests for predict.py and monitor_drift.py
notebooks/                 # exploratory work: FE, model comparison, evaluation
reports/                   # this documentation
app.py                     # Streamlit dashboard
```
