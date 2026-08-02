# Final Model Selection

## Test Results

(2017–2018 for Experiment 1 candidates, evaluated on a consistent
pipeline; 2018 for the Experiment 2 candidate, evaluated on its own
correctly-preprocessed holdout — see note on comparability below)

| Model | Test AUC | Test KS | Test Brier | Top10% Lift |
|---|---|---|---|---|
| XGBoost — Modern (2016 train, with tradeline) | 0.7251 | 0.3287 | 0.2036 | 2.15x |
| XGBoost — With Tradeline (2007–2015 train) | 0.7106 | 0.3047 | 0.2175 | 2.04x |
| LightGBM — With Tradeline (2007–2015 train) | 0.7092 | 0.3048 | 0.2180 | 2.02x |

## A Mislabeling Bug Found and Corrected

An earlier version of this evaluation mislabeled the Experiment 2 model
as "XGBoost — Without Tradeline" — it is not; Experiment 2 was trained
*with* all 14 tradeline fields, on 2016 data where they are fully
populated. There was, in fact, no genuine "without tradeline" XGBoost
model loaded or evaluated in that version at all (`xgb_no_tradeline` was
referenced by two downstream cells — a calibration curve and a SHAP
explainer — but never actually loaded, meaning both cells would have
thrown `NameError` if run). This was caught by cross-referencing which
models were actually `joblib.load()`-ed at the top of the notebook against
what the results table claimed, and corrected before drawing any
conclusions from it.

## Reading the Results

XGBoost — Modern shows a noticeably higher test AUC (+0.014–0.016) than
the other two candidates. This is not attributable to test-set sampling
differences (Experiment 1's data is a 75% sample; Experiment 2's is the
full population) — a direct check, evaluating XGBoost — Modern on both a
75%-sampled test set and its own 100% unsampled test set, showed a
difference of only 0.0005 AUC. The gap is a real signal difference.

## Decision Criterion: Pipeline Robustness Over Raw Test Score

Despite the AUC gap, **XGBoost — With Tradeline (2007–2015) was selected
as the final production model.**

XGBoost — With Tradeline is the only candidate whose entire pipeline —
every imputation median, winsorization cap, and category encoding — is
fit once, consistently, across train, validation, and test, spanning
nine years (2007–2015) that include multiple credit cycles, most notably
the 2008 financial crisis.

XGBoost — Modern's higher AUC comes from a pipeline fit entirely on a
single year (2016). Its winsor caps, imputation statistics, and
rare-category list all reflect one year's borrower population and
economic conditions. That narrower foundation is more fragile to future
drift than a pipeline anchored across a much broader historical range —
a meaningful risk for a model expected to score future, unseen loan
vintages that the training pipeline has never conceptually encountered.

| Model | Overfitting Gap (Train − Val AUC) |
|---|---|
| XGBoost — With Tradeline | 0.024–0.026 (good) |
| LightGBM — With Tradeline | 0.024–0.034 |
| XGBoost — Modern | 0.036–0.037 (wider) |

XGBoost — Modern also showed a wider train/validation gap, consistent
with fitting a smaller, single-year training set more tightly —
reinforcing the same conclusion from a second angle.

**Decision:** a ~1.5-point AUC advantage was traded for a pipeline built
on far more historical breadth, judged more important for a model
expected to generalize to future loan vintages it has not seen examples
of during training.

## Known Limitations

- Test scores are somewhat lower than validation scores. Expected: many
  2017–2018 loans hadn't been outstanding long enough by the Dec 2018
  dataset cutoff for their final outcome to be fully known.
- Raw model probabilities are not well-calibrated — see
  `04_calibration.md` for the fix applied.
- ~0.71–0.73 AUC appears to be the practical ceiling for this dataset —
  tested across depths, learning rates, training windows, and tradeline
  inclusion, with no configuration pushing meaningfully past this range.
  Consistent with published benchmarks on this same dataset. Reflects the
  amount of signal available in LendingClub's disclosed at-approval-time
  features, not a modeling shortfall.
- Experiment 1's data is a 75% random subsample — a documented deviation
  from the full population, empirically shown not to materially affect
  results (see above).

## Final Model

- **Model file:** `xgb_with_tradeline.pkl` (raw) /
  `xgb_with_tradeline_calibrated.pkl` (calibrated, used in production)
- **Features file:** `xgb_with_tradeline_features.pkl`
