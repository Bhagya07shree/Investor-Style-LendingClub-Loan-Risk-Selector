# Known Limitations

This file consolidates the main caveats of the LendingClub risk-ranking project. Other reports should link back here instead of repeating these limitations separately.

---

## 1. Predictive Accuracy Has a Natural Ceiling

Across the main model experiments, validation and test ROC-AUC stayed within a relatively narrow range.

From the stable training notebook:

| Model | Validation ROC-AUC |
|---|---:|
| Logistic Regression | 0.7151 |
| Random Forest | 0.7109 |
| XGBoost with tradeline features | 0.7237 |
| LightGBM with tradeline features | 0.7228 |
| XGBoost without tradeline features | 0.7255 |
| LightGBM without tradeline features | 0.7234 |

From final model evaluation:

| Model | Test ROC-AUC |
|---|---:|
| XGBoost with tradeline features | 0.7106 |
| LightGBM with tradeline features | 0.7092 |
| XGBoost modern experiment | 0.7251 |

**Takeaway:** the model is useful for ranking loans by relative risk, but it should not be treated as a high-certainty predictor of individual borrower outcomes.

---

## 2. Raw Model Scores Were Not Calibrated

The raw XGBoost model ranked risk reasonably well, but its raw probability scores were too high.

From the final evaluation notebook:

| Score Type | Mean Predicted Probability | Actual Test Default Rate |
|---|---:|---:|
| Raw model | 0.4612 | 0.2623 |
| Calibrated model | 0.2427 | 0.2623 |

Calibration improved probability reliability:

| Model Output | Brier Score | ROC-AUC |
|---|---:|---:|
| Raw model | 0.2171 | 0.7106 |
| Calibrated model | 0.1734 | 0.7104 |

**Takeaway:** the raw model score should not be displayed as a default probability. The app should use the calibrated output for probability-style communication, while SHAP can still explain the underlying XGBoost model directionally.

---

## 3. Some Tradeline Features Have Historical Missingness

Several credit-bureau tradeline fields were introduced later in LendingClub's data history. In the feature engineering notebook, these fields showed very high missingness in older years, especially before 2015.

The final tradeline group contains 14 columns, including fields such as:

```text
open_acc_6m
open_rv_12m
open_rv_24m
open_il_12m
open_il_24m
open_act_il
total_bal_il
inq_fi
total_cu_tl
inq_last_12m
max_bal_bc
all_util
mths_since_rcnt_il
il_util
```

Examples of training-set missingness from the feature engineering output:

| Feature | Train Missing % |
|---|---:|
| il_util | 98.01% |
| mths_since_rcnt_il | 97.77% |
| open_rv_12m | 97.71% |
| all_util | 97.71% |

**Takeaway:** these columns contain schema-driven missingness, not ordinary borrower-level missingness. This affects older training vintages and must be considered when interpreting feature importance and model stability.

---

## 4. Recent Test Loans Are Partially Right-Censored

The dataset ends at the end of 2018, so some loans issued close to that point may not have had enough time to fully resolve.

From the modern experiment split:

| Split | Year | Rows | Default Rate |
|---|---:|---:|---:|
| Train | 2016 | 297,651 | 24.46% |
| Validation | 2017 | 177,325 | 26.60% |
| Test | 2018 | 63,539 | 25.33% |

The feature engineering notebook used a conservative rule for `Current` loans: only loans that completed 100% of their term were recovered as non-default. In the stable feature engineering run, only 55 mature `Current` loans were recovered.

**Takeaway:** final test results should be interpreted as performance on the available observed outcomes, not as a perfect lifetime-default measurement for every recent loan.

---

## 5. Calibration Improves Probabilities but Does Not Change Ranking Much

Calibration corrected the probability scale, but ranking performance stayed almost unchanged.

From final evaluation:

| Output | ROC-AUC | Brier Score |
|---|---:|---:|
| Raw XGBoost | 0.7106 | 0.2171 |
| Calibrated XGBoost | 0.7104 | 0.1734 |

**Takeaway:** calibration makes the displayed probability more realistic, but it does not materially change which loans are ranked as riskier or safer.

---

## 6. The Modern Experiment Scored Higher but Used a Narrower Training Window

The 2016-only modern experiment achieved the best final test ROC-AUC:

| Model | Test ROC-AUC | Test PR-AUC | Test Brier | Top 10% Lift |
|---|---:|---:|---:|---:|
| XGBoost modern | 0.7251 | 0.4570 | 0.2036 | 2.1478 |
| XGBoost with tradeline | 0.7106 | 0.4503 | 0.2171 | 2.0345 |
| LightGBM with tradeline | 0.7092 | 0.4473 | 0.2180 | 2.0162 |

However, the modern model was trained only on 2016 data, while the stable model was trained on a broader historical window.

**Takeaway:** the modern model gives stronger test metrics, but the broader-window model may be easier to justify as a more conservative production choice. This is a model-governance tradeoff, not just a leaderboard decision.

---

## 7. Sampling Was Used During Development

The project used a 75% chunk-based sample of the original LendingClub dataset to reduce runtime and memory usage during experimentation. This sampling strategy preserved coverage across loan vintages because each CSV chunk was sampled using a fixed random seed.

To check whether sampling materially changed the result, the final model was compared against a larger/full-data run. The difference in ROC-AUC was about **0.005**, which is small enough that the sampled workflow was considered acceptable for development.

**Takeaway:** sampling made experimentation practical and did not materially change the model conclusion, but final production retraining should still be done on the full dataset when compute resources allow.
