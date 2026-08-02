# Model Comparison

## Models Evaluated

| Model | Feature Set | Purpose |
|---|---|---|
| Logistic Regression | Sentinel/tradeline dropped, correlation + VIF cleaned, scaled | Interpretable baseline |
| Random Forest | Tradeline included | Non-linear check |
| XGBoost | Tested with AND without tradeline | Primary candidate |
| LightGBM | Tested with AND without tradeline | Secondary candidate |

## Logistic Regression

Final feature set built in a single consolidated pass (not iterative
rounds — an earlier iterative approach, dropping features across LR1 →
LR2, showed identical AUC before and after, confirming the removed
columns were redundant, not informative — so a one-pass approach was
adopted going forward):

1. Correlation-based drops (Section 8 above, LR-specific:
   `int_rate`, `tot_cur_bal`, `tot_hi_cred_lim`, `total_bal_il`,
   `has_rcnt_il`, `pti_ratio`)
2. Structural drops: 5 sentinel columns, 14 tradeline columns,
   `issue_year`, `delinq_amnt` (zero variance)
3. Single VIF pass: `num_rev_accts`, `pub_rec`

**Result:** Val ROC-AUC 0.7152, Val KS 0.3100, Val Brier 0.2222,
Train/Val gap 0.0106 (good fit, no overfitting).

Not deployed — served as the benchmark proving tree models' non-linear
advantage was real but modest.

## Random Forest

Config: `n_estimators=200`, `class_weight='balanced'`,
`max_features='sqrt'`, `min_samples_leaf=50`.

| max_depth | Train AUC | Val AUC | Train/Val Gap |
|---|---|---|---|
| Unlimited | 0.8182 | 0.7175 | 0.1007 (overfitting) |
| 8 | 0.7271 | 0.7109 | 0.0163 (underfitting) |
| 12 | 0.7600 | 0.7155 | 0.0445 |

**Finding:** no depth setting meaningfully outperformed the LR baseline
(0.7151) on validation. Conclusion: the feature-target relationship in
this dataset is close to linear — the top predictors (sub_grade/int_rate,
FICO, DTI) already have a near-monotonic relationship with default risk,
exactly the setting where trees have little edge over a well-regularized
linear model.

**A fairness issue was identified and corrected:** RF was initially
trained *with* `issue_year` included, while XGBoost/LightGBM were trained
*without* it — an inconsistent comparison, since `issue_year` carries real
drift signal (yearly default rate climbed from 13.75% to 26.60% across
the training window). RF was not pursued further given it already
underperformed even with this advantage, so it was not retrained, but the
inconsistency is noted here for the record.

## XGBoost / LightGBM — With vs. Without Tradeline (Experiment 1 internal comparison)

| Model | Features | Val ROC-AUC | Val KS | Train/Val Gap |
|---|---|---|---|---|
| XGBoost (with tradeline) | 143 | 0.7255 | 0.3270 | 0.0304 (mild overfit) |
| XGBoost (without tradeline) | 129 | 0.7237 | 0.3240 | 0.0259 (good fit) |
| LightGBM (with tradeline) | 143 | 0.7236 | 0.3222 | 0.0337 |

**With tradeline wins on every predictive metric** (AUC, KS, Brier), at
the cost of a marginally wider train/val gap — still within acceptable
range. This became the Experiment 1 candidate carried to test evaluation.

## Experiment 1 vs. Experiment 2 (validation stage)

| Metric | Exp 1 (no tradeline, 2007–2015) | Exp 2 (with tradeline, 2016) | Difference |
|---|---|---|---|
| Val ROC-AUC | 0.7254 | 0.7253 | -0.0001 |
| Val KS | 0.3267 | 0.3261 | -0.0006 |
| Val Brier | 0.2145 | 0.2093 | -0.0052 |
| Train/Val Gap | 0.0322 (good) | 0.0359 (mild overfit) | +0.0037 |
| Training rows | 623,267 | 297,651 | -52% |

Both models scored almost identically on validation despite Experiment 2
using half the training data. This decided which candidates moved to
test evaluation — not the final model (that decision is made only on the
held-out test set; see `03_final_model_selection.md`).

## Feature Redundancy Fixes Applied to the Final Tree Feature Set

The 5 redundant flag columns (`ever_delinquent`, `has_public_record`,
`has_bc_delinquency`, `has_major_derog`, `has_revol_delinq`) were dropped
from the tree feature set, on top of `issue_year` and `delinq_amnt` —
bringing the final "with tradeline" tree feature count from 148 to 143.
This fix was applied consistently across Experiment 1 and Experiment 2
(the latter required restoring `has_rcnt_il`, which had been mistakenly
dropped in an earlier version based on a since-reversed decision that
this drop should be LR-only, not universal).

## Summary

XGBoost and LightGBM meaningfully outperformed Logistic Regression on
non-linear pattern capture, while LR remained a strong, interpretable
benchmark. XGBoost consistently edged out LightGBM. Tradeline features
added real signal for tree models. Random Forest showed no advantage over
the linear baseline and was not pursued further.
