# Choosing the Final Model

## Testing on Data Never Seen Before

Three finalist models were evaluated on 2017–2018 loan data — data that
had been set aside from the very beginning and never used to make any
decisions about which model or settings to use.

| Model | Test Accuracy (AUC) | Test Error Separation (KS) |
|---|---|---|
| XGBoost, trained on 2016 only | 0.7251 | 0.3287 |
| XGBoost, trained on 2007–2015 | 0.7106 | 0.3047 |
| LightGBM, trained on 2007–2015 | 0.7092 | 0.3048 |

## Why the Higher-Scoring Model Wasn't Chosen

The model trained only on 2016 scored the highest on this final test.
However, it was **not** selected as the final model.

The reasoning: every part of that model's setup — how missing values
were filled in, how extreme values were capped, how categories were
encoded — was learned from just one year of data. That's a narrow
foundation. The model trained on nine years (2007–2015) has learned from
a much wider range of economic conditions, including the 2008 financial
crisis, making it more likely to hold up well when scoring future loans
under conditions it has actually seen some version of before.

To confirm this wasn't just intuition, overfitting was also checked:

| Model | Gap between training and validation accuracy |
|---|---|
| XGBoost, 2007–2015 | 0.024–0.026 (healthy) |
| LightGBM, 2007–2015 | 0.024–0.034 |
| XGBoost, 2016 only | 0.036–0.037 (wider) |

The 2016-only model also showed a larger gap between how well it did on
training data versus new data — a second, independent signal that it may
be fitting itself a bit too closely to that one year's specific
patterns.

**Decision:** a small (roughly 1.5 percentage point) accuracy advantage
was traded for a model built on a much broader, more representative
slice of history — judged the more reliable choice for scoring loans in
the future.

## What This Model Can and Can't Do Well

- Test-year results (2017–2018) are slightly understated, since many of
  those loans hadn't had time to fully play out by the end of the
  dataset.
- The model's raw probability outputs needed correction before they
  could be trusted as real percentages (see `04_calibration.md`).
- Across every configuration tried, accuracy consistently landed around
  71–73%. This matches results published by others working with this
  same public dataset — it reflects the real limits of what LendingClub's
  publicly available loan-approval data can predict, not a shortcoming
  in how the model was built.

## Final Model Files

- `xgb_with_tradeline.pkl` — the trained model
- `xgb_with_tradeline_calibrated.pkl` — the same model with corrected
  probability outputs (used for anything shown to an actual user)
- `xgb_with_tradeline_features.pkl` — the exact list of inputs the model expects
