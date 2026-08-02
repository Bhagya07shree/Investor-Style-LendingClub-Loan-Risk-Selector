# Comparing the Models

## Models Tested

Four models were trained and compared:

| Model | What it's good at |
|---|---|
| Logistic Regression | Simple, interpretable, easy to explain to a non-technical audience |
| Random Forest | Can capture non-linear patterns |
| XGBoost | Strong performance, handles missing data natively |
| LightGBM | Similar to XGBoost, generally faster to train |

## The Simple Model (Logistic Regression)

Before training, redundant and highly correlated features were removed
in a single pass — first by checking correlations across all features,
then by checking for more subtle multicollinearity (VIF analysis). An
earlier attempt did this cleanup in stages and confirmed accuracy stayed
identical before and after — proof the removed columns weren't
contributing anything real, just overlap with other features.

**Result:** Validation accuracy (AUC) of 0.7152, with no signs of
overfitting. This became the benchmark every other model needed to beat.

## Random Forest

Several settings were tested. Deeper trees fit the training data almost
perfectly but performed no better on new data — a classic sign of
overfitting. Shallower trees avoided overfitting but also didn't beat the
Logistic Regression benchmark at any depth tested.

**Conclusion:** for this dataset, the relationship between a borrower's
details and their default risk is close to linear — the strongest
predictors (loan grade, credit score, debt-to-income ratio) already
behave in a fairly straightforward, consistent way. This is exactly the
situation where a simple model and a complex one perform similarly.
Random Forest was not pursued further.

## XGBoost and LightGBM

Both were tested with and without the credit-bureau fields that were
missing before 2015 ("tradeline" columns). Including them gave a small
but consistent improvement across every metric checked (accuracy, error
separation, calibration).

XGBoost slightly outperformed LightGBM throughout, and including the
tradeline fields was worth the modest added complexity.

## Comparing Two Training Strategies

Two different approaches were tested side by side:

| | Trained on 2007–2015 | Trained on 2016 only |
|---|---|---|
| Validation accuracy | 0.7254 | 0.7253 |
| Training data size | 623,267 loans | 297,651 loans |

Despite using less than half the data, the 2016-only version performed
just as well on validation. This result decided which two candidates
would move forward to the final, held-out test — it did not decide the
winner. That decision came later, after testing on data neither model
had ever seen (see `03_final_model_selection.md`).

## Summary

Tree-based models (particularly XGBoost) modestly outperformed the
simpler Logistic Regression model, confirming there is some real
non-linear signal in the data — but the gap was smaller than might be
expected, reinforcing that the relationship between borrower attributes
and default risk is largely straightforward. Random Forest showed no
advantage over the simpler baseline.
