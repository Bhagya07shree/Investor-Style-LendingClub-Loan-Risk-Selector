# Fixing the Model's Probability Outputs

## The Problem

The model was intentionally trained to be extra sensitive to catching
defaults, to make up for the fact that most loans in the data don't
default (about 79% don't, 21% do). This is a reasonable choice for
ranking risky vs. safe borrowers — but it comes with a side effect: the
raw probability numbers the model produces run much higher than reality.

On the test set, the model's predictions averaged **46%** estimated
default risk — but the real default rate was only **26%**. The model was
still correctly identifying who was riskier than whom, but the actual
numbers it gave couldn't be trusted at face value.

## The Fix

A standard correction technique (isotonic calibration) was applied on
top of the existing model. This doesn't change how the model ranks
borrowers — it simply adjusts the final probability number to better
reflect real-world outcomes.

## Result

| | Before | After |
|---|---|---|
| Average predicted risk | 46.1% | 24.3% |
| Actual default rate | 26.2% | 26.2% |
| Accuracy of probability values | Poor | Good |
| Risk ranking ability (AUC) | 0.7106 | 0.7104 (unchanged) |

The predicted numbers now sit much closer to reality, while the model's
ability to correctly rank risky vs. safe borrowers stayed exactly the
same — confirming the fix only touched what it was meant to.

## How This Is Used

- The uncorrected model is only appropriate for pure ranking tasks
  (e.g., "flag the riskiest 10% of applicants").
- The corrected model is used anywhere an actual risk percentage is
  shown to a person or used in a calculation — this is the version used
  in the live app and in any business-impact estimates.
