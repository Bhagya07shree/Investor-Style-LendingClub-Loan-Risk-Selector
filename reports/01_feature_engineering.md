# Feature Engineering — Decisions and Reasoning

## 1. Data Ingestion

The raw dataset (`accepted_2007_to_2018Q4.csv`) contains ~2.26 million rows
and 151 columns — too large to load directly into memory on the development
machine (7.65 GB total RAM, frequently under 1.5 GB available). The CSV was
read in chunks of 50,000 rows, with a 75% random sample taken from each
chunk (fixed seed = 42), producing a working dataset of 1,695,526 rows.
Chunk-level sampling (rather than sampling the whole file at once) keeps
all loan vintages (2007–2018) proportionally represented.

Processed data was cached as Parquet rather than re-read from CSV each
session — faster loads, preserved dtypes, smaller footprint.

**Verified impact of the 75% sample:** a direct check (evaluating the same
trained model on both a 75%-sampled test set and a 100%, fully unsampled
test set drawn independently) showed only a 0.0005 AUC difference — the
sampling rate does not meaningfully distort results.

## 2. Leakage Columns Removed (63 total)

Nine categories of leakage were identified and dropped before any modeling:

| Category | Examples | Why |
|---|---|---|
| Identifiers | `id`, `member_id`, `url`, `policy_code` | No predictive value |
| High-cardinality text | `title`, `zip_code` | Noise / privacy-proxy risk |
| Redundant | `funded_amnt`, `funded_amnt_inv`, `num_sats`, `num_rev_tl_bal_gt_0` | Correlation > 0.97 with retained features |
| Post-loan payments | `out_prncp`, `total_pymnt`, `total_rec_int`, `recoveries`, `last_pymnt_amnt`, etc. | Only known after the loan has already run its course |
| Post-loan dates | `last_pymnt_d`, `next_pymnt_d`, `last_credit_pull_d` | Future information |
| Post-loan FICO | `last_fico_range_high/low` | Updated after origination |
| Settlement | `debt_settlement_flag`, `settlement_amount`, etc. (7 fields) | Only exist for defaulted loans |
| Hardship | `hardship_flag`, `hardship_type`, `hardship_dpd`, etc. (15 fields) | Mid-loan distress signals, not known at approval |
| Secondary applicant | `annual_inc_joint`, `sec_app_*` (16 fields) | 99%+ missing, joint-application-only |

Including any of these would have artificially inflated model performance
in a way that would not generalize to real-world, at-approval-time scoring.

## 3. Target Variable Construction

**Basel III-aligned binary default definition:**

| Loan Status | Label | Reason |
|---|---|---|
| Fully Paid | 0 | Completed successfully |
| Charged Off | 1 | Written off after 120+ days non-payment |
| Default | 1 | Meets regulatory default definition |
| Late (31–120 days) | 1 | Meets 90-day Basel III threshold |
| Current (fully matured) | 0 | Completed 100% of term |
| Current (not fully matured) | Excluded | Outcome not yet resolved |
| In Grace Period | Excluded | 1–15 days late — within normal grace window, outcome uncertain |
| Late (16–30 days) | Excluded | Self-curing is common but not guaranteed; labeling either way would bias one class |

**Handling "Current" loans — the maturity threshold problem.** Excluding
all "Current" loans biases the dataset toward older vintages, since recent
loans simply haven't had time to resolve. Two thresholds were tested:

- **First attempt — 67% of term elapsed:** recovered ~100,000 additional
  loans for 2016 alone. Rejected because it caused validation AUC to
  exceed training AUC — a sign of underfitting, indicating the recovered
  "partially matured" loans were introducing label noise rather than
  removing it.
- **Final decision — 100% of term elapsed:** a loan is only trusted as
  non-default once it has fully completed its term. Stricter, but avoids
  the noise from the 67% threshold.

**Remaining, unfixable limitation:** 2017–2018 loans cannot reach full
term completion before the dataset's Dec 2018 cutoff, so the test-set
default rate is measured on an immature population — test metrics should
be read as a lower/partial-information bound, not the loans' true
lifetime outcome.

Final class distribution: 78.75% non-default, 21.25% default.

## 4. Data Quality Fixes

- **`annual_inc` == 0** treated as missing (replaced with NaN).
- **Implausible income/loan-size combinations:** borrowers with
  `annual_inc < $5,000` AND `loan_amnt / annual_inc > 3` were flagged as
  unreliable and their income set to NaN (408 rows, 0.04% of data) —
  informed by finding loans like a $35,000 loan against a reported $4,177
  income, inconsistent with real underwriting practice.
- **`dti` == -1**: a known sentinel/placeholder, replaced with NaN.
- **`tot_hi_cred_lim` / `total_rev_hi_lim` == 9,999,999**: system
  placeholder for "no credit limit on file," replaced with NaN before
  winsorization (otherwise it would have massively distorted the 99th
  percentile cap).

## 5. Informative Missingness Flags

Eight binary flags were created **before** imputation, to preserve signal
that would otherwise be destroyed by filling missing values:

| Flag | Source Column | Meaning |
|---|---|---|
| `ever_delinquent` | `mths_since_last_delinq` | Ever delinquent |
| `has_public_record` | `mths_since_last_record` | Public record exists |
| `has_bc_delinquency` | `mths_since_recent_bc_dlq` | Bankcard delinquency on record |
| `has_major_derog` | `mths_since_last_major_derog` | Major derogatory mark |
| `has_installment_accts` | `il_util` | Has installment accounts |
| `has_rcnt_il` | `mths_since_rcnt_il` | Recent installment account |
| `has_revol_delinq` | `mths_since_recent_revol_delinq` | Revolving delinquency |
| `has_bankcard` | `bc_util` | Has bankcard |

**Why this matters:** for the five `mths_since_*` columns, missing does
not mean "unknown" — it means the event never happened. Imputing with a
median would falsely suggest every "never delinquent" borrower had a
delinquency some months ago, destroying the most important signal in the
column: the absence of a negative event entirely.

**Sentinel value strategy (999).** Rather than median-imputing the five
`mths_since_*` columns, missing values were filled with 999 — a
deliberately extreme value, so trees can cleanly split "never happened"
(999) from "happened recently" (low value) from "happened long ago" (high
value). This value only works for tree models; for Logistic Regression,
these five columns are dropped entirely, relying on the binary flags
instead (see Section 8).

## 6. Structural Missingness — Bureau Tradeline Features

Fourteen columns (`open_acc_6m`, `il_util`, `all_util`,
`mths_since_rcnt_il`, etc.) showed ~97–98% missingness in the 2007–2015
training window. Investigation (grouping missingness by issue year)
confirmed this is **structural, not random**: these fields were not
collected by LendingClub before ~2015 (see `05_feature_drift_analysis.md`
for the full year-by-year confirmation, which also uncovered a second,
earlier structural event around 2011 affecting 22 other features).

**Modeling decision:**
- **Logistic Regression:** tradeline columns excluded entirely — training
  a linear model mostly on synthetic (imputed) values while evaluating on
  real values would be unsound.
- **Tree models (XGBoost/LightGBM):** tradeline columns retained with NaN
  — trees handle missing values natively, and the missingness itself may
  carry temporal signal.

## 7. Engineered Features

| Feature | Formula | Rationale |
|---|---|---|
| `fico_score` | midpoint of `fico_range_low`/`fico_range_high` | One clean score instead of two raw columns |
| `pti_ratio` | `installment / (annual_inc / 12)` | Monthly payment as % of income |
| `loan_to_income` | `loan_amnt / annual_inc` | Loan size relative to income |
| `overall_util` | `tot_cur_bal / (tot_hi_cred_lim + 1)` | Overall credit utilization |
| `credit_age_yrs` | `(issue_d - earliest_cr_line).days / 365` | Years of credit history at issuance |
| `issue_year` | `issue_d.dt.year` | Drives the temporal train/val/test split only — dropped from all final feature sets afterward |

`emp_title` was explored as a source for a grouped `emp_category` feature
(via rule-based text normalization), but this did not improve AUC and
introduced noise — not retained in the final feature set.

## 8. Redundant Feature Removal — Full Correlation Scan

A systematic correlation scan across the full ~150-column `X_train`
(threshold: |correlation| > 0.85) found 13 highly correlated pairs,
resolved as follows:

**Universal drops (applied to X_train directly, affecting all models):**
- `int_rate` (0.98 corr with `sub_grade`) — kept `sub_grade` as more
  interpretable/policy-driven. *Caveat: `int_rate` was LightGBM's single
  most important feature (importance 450) — this tradeoff was made
  deliberately, accepting a small predictive cost for interpretability.*
- `tot_cur_bal`, `tot_hi_cred_lim` (0.96/0.93 corr with `avg_cur_bal`) —
  kept `avg_cur_bal` (appeared in both models' top-20 importance; the
  other two did not).
- `total_bal_il` (0.96 corr with `total_il_high_credit_limit`) — kept the
  latter (far less missingness: ~5% vs. ~59%).
- `has_rcnt_il` (0.94 corr with `has_installment_accts`) — both flags
  derived from the same 2015+ structural introduction.
- `pti_ratio` (0.94 corr with `loan_to_income`) — kept `loan_to_income`.

**Sentinel/flag pairs (5 pairs, correlation ≈ 1.00) — treated oppositely
per model type:**
- `ever_delinquent`, `has_public_record`, `has_bc_delinquency`,
  `has_major_derog`, `has_revol_delinq` each correlate ~1.00 with their
  corresponding `mths_since_*` sentinel column (structural: the flag was
  literally derived from the sentinel's missingness).
- **Logistic Regression:** drops the 5 sentinel columns, keeps the 5
  flags (avoids the 999 sentinel distorting a linear coefficient).
- **Tree models:** drops the 5 flags, keeps the 5 sentinels (the sentinel
  subsumes the flag's information — a tree can split on `<999` vs. `==999`
  for free, plus gets recency detail the flag discards).

## 9. Encoding

- **`term`**: extracted numeric value ("36 months" → 36).
- **`initial_list_status`**: binary map (`f`→0, `w`→1).
- **`sub_grade`**: ordinal, A1–G5 → 1–35 (preserves natural risk ordering).
- **`verification_status`**: ordinal, but the order was *data-validated*,
  not assumed — checked actual default rates (Not Verified 13.7% 
  Source Verified 20.1% < Verified 21.5%, confirming LendingClub's known
  selection bias: verification is triggered by risk signals, so verified
  applicants are not "safer," they're flagged for scrutiny).
- **`home_ownership`**: `ANY`/`NONE` merged into `OTHER` before one-hot
  encoding (extremely rare categories — 3 and 38 rows respectively out of
  600K+). *A bug was found and fixed where this merge step existed only
  as an unused variable in one script run without the actual `.replace()`
  being applied — silently reintroducing `ANY`/`NONE` as separate
  categories (5 dummies instead of 3). Caught via a systematic
  `train_xgb.shape` mismatch (145 vs. expected 143 columns) and confirmed
  the model's actually-saved `ohe_encoder.pkl` had the correct, merged
  4-category version — meaning the deployed model was correct throughout,
  only a later notebook re-run had regressed.*
- **`addr_state`**: states with < 500 training rows merged into
  `OTHER_STATE` before encoding (fit independently per experiment/training
  window — legitimately produces different dummy counts across
  Experiment 1 vs. 2, since different years have different volume
  distributions across states).
- **`purpose`, `addr_state`, `home_ownership`**: one-hot encoded via a
  single `OneHotEncoder(handle_unknown='ignore', drop='first')`, fit once
  on training data and reused (never re-fit) for validation/test/inference.

## 10. VIF Analysis (Logistic Regression only)

A single-pass VIF check (threshold = 10, iterative — drop highest VIF,
recheck, repeat) on the LR-specific feature set (post correlation-drops,
sentinel/tradeline exclusions) found:

- Round 1: `num_rev_accts` (VIF 17.12), `pub_rec` (VIF 11.44)
- Remaining moderate-band features (5–10) were reviewed and left in place
  — VIF 5–10 is "monitor," not "drop," per standard practice.

## 11. Imputation

- **Numeric columns** (excluding flags, sentinel columns, tradeline
  columns): median imputation, fit on training data only.
- **Categorical columns**: mode imputation, fit on training data only.
- Both fitted imputers are saved as artifacts and reused identically at
  inference time — never re-fit on validation, test, or new prediction data.
