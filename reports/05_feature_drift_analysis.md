# Checking Whether the Data Changes Over Time

## Why This Matters

A model is only as good as its assumption that the future will look
somewhat like the past. If the kinds of loans coming in start looking
very different from what the model was trained on, its predictions
become less trustworthy. This check compares each feature's typical
values year by year, to catch that kind of shift early.

## An Early Mistake, Caught and Fixed

The first attempt at this check produced some extremely large, strange
numbers for a handful of columns. Investigating further showed the
problem: those particular columns were almost entirely one single value
in the early years (essentially "on" or "off" with barely any variation),
which breaks the standard drift-measuring formula. The fix was to treat
these columns differently — simple, direct percentage comparisons
instead of the more complex method used for continuous numbers.

## Finding 1: A Data Collection Change Around 2011

22 separate features were found to be **completely absent** for every
single loan issued before 2011 — not occasionally missing, but 100%
absent, every time. Starting in 2011, these fields began being reported,
and by 2013 they were essentially fully available.

These include several credit-balance and account-activity fields (things
like total current balance, number of active accounts, and account age
details).

**What this means in practice:** for roughly the earliest 4 of 9
training years, these 22 features carried no real information — only a
filled-in placeholder value. Since loan volume grew a great deal each
year, though, this only affects a small share of the *total* training
data by row count, even though it spans nearly half the years. This
likely explains why it didn't noticeably hurt the model's performance,
but it's a useful thing to know when interpreting which features the
model relies on most.

## Finding 2: A Second Data Collection Change Around 2015

Fourteen additional credit-bureau fields were found to follow the same
pattern, starting several years later, around 2015 — confirming
LendingClub expanded what data it collected on two separate occasions,
not just once.

## Finding 3: A Genuine Change in Lending Practice

One feature — whether a borrower's income was verified — showed a
steady, real shift over time, not a sudden data-collection change.
LendingClub appears to have increased how often it verified applicants'
income as the years went on. This reflects an actual change in business
practice, not a data quality issue.

## Finding 4: A Discontinued Loan Category

Loans listed for "wedding" purposes dropped from about 2.4% of all loans
down to exactly zero starting in 2014 — confirming LendingClub stopped
offering this loan category. This matches a separate, independent
finding from earlier in the project (a mismatch in how many loan
purposes appeared in different training periods), giving extra
confidence in the result.

## Ongoing Monitoring

The same drift-checking method was built into a small, separate,
reusable tool that can be run periodically against new incoming loan
data, to catch early warning signs if the model starts seeing
significantly different kinds of applicants than it was trained on. It
was tested using older data as a stand-in and correctly flagged the same
2011 data-collection pattern already identified above — confirming it
works as intended.
