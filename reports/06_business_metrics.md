# Turning the Risk Score Into a Business Decision

## The Idea

A risk score on its own doesn't tell a lender what to actually do. This
analysis picks a cutoff point — approve loans below a certain risk
level, reject anything above it — and estimates what that decision would
actually cost or save, in dollar terms.

## Assumptions Used

- A typical performing loan earns about 13% interest.
- A defaulted loan loses about 60% of what was lent out.

These are reasonable, standard industry estimates — not numbers measured
directly from this dataset. A real lending team would refine these using
their own historical recovery data.

## Results

| Risk Cutoff | Share of Applicants Approved | Default Rate Among Approved | Estimated Net Impact |
|---|---|---|---|
| 15% | 32% | 10.5% | +$114 million |
| 20% | 42% | 12.8% | +$60 million |
| 25% | 56% | 15.8% | -$21 million |
| 30% | 65% | 17.7% | -$87 million |
| 50% | 94% | 24.2% | -$374 million |

## The Key Insight

Many classification tools default to a 50% cutoff without much thought
— it's the "obvious" halfway point. This analysis shows that, under
these assumptions, **50% is actually the worst possible choice on this
entire list** — it approves too many risky borrowers and would lose far
more money than a more careful cutoff.

## Recommendation

A cutoff of roughly **20%** balances the two competing costs reasonably
well: rejecting too many good borrowers wastes potential profit, while
approving too many risky ones creates real losses. This is the default
setting used in the live app (and can be adjusted by the user to see how
the tradeoff shifts).

## A Note on Precision

This analysis is meant to show *how* a risk score translates into a
business decision, not to serve as an exact profit forecast. The precise
dollar figures would shift if the interest-rate or loss assumptions were
different.
