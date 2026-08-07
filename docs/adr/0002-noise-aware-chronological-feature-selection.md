# 2. PFISelector measures importance chronologically, above the permutation noise, and per cyclical pair

- **Status:** accepted
- **Date:** 2026-08-06
- **Phase:** 1c (feature-selection correction)
- **Supersedes the threshold decided in:** [ADR 0001](0001-feature-selection-threshold.md)

## Context

ADR 0001 replaced `PFISelector`'s inherited 75th-percentile cut with an absolute
`importance > 0`, because a quantile ties a feature's fate to how many columns
the weather provider happens to deliver. It closed with three known follow-ups,
all recorded as open points in chapter 6 of the Umsetzungsplan:

- **A defect, not a preference.** `PFISelector.fit` split with
  `train_test_split(..., test_size=0.1, random_state=42)`, which shuffles by
  default. On hourly data, neighbouring rows are strongly autocorrelated, so
  every test row had its neighbours in the training half. Importances measured
  that way are systematically flattered.
- **A noise-aware threshold.** `permutation_importance` returns
  `importances_std` alongside the mean at no extra cost. A mean smaller than the
  spread of its own repeats is not evidence of anything.
- **Everything else** — Boruta, `SequentialFeatureSelector`, dropping the
  selector entirely — needs data from more than one plant and stays open.

A third problem surfaced while measuring the first two. The selector treats the
`sin` and `cos` columns of one cyclical encoding as independent features and
will keep one without the other. That already happens on `main`
(`time__time_month_sin` is kept, `time__time_month_cos` is dropped). A `cos`
without its `sin` is ambiguous: it cannot distinguish morning from afternoon,
nor east from west.

## Measurements

Reference dataset, 9,580 hourly rows, 720-row holdout, no hyperparameter
tuning, all variants on one machine — the same protocol as ADR 0001. Frozen
baseline for comparison: **620.88 Wh MAE, R² 0.8859**.

| Split | Threshold | Pairs | MAE (Wh) | R² | Features kept |
|---|---|---|---|---|---|
| shuffled | `> 0` | independent | 641.54 | 0.8831 | 24 |
| chronological | `> 0` | independent | 637.39 | 0.8848 | 19 |
| chronological | `mean − 1·std > 0` | independent | 628.91 | 0.8850 | 17 |
| chronological | `mean − 2·std > 0` | independent | 639.22 | 0.8815 | 13 |
| chronological | `mean − 3·std > 0` | independent | 631.60 | 0.8885 | 8 |
| shuffled | `mean − 2·std > 0` | independent | 646.73 | 0.8769 | 16 |
| chronological | `> 0` | grouped | 630.85 | 0.8853 | 21 |
| **chronological** | **`mean − 1·std > 0`** | **grouped** | **607.82** | **0.8921** | **20** |
| chronological | `mean − 2·std > 0` | grouped | 631.40 | 0.8844 | 15 |
| — | no selection at all | — | 638.23 | 0.8836 | 28 |

Two things are worth reading out of this table, and one thing is not.

**The split fix shows up in the feature count, not in the metric.** Going
chronological drops seven features that the shuffled split had kept, among them
`time__time_dst` and `sun__time_daylight` — both near-constant over long
stretches and therefore the easiest of all to reconstruct from a neighbouring
row. That is the defect made visible.

**Grouping cyclical pairs helps at every threshold tested** — 637.39 → 630.85
at `k=0`, 628.91 → 607.82 at `k=1`, 639.22 → 631.40 at `k=2`. Three independent
values of `k` moving the same direction is a stronger signal than any single
number in the table.

**The value of `k` cannot be read out of this table.** MAE is not monotone in
`k` (628.91, 639.22, 631.60) and every variant sits within about 3% of every
other. Picking the row with the lowest MAE would be tuning the threshold on the
holdout that Phase 1c is accepted against — precisely what ADR 0001 rejected
when it declined percentile 50.

## Decision

1. **Measure importance on the most recent tenth of the rows.** The selector
   documents that callers pass rows in chronological order; `Forecaster.train`
   already sorts by the time column before fitting.
2. **Keep a feature when `mean − 1·std > 0`.** `k = 1` is chosen a priori, not
   from the table: `importances_std` is the spread across the `n_repeats`
   permutations, i.e. the run-to-run noise of the measurement itself, so
   `k = 1` discards exactly those features whose measured importance is smaller
   than the noise of measuring it. A larger `k` discards features that are
   measurably positive, which is a stronger claim than "this is noise" and one
   this data cannot support.
3. **Keep the `sin` and `cos` halves of a cyclical encoding together**, kept
   when either half clears the threshold. Grouping is by column-name suffix,
   which is what the selector can see: it receives the transformed frame, not
   the encoders that produced it.

The fallback from ADR 0001 stands: if nothing clears the bar, keep everything.

## Consequences

Every persisted model is invalidated. `feature_schema_version` does not cover
this — the feature vocabulary in `pvlearn.schema` is unchanged, only the way
those columns are assembled into a model. A new `pipeline_version` in the model
metadata carries that, compared in `ModelMetadata.raise_on_mismatch` exactly
like the existing fields. It defaults to `1` so that sidecars written before the
field existed parse and are then rejected with a precise reason, rather than
being reported as unreadable metadata.

Forecast quality on the reference dataset lands at 607.82 Wh MAE and R² 0.8921,
better than the frozen baseline's 620.88 Wh and 0.8859. That confirms the
decision does not regress; it does not establish that `k = 1` is optimal, and
the number should not be read that way.

Doing all three changes in one release is deliberate. Each of them invalidates
every trained model on its own, so splitting them across releases would cost
every existing installation two or three forced retrainings instead of one.

## Alternatives considered

**Fixing only the split.** The smallest correct change, and it was on the table
as its own option. Rejected because the noise-aware threshold costs no extra
fit, uses a number `permutation_importance` already returns, and would otherwise
invalidate every model a second time later.

**`k = 2` or `k = 3`.** Both defensible as confidence levels under an
approximate normality argument, and both measured no worse than `k = 1` within
the noise band. Rejected because `std` here is the spread of the repeats, not a
standard error, so treating it as a confidence interval overstates what the
number means. `k = 1` makes the weaker and better-supported claim.

**Deferring `k` until data from several plants exists.** Honest about the
evidence, and it remains the right frame for the larger question of whether the
selector should exist at all (chapter 6, point 7 of the Umsetzungsplan). Not
chosen here because it would leave the known-flattered importances in place for
another release cycle.

**Grouping cyclical pairs by asking the encoders.** Structurally cleaner than
matching on a `_sin`/`_cos` suffix, but the selector is a pipeline step that
receives a transformed frame and has no reference to the `ColumnTransformer`
that built it. Introducing one would couple the selector to the pipeline shape
for a naming convention the encoders in `pvlearn.encoders` fully control.
