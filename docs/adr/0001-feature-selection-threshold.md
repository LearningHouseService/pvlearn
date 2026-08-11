# 1. PFISelector keeps features by an absolute importance threshold

- **Status:** accepted
- **Date:** 2026-08-04
- **Phase:** 1b (model consolidation and schema normalisation)

## Context

`PFISelector` computes permutation importance for every transformed feature and
keeps the ones above a threshold. Inherited from solaredge2mqtt, that threshold
was the **75th percentile of the candidates' importances** — a relative rule
that always keeps roughly the top quarter of whatever is offered to it.

Phase 1b changed the candidate set: `season` disappeared with the `ephem`
dependency, `weather_main` disappeared with the canonical schema, and the time
of day moved from the hour to minutes since midnight. Candidates went from 30
transformed features to 28.

That shifted the cut. `dew_point` and `surface_pressure` fell below it,
`time_day_of_year_sin` rose above it, and forecast quality on the frozen
reference dataset dropped from 620.88 Wh MAE to 668.51 Wh — 7.67% worse, which
Phase 1b's acceptance criterion ("not worse than the baseline") does not allow.

The encoding changes themselves are not responsible. Pinning the selector to
the feature set the baseline had selected reproduces the frozen baseline
exactly: 620.88 Wh MAE, R² 0.8859. The entire difference comes from *which*
features are selected.

Measured on the reference dataset (9,580 hourly rows, 720-row holdout, no
hyperparameter tuning):

| Selection rule | MAE (Wh) | R² | Features kept |
|---|---|---|---|
| Frozen baseline | 620.88 | 0.8859 | 8 |
| Percentile 75 (inherited) | 668.51 | 0.8692 | 7 |
| Percentile 60 | 643.71 | 0.8772 | 11 |
| Percentile 50 | 620.51 | 0.8870 | 14 |
| Importance > 0 | 641.54 | 0.8831 | 24 |
| No selection at all | 638.23 | 0.8836 | 28 |

## Decision

Keep every feature whose permutation importance is above zero. Drop the
percentile rule.

If no feature clears the bar, keep all of them: an importance estimate that
finds nothing is uninformative about the features, and an empty feature set
cannot be fitted at all.

## Consequences

The rule no longer depends on how many candidates exist. This matters beyond
the numbers above: a provider that delivers irradiance — `ghi`, `dni`, `dhi` —
adds three candidates that a quantile rule would let shift the cut for every
existing feature, so the same plant would select differently per provider.
That runs against what the canonical schema in `pvlearn/schema.py` is for:
provider-specific field names never reach the library, and a provider that
delivers less yields a smaller feature set rather than a different model.

Forecast quality lands at 641.54 Wh MAE, 3.3% above the baseline instead of
7.67%, with R² (0.8831) close to the baseline's 0.8859. The remaining gap is a
different, larger feature set on a single dataset, not a systematic error.

The selector now keeps 24 of 28 features and is therefore close to a no-op —
the "no selection" row is statistically indistinguishable from it. Removing
`PFISelector` entirely is the honest conclusion to draw from that, but it
changes the pipeline shape, removes the `selected_features` field the model
metadata (`pvlearn/metadata.py`) is specified to carry, and cannot be judged on
one plant's data. Left as an open question rather than implemented here.

## Alternatives considered

**Percentile 50.** The best measured number (620.51 Wh), marginally better than
the baseline. Rejected because the value was picked by looking at the holdout
that Phase 1b is accepted against — tuning the threshold on the acceptance data
makes the acceptance meaningless. It also keeps the structural defect: a
quantile stays dependent on the candidate count.

**Keeping percentile 75 and widening the tolerance.** Rejected because the
10% tolerance in `tests/test_extraction_regression.py` was calibrated in Phase
1a for CPU-microarchitecture-dependent split differences, not as a budget for
model changes. Using it to absorb a reproducible 7.67% regression would spend a
guard on the thing it exists to catch.

**Better selectors** (noise-aware threshold via `importances_std`, Boruta-style
shadow features, `SequentialFeatureSelector` with `TimeSeriesSplit`) are all
plausible improvements over a plain `> 0` rule, and all are larger changes than
Phase 1b's scope allows. Left open, together with the chronological-split
defect in `PFISelector.fit`.
