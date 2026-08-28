# 4. Found hyperparameters outlive the search that produced them

- **Status:** accepted
- **Date:** 2026-08-28

## Context

`Forecaster.train` builds a fresh pipeline through `_prepare_model_pipeline` on
every call. With `hyperparametertuning` enabled it replaced that pipeline with a
clone of `HalvingGridSearchCV.best_estimator_`; with tuning disabled it kept the
constructor defaults of `HistGradientBoostingRegressor`. Nothing carried the
search result from one call to the next — `best_params_` was logged and then
dropped.

The two costs in a training run are not comparable. A single fit is one pass
over the plant's history; the search fits many candidates over the halving
rungs, and each candidate fit includes `PFISelector`'s permutation importance.
The repository does not measure the ratio, so no factor is claimed here — but
the search is a multiple of the fit by construction, since it contains many of
them.

Consumers have moved to schedules that reflect that. `solaredge2mqtt` writes
training data hourly and rebuilds the model at most daily, and wants the search
on a slower cadence still. Under the old API the only way to express that was to
flip `Forecaster.enable_hyperparameter_tuning` between calls from outside, which
made the model alternate between tuned parameters and library defaults: the
tuning held until exactly the next retraining and was then discarded. A consumer
therefore had to choose between fresh data and tuned parameters.

## Decision

Tuning becomes a property of the run, and its result becomes state of the
forecaster.

`train` takes `hyperparametertuning: bool | None = None`. `None` follows the
configured `enable_hyperparameter_tuning`; `True` and `False` decide for that
one call. Consumers stop mutating the attribute from outside.

`_hyperparametertuning` returns the tuned pipeline **and** `best_params_`. After
a successful run `Forecaster.hyperparameters` and
`Forecaster.hyperparameters_tuned_at` hold them, published at the same point as
`model_pipeline` and `metadata`, so a failed run leaves the previous ones in
place. A run that does not tune applies the stored parameters to the fresh
pipeline with `Pipeline.set_params`.

`ModelMetadata` carries both fields, so they survive a process restart, and
`Forecaster.load` restores them. `hyperparameters_tuned_at` is the timestamp of
the search, not of the training run, and is carried forward unchanged by untuned
runs — a consumer reads it to decide whether a new search is due. Neither field
takes part in `raise_on_mismatch`: they describe the model, they do not decide
whether it can be loaded, and ADR 0003 keeps that decision at the release
version alone. Both have defaults, so a sidecar written before this change still
validates.

Applying stored parameters never fails a training run. `set_params` is wrapped,
and on `ValueError` the rejected keys are logged, the stored parameters are
dropped, and the run continues with the defaults. The parameter grid can change
between releases, and a stale key has to degrade into "train untuned" rather
than into an exception in the consumer's retraining loop.

The keys are pipeline-scoped — `model__max_iter`, `model__max_depth`,
`model__learning_rate` — so they reach the `model` step only. `PFISelector`
holds its own clone of the base estimator and keeps the defaults. That is
deliberate: the selector's job is to rank features, not to be the best possible
regressor, and the search scores the pipeline end to end, so parameters tuned
through it were never measured against the selector's internal fit.

## Consequences

A consumer can retrain on its data cadence and search on a much slower one, and
every retraining in between fits with the parameters the last search found.

With no stored parameters and no per-call override the code path is identical to
the previous one, which is what `tests/test_baseline_forecast.py` and
`tests/test_extraction_regression.py` hold in place: they were not modified by
this change and stay green.

The stored parameters are training state, not configuration. They are not
readable from `ForecasterConfig` and cannot be pinned there; the only way to
change them is another search.

A model persisted before this change loads with no stored parameters and trains
untuned with the defaults until the next search, which matches what it was
already doing.

## Alternatives considered

**Keeping tuning coupled to every training run.** The status quo. Rejected
because it forces the consumer to choose between fresh data and tuned
parameters: with the search on, every retraining pays for it; with it off, the
next retraining throws the previous search away.

**Pinning the parameters in `ForecasterConfig`.** A consumer could copy the
logged `best_params_` into its configuration and get stable parameters without
any new state. Rejected because they are a training result, not a setting: they
are derived from the plant's own history, they change when that history changes,
and a value pinned in configuration would silently outlive the data it was
measured on. It also moves the responsibility for a correct parameter dictionary
to the consumer, where a typo becomes a `ValueError` at load time rather than a
dropped key in a log line.

**Storing the tuned estimator itself instead of the parameters.** Persisting
`best_estimator_` and refitting it would carry more than the parameters — it
would carry a fitted state that a later run has to be careful to discard.
Parameters are the smaller, inspectable artefact, they survive in the JSON
sidecar next to the metrics, and a consumer can read them without unpickling a
model.
