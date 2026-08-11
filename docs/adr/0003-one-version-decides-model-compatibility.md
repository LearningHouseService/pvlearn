# 3. The pvlearn release alone decides model compatibility

- **Status:** accepted
- **Date:** 2026-08-10
- **Supersedes the versioning mechanism introduced in:** [ADR 0002](0002-noise-aware-chronological-feature-selection.md)

## Context

`ModelMetadata.raise_on_mismatch` compared three version fields — the feature
vocabulary (`feature_schema_version`), the shape of the pipeline
(`pipeline_version`), and the installed scikit-learn minor version — while
`pvlearn_version` was recorded but deliberately excluded, on the grounds that
"not every release changes how features are built".

Two problems with that split.

**It is redundant.** Runtime dependencies are pinned with `==`, so no dependency
can move without a pvlearn release, and neither the schema nor the pipeline can
change without one either. Every event the three fields detect is therefore
already a new release. The fields are a strictly finer partition of something
the release number alone separates.

**It is also incomplete**, which matters more. The three fields cover
scikit-learn but not numpy, joblib, or pydantic, and the persisted artefact
depends on those too: joblib's pickle path is numpy-version-sensitive, visible
today as a `DeprecationWarning` from `joblib/numpy_pickle.py` about NumPy 2.5
changing how an array's shape may be set. A model pickled under one numpy and
unpickled under another is exactly the silent-failure case the sidecar exists to
prevent, and the three fields do not see it.

The claim that numpy, pandas and scipy do not affect reproduction — recorded in
a `pyproject.toml` comment and in the message of commit `ad94aac` — is not
backed by anything executable. There is no dependency matrix in CI: the two jobs
differ only in Python version, and because dependencies are pinned exactly, both
install identical numpy, pandas, scipy and scikit-learn. No script in the
repository reproduces the check. It rests on a single manual run recorded only
as prose, and cannot fail if it stops being true.

Its counterpart from the same commit — that the frozen Phase 0 baseline "is
only reproducible against 1.9.0", which is what made scikit-learn's pin
load-bearing — fails harder. `test_records_the_versions_it_depends_on` looks
like it guards the claim but asserts only that the frozen sidecar records
`1.9.0`; it would pass unchanged if a newer scikit-learn reproduced the
baseline perfectly. And the claim is contradicted by a measurement the project
did make, recorded in `tests/test_extraction_regression.py`: the baseline does
not reproduce bit-for-bit across CPUs even with every version held fixed, CI
landing 5.26% off the MAE with `random_state=42` pinned throughout. The
variable that was actually measured to break reproduction is the machine, not
scikit-learn. Both comments now state only what is measured; the pins stay, as
a precaution rather than a finding.

## Decision

Compare one version: the release segment of `pvlearn_version`.
`feature_schema_version`, `pipeline_version` and `sklearn_version` are removed
from the sidecar, along with `FEATURE_SCHEMA_VERSION`, `PIPELINE_VERSION` and
`sklearn_minor_version()`.

Only the release segment takes part — `0.3.0` out of
`0.3.0.post2+gd22c402c0.d20260807`. `setuptools_scm` is configured with
`local_scheme = "node-and-date"`, so the full version string carries the commit
hash and, on a dirty tree, the date. Comparing it whole would invalidate every
locally trained model on every commit and again on every new day.

This is the conservative direction. The release is a superset trigger: it never
invalidates too little, only sometimes too much.

## Consequences

A release that changes neither schema, pipeline, nor dependencies — a
documentation or CI release — now invalidates every persisted model. That cost
is real and accepted: training needs 60 hours of history the caller already
holds, hyperparameter tuning is opt-in and off by default, and the alternative
is the known gap above. The project's stated order of preference is unchanged —
a needless retrain is cheap, a silently wrong forecast is not.

Between two releases the version does not move, so a developer who changes the
pipeline locally keeps loading the model trained before the change. Deleting the
persisted pair is the remedy; this is a development-time concern only, since
released versions differ by construction.

Every persisted model is invalidated once more by this change itself, but
cleanly: pydantic ignores unknown fields, so a sidecar carrying the three
removed ones still parses and is then rejected on its release with a precise
reason, rather than being reported as unreadable metadata.

## Alternatives considered

**Keeping the three fields and adding more.** Fields for numpy and joblib would
close the specific gap named above. Rejected because it treats the symptom: the
list is only ever as complete as the last audit of which dependency touches the
artefact, and each addition is another thing to remember to bump. The release
number needs no audit — it moves whenever anything ships.

**Keeping the three fields as they are.** The status quo optimises for avoiding
unnecessary retrains, at the price of a gap it cannot close without becoming the
alternative above. Rejected because the thing being optimised — retraining cost
— is the cheap side of the trade.

**Comparing the full version string.** Strictly the most conservative option and
the simplest to implement. Rejected for the `local_scheme` reason above: it
makes every local model unusable after any commit, which trains developers to
work around the check rather than trust it.
