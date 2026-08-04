# pvlearn — Agent & AI Instructions

This file is the single source of truth for AI coding assistants (Claude Code, GitHub Copilot,
Cursor, etc.). Tool-specific configurations in `.github/` reference this file and add only
tool-specific syntax on top.

The authoritative roadmap, architecture decisions, and canonical data model live in
[`pvlearn-umsetzungsplan.md`](pvlearn-umsetzungsplan.md). Read it before starting any non-trivial
task — it defines what phase the project is in and which changes are in scope.

## Project Overview

- **Purpose:** Self-learning PV production forecast library and REST service. Trains on a plant's
  own historical measurements instead of a generic physical plant model (orientation, tilt, kWp) —
  that is the differentiator against Forecast.Solar and Solcast.
- **Language:** Python (>=3.12, <4)
- **Package Manager:** pip with `pyproject.toml`
- **Origin:** extracted from the forecast module of `DerOetzi/solaredge2mqtt`. See the
  Umsetzungsplan for the extraction phases and what stays behind in that repository.

### Architecture Principle — the library is I/O-free

`pvlearn` core has **no** MQTT, InfluxDB, HTTP, or filesystem access outside explicitly passed
paths. In: DataFrames and typed models. Out: forecasts. Everything else — weather-provider HTTP
clients, persistence, the REST API — lives under `pvlearn.service` (the `[service]` extra) or in
consuming applications. Never add a dependency on `fastapi`, `httpx`, or similar to the core
package; if a change needs one, it belongs in `pvlearn.service`.

### Directory Structure

```
pvlearn/
├── pvlearn/            # core library — I/O-free, DataFrames in, forecasts out
│   └── service/        # REST service (FastAPI), brain persistence, weather-provider adapters
│                        #   only importable when the `service` extra is installed
tests/                   # unit and integration tests mirroring pvlearn/ structure
scripts/                 # one-off maintenance and data-preparation scripts, not shipped
                         #   with the package and free to use dependencies pvlearn does not
pvlearn-umsetzungsplan.md  # phased roadmap and architecture decisions — read first
```

### Canonical Data Model

The weather feature schema, time/sun-position features, target variable, and model metadata /
invalidation rules are specified in chapter 3 of the Umsetzungsplan. Do not invent field names or
diverge from that schema — every trained model becomes invalid if it changes, so a change there is
a deliberate, versioned decision (`feature_schema_version`), not a normal refactor.

---

## Developer Commands

```bash
# Install with all development dependencies
pip install -e ".[dev,service]"

# Lint (must pass before commit)
ruff check .
ruff check . --fix   # auto-fix
ruff format .

# Type check
pyright

# Tests (run in parallel by default via pytest-xdist, -v --tb=short set in pyproject.toml)
pytest
pytest --cov=pvlearn --cov-report=xml:coverage.xml
pytest tests/path/to/test_file.py
```

### Commits and pull requests

This repository enforces the Developer Certificate of Origin. Every commit needs a
`Signed-off-by` trailer or the DCO check blocks the pull request — commit with `git commit -s`.
To repair a branch where it is missing: `git rebase <base> --signoff` followed by
`git push --force-with-lease`.

**One pull request per phase or sub-phase of the roadmap, not per commit.** A phase lands as
a single branch with however many commits it takes, reviewed and merged as one unit. Phase 0
was built the other way round — a pull request per step — which produced eight of them for
what is really one deliverable and made the phase impossible to review as a whole.

Commits within a branch stay individually meaningful, so the branch reads as a sequence rather
than one opaque change. Merges are squashed, which is why the pull request description carries
the reasoning that survives into `main`'s history.

---

## Code Conventions

- Use Python >=3.12 syntax and language features; do not rely on 3.13-only syntax since the
  compat matrix in CI tests down to 3.12.
- All code comments and documentation must be in **English**, independent of the language used in
  planning documents.
- Type hints are mandatory on public functions and methods; `pyright` runs in CI.
- Pydantic models for anything crossing a boundary (config, API payloads, persisted metadata).
- For diagrams use Mermaid.

### Project Patterns

- **Multi-tenancy:** nothing may be process-global. Timezone, location, and weather provider are
  always explicit parameters on the relevant object (a "brain" in service terms), never read from
  a module-level default or the local system. This was a real bug class in the codebase pvlearn
  was extracted from — see chapter 5 (Testing) and chapter 7 (Risks) of the Umsetzungsplan.
- **Custom sklearn transformers** (encoders, selectors) must support `sklearn.clone()` and
  pickling — constructor arguments must be primitive and serializable, not settings objects.
- **Model persistence:** joblib/pickle with a metadata sidecar per chapter 3.4. Loading a model
  whose `feature_schema_version`, sklearn minor version, weather provider, or location does not
  match current config means hard rejection and retraining — never a best-effort load.

### Testing

- Test files mirror the source structure under `tests/`.
- Use `pytest`; fixtures in `tests/conftest.py`.
- Test classes prefixed with `Test`; methods prefixed with `test_`.
- Property tests for schema tolerance: missing optional weather feature columns must never raise,
  only shrink the feature set.
- Timezone tests must cover at least one non-UTC zone and a DST transition.
- Service tests mock the weather provider so CI runs without network access.
- All new code must be covered by unit tests; minimum coverage threshold is **90%**.

---

## Security Guidelines

- Never commit secrets or credentials.
- Validate all external inputs (API payloads, provider responses) through Pydantic models.
- API communication uses HTTPS; the service's API-key mechanism follows the pattern from
  `learninghouse`.
- Filter sensitive data (API keys, tokens) from log output.
