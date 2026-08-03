#!/usr/bin/env python3
"""Freeze baseline forecasts from the current solaredge2mqtt production code.

Phase 0 of the roadmap needs the forecasts the existing implementation produces on
the reference dataset, recorded before any code is moved. Phase 1a is accepted only
if the extracted pvlearn code reproduces them, so this artifact is what makes the
extraction verifiable at all.

The script trains solaredge2mqtt's own `Forecaster` on the dataset minus a holdout
tail, predicts that tail, and writes the predictions plus metrics and the exact
library versions they depend on.

Run it from a throwaway virtualenv, never from the pvlearn dev environment — the
baseline has to be produced under solaredge2mqtt's own pinned dependency versions,
which are not pvlearn's. Importing `Forecaster` pulls in the InfluxDB and MQTT
clients through `core.settings`, so the whole package has to be installed:

    python -m venv /tmp/baseline && . /tmp/baseline/bin/activate
    pip install -e "/path/to/solaredge2mqtt[forecast]" pyarrow
    TZ=Europe/Berlin python scripts/freeze_baseline_forecast.py \\
        --solaredge2mqtt /path/to/solaredge2mqtt \\
        --out tests/fixtures/baseline_forecast

Determinism: training is reproducible because `HistGradientBoostingRegressor`,
`train_test_split` and `permutation_importance` all pin `random_state=42`. It is
*not* reproducible across scikit-learn versions, which is why the versions are
recorded alongside the predictions and why chapter 3.4 makes them part of the
model metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "tests" / "fixtures" / "reference_dataset.parquet"
DEFAULT_METADATA = REPO_ROOT / "tests" / "fixtures" / "reference_dataset.json"

# Wh per kWh. Forecaster.predict() runs energy predictions through
# ForecasterType.prepare_value, which divides by this to publish kWh, while the
# reference dataset records energy in Wh. Metrics are computed in Wh.
WH_PER_KWH = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solaredge2mqtt",
        type=Path,
        required=True,
        help="Path to a solaredge2mqtt working copy (the dir holding its package)",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--holdout-hours",
        type=int,
        default=720,
        help="Trailing hours held out of training and predicted (default: 720)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path without suffix; .parquet and .json are written",
    )
    return parser.parse_args()


def assert_process_timezone_matches(dataset_timezone: str) -> None:
    """Fail unless the process timezone matches the dataset's.

    solaredge2mqtt reads the timezone once at import time
    (`encoders.py`: `LOCAL_TZ = get_localzone_name()`) and SunEncoder builds its
    location from it. Baselines frozen under a different process timezone are
    silently not comparable. Removing this global is chapter 3.2 of the roadmap;
    until then it has to be asserted.
    """
    from tzlocal import get_localzone_name  # pyright: ignore[reportMissingImports]

    process_timezone = get_localzone_name()
    if process_timezone != dataset_timezone:
        raise SystemExit(
            f"Process timezone {process_timezone!r} does not match the dataset's "
            f"{dataset_timezone!r}. solaredge2mqtt resolves the timezone globally at "
            f"import time, so the baseline would not be reproducible. "
            f"Re-run with TZ={dataset_timezone}."
        )


def main() -> int:
    args = parse_args()

    metadata = json.loads(args.metadata.read_text())
    dataset_timezone = metadata["location"]["timezone"]

    assert_process_timezone_matches(dataset_timezone)

    sys.path.insert(0, str(args.solaredge2mqtt.resolve()))

    import pandas as pd
    import sklearn
    from sklearn.metrics import (  # pyright: ignore[reportMissingImports]
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    from solaredge2mqtt.core.settings.models import (  # pyright: ignore[reportMissingImports]
        LocationSettings,
    )
    from solaredge2mqtt.services.forecast.models import (  # pyright: ignore[reportMissingImports]
        ForecasterType,
    )
    from solaredge2mqtt.services.forecast.service import (  # pyright: ignore[reportMissingImports]
        Forecaster,
    )
    from solaredge2mqtt.services.forecast.settings import (  # pyright: ignore[reportMissingImports]
        ForecastSettings,
    )

    data = pd.read_parquet(args.dataset).sort_values("_time").reset_index(drop=True)
    # What ForecastService.train() does before handing the frame to the Forecaster.
    data["time"] = data["_time"].dt.tz_convert(dataset_timezone)

    holdout = args.holdout_hours
    if holdout >= len(data):
        raise SystemExit(f"Holdout {holdout} exceeds the {len(data)} available rows.")

    train_data = data.iloc[:-holdout]
    holdout_data = data.iloc[-holdout:]

    # enable=False keeps the cachingdir validator returning None, so the pipeline
    # runs without joblib memoisation. Tuning stays off: GridSearchCV would change
    # the selected hyperparameters and defeat the point of a stable baseline.
    settings = ForecastSettings(enable=False, hyperparametertuning=False)
    location = LocationSettings(
        latitude=metadata["location"]["latitude"],
        longitude=metadata["location"]["longitude"],
    )

    predictions = pd.DataFrame({"_time": holdout_data["_time"].to_numpy()})
    metrics: dict[str, dict[str, object]] = {}

    for typed in ForecasterType:
        target = typed.target_column
        print(f"Training {target} on {len(train_data)} rows ...")

        forecaster = Forecaster(typed, location, settings)
        forecaster.train(train_data)

        predicted = forecaster.model_pipeline.predict(holdout_data)
        published = pd.Series(predicted).apply(typed.prepare_value)

        predictions[f"{target}_raw"] = predicted
        predictions[f"{target}_published"] = published.to_numpy()

        # Energy is published in kWh but recorded in Wh; compare in Wh.
        scale = WH_PER_KWH if target == "energy" else 1
        actual = holdout_data[target].to_numpy()
        comparable = published.to_numpy() * scale

        mae = float(mean_absolute_error(actual, comparable))
        unit = "Wh" if target == "energy" else "W"

        metrics[target] = {
            "mae": mae,
            "rmse": float(mean_squared_error(actual, comparable) ** 0.5),
            "r2": float(r2_score(actual, comparable)),
            "unit": unit,
            "selected_features": forecaster.model_pipeline.named_steps[
                "feature_selector"
            ].important_features_,
        }
        print(f"  {target}: MAE {mae:.2f} {unit}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = args.out.with_suffix(".parquet")
    sidecar_path = args.out.with_suffix(".json")

    predictions.to_parquet(parquet_path, index=False)

    sidecar = {
        "dataset_rows": int(len(data)),
        "training_rows": int(len(train_data)),
        "holdout_rows": int(len(holdout_data)),
        "holdout_first_timestamp": holdout_data["_time"].min().isoformat(),
        "holdout_last_timestamp": holdout_data["_time"].max().isoformat(),
        "location": metadata["location"],
        "hyperparameter_tuning": False,
        "metrics": metrics,
        "versions": {
            "python": sys.version.split()[0],
            "scikit-learn": sklearn.__version__,
            "numpy": __import__("numpy").__version__,
            "pandas": pd.__version__,
        },
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")

    print(f"\nWrote {len(predictions)} predictions to {parquet_path}")
    print(f"Wrote baseline metadata to {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
