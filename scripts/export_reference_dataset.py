#!/usr/bin/env python3
"""Export a reference training dataset from a solaredge2mqtt InfluxDB instance.

Phase 0 of the roadmap needs a frozen dataset that the extraction work in Phase 1a
can be verified against. This script pulls the `forecast_training` measurement,
pivots it into the same wide shape that `Forecaster.train()` consumes, and writes
it as Parquet next to a JSON sidecar describing the export.

The measurement carries weather forecast values and the measured production of the
corresponding hour. It contains no account identifiers, so the only thing worth
reducing is the plant location, which lives in the sidecar rather than the data and
is rounded to two decimals (~1 km) by default.

Run it against a read-only InfluxDB token:

    pip install influxdb-client pandas pyarrow
    export INFLUXDB_TOKEN=...
    python scripts/export_reference_dataset.py \\
        --url http://10.0.1.6:8086 \\
        --org <org-id> \\
        --bucket solaredge \\
        --days 90 \\
        --latitude 49.45 --longitude 11.08 --timezone Europe/Berlin \\
        --out tests/fixtures/reference_dataset
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Columns InfluxDB adds to every pivoted result and that carry no training signal.
INFLUX_INTERNAL_COLUMNS = ["result", "table", "_start", "_stop", "_measurement"]

FLUX_QUERY = """
from(bucket: "{bucket}")
    |> range(start: -{days}d)
    |> filter(fn: (r) => r["_measurement"] == "forecast_training")
    |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", required=True, help="InfluxDB base URL, e.g. http://host:8086"
    )
    parser.add_argument("--org", required=True, help="InfluxDB organisation name or id")
    parser.add_argument(
        "--bucket", required=True, help="Bucket holding forecast_training"
    )
    parser.add_argument(
        "--days", type=int, default=90, help="Days of history to export"
    )
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument(
        "--timezone", required=True, help="IANA name, e.g. Europe/Berlin"
    )
    parser.add_argument(
        "--coordinate-precision",
        type=int,
        default=2,
        help="Decimals to round the location to in the sidecar (default: 2)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path without suffix; .parquet and .json are written",
    )
    return parser.parse_args()


def query_training_data(args: argparse.Namespace, token: str):
    # influxdb-client is deliberately not a project dependency: pvlearn itself must stay
    # free of I/O clients. This one-off Phase 0 script asks you to install it ad hoc.
    from influxdb_client import (  # pyright: ignore[reportMissingImports]
        InfluxDBClient,
    )

    query = FLUX_QUERY.format(bucket=args.bucket, days=args.days)

    with InfluxDBClient(url=args.url, token=token, org=args.org) as client:
        frame = client.query_api().query_data_frame(query)

    # A multi-table result comes back as a list of frames.
    if isinstance(frame, list):
        import pandas as pd

        frame = pd.concat(frame, ignore_index=True)

    frame = frame.drop(columns=INFLUX_INTERNAL_COLUMNS, errors="ignore")
    return frame.sort_values("_time")


def main() -> int:
    args = parse_args()

    token = os.environ.get("INFLUXDB_TOKEN")
    if not token:
        print("INFLUXDB_TOKEN is not set.", file=sys.stderr)
        return 1

    data = query_training_data(args, token)

    if data.empty:
        print(
            "Query returned no rows - check bucket, org and time range.",
            file=sys.stderr,
        )
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = args.out.with_suffix(".parquet")
    sidecar_path = args.out.with_suffix(".json")

    data.to_parquet(parquet_path, index=False)

    precision = args.coordinate_precision
    sidecar = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_measurement": "forecast_training",
        "weather_provider": "openweathermap",
        "requested_days": args.days,
        "rows": int(len(data)),
        "first_timestamp": data["_time"].min().isoformat(),
        "last_timestamp": data["_time"].max().isoformat(),
        "columns": sorted(data.columns.tolist()),
        "location": {
            "latitude": round(args.latitude, precision),
            "longitude": round(args.longitude, precision),
            "timezone": args.timezone,
        },
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")

    print(f"Wrote {len(data)} rows to {parquet_path}")
    print(f"Wrote export metadata to {sidecar_path}")
    print(f"Range: {sidecar['first_timestamp']} .. {sidecar['last_timestamp']}")
    print(f"Columns: {', '.join(sidecar['columns'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
