"""CLI pipeline: pixel samples -> features -> risk -> forecast table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .features import add_temporal_features, build_segment_feature_table
from .forecast import build_forecast_rows
from .risk import compute_risk_scores


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format for {path}. Use .parquet or .csv")


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported output format for {path}. Use .parquet or .csv")


def build_timeseries(pixel_df: pd.DataFrame) -> pd.DataFrame:
    features = build_segment_feature_table(pixel_df)
    return build_timeseries_from_features(features)


def build_timeseries_from_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    temporal = add_temporal_features(feature_df)
    scored_hist = compute_risk_scores(temporal)

    scored_hist["as_of_date"] = scored_hist["date"]
    scored_hist["target_date"] = scored_hist["date"]
    scored_hist["horizon_months"] = 0
    scored_hist["is_forecast"] = False
    scored_hist["forecast_lower"] = np.nan
    scored_hist["forecast_upper"] = np.nan

    forecast = build_forecast_rows(scored_hist)
    combined = pd.concat([scored_hist, forecast], ignore_index=True, sort=False)
    combined = combined.sort_values(["segment_id", "target_date", "horizon_months"]).reset_index(drop=True)
    combined["model_version"] = "baseline_v1"

    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Vegetation Vision segment timeseries table.")
    parser.add_argument("--input", required=True, help="Input pixel sample table (.parquet or .csv)")
    parser.add_argument("--output", required=True, help="Output timeseries table (.parquet or .csv)")
    parser.add_argument(
        "--input-grain",
        default="pixel",
        choices=["pixel", "feature"],
        help="Input row grain: pixel-level rows or pre-aggregated segment feature rows.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    input_df = _read_table(input_path)
    if args.input_grain == "pixel":
        output_df = build_timeseries(input_df)
    else:
        output_df = build_timeseries_from_features(input_df)
    _write_table(output_df, output_path)

    print(f"Wrote {len(output_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
