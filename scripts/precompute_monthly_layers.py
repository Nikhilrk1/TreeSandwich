"""Precompute per-month map layers by joining segment geometry with risk rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("timeseries must be .parquet or .csv")


def _load_geojson(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("type") != "FeatureCollection":
        raise ValueError("segments file must be a FeatureCollection GeoJSON")
    return payload


def _write_geojson(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes", "y"})
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeseries", required=True, help="Segment timeseries table (.parquet/.csv)")
    parser.add_argument("--segments", required=True, help="Segment geometry FeatureCollection GeoJSON")
    parser.add_argument("--outdir", required=True, help="Output directory for monthly layers")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-risk", type=int, default=0)
    args = parser.parse_args()

    df = _read_table(Path(args.timeseries))
    if df.empty:
        raise ValueError("timeseries table is empty")

    df["segment_id"] = df["segment_id"].astype(str)
    df["target_date"] = pd.to_datetime(df["target_date"]).dt.to_period("M").dt.to_timestamp()
    df["is_forecast"] = _to_bool(df["is_forecast"])

    hist = df[
        (df["horizon_months"].astype(int) == 0)
        & (~df["is_forecast"])
        & (df["risk_confidence"].astype(float) >= args.min_confidence)
        & (df["risk_score"].astype(float) >= args.min_risk)
    ].copy()
    if hist.empty:
        raise ValueError("No historical rows passed filtering.")

    geom = _load_geojson(Path(args.segments))
    outdir = Path(args.outdir)

    months = sorted(hist["target_date"].unique().tolist())
    for month in months:
        rows = hist[hist["target_date"] == month]
        risk_map = rows.set_index("segment_id")[
            ["risk_score", "risk_level", "risk_confidence", "ndvi_median", "ndvi_anomaly"]
        ].to_dict(orient="index")

        features: list[dict[str, object]] = []
        for feature in geom["features"]:
            props = feature.get("properties", {}) or {}
            segment_id = str(props.get("segment_id", ""))
            if segment_id not in risk_map:
                continue
            metrics = risk_map[segment_id]
            merged_props = dict(props)
            merged_props.update(
                {
                    "date": pd.Timestamp(month).strftime("%Y-%m"),
                    "risk_score": float(metrics["risk_score"]),
                    "risk_level": metrics["risk_level"],
                    "risk_confidence": float(metrics["risk_confidence"]),
                    "ndvi_median": float(metrics["ndvi_median"]) if pd.notna(metrics["ndvi_median"]) else None,
                    "ndvi_anomaly": float(metrics["ndvi_anomaly"]) if pd.notna(metrics["ndvi_anomaly"]) else None,
                }
            )
            features.append(
                {
                    "type": "Feature",
                    "geometry": feature.get("geometry"),
                    "properties": merged_props,
                }
            )

        payload = {"type": "FeatureCollection", "features": features}
        out_path = outdir / f"layer_{pd.Timestamp(month).strftime('%Y-%m')}.geojson"
        _write_geojson(out_path, payload)
        print(f"Wrote {out_path} ({len(features)} features)")


if __name__ == "__main__":
    main()
