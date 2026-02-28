"""Generate map-ready mock segment GeoJSON from segment ids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported format: {path}")


def build_mock_geojson(
    segment_ids: list[str],
    center_lon: float,
    center_lat: float,
    seed: int = 42,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    features: list[dict[str, object]] = []

    for idx, segment_id in enumerate(segment_ids):
        corridor_id = segment_id.split("_s")[0] if "_s" in segment_id else f"corr_{idx // 6:03d}"
        lon_offset = rng.normal(0.0, 0.18)
        lat_offset = rng.normal(0.0, 0.11)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        half_len_deg = rng.uniform(0.003, 0.008)

        x0 = center_lon + lon_offset - half_len_deg * np.cos(angle)
        y0 = center_lat + lat_offset - half_len_deg * np.sin(angle)
        x1 = center_lon + lon_offset + half_len_deg * np.cos(angle)
        y1 = center_lat + lat_offset + half_len_deg * np.sin(angle)

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "segment_id": segment_id,
                    "corridor_id": corridor_id,
                    "segment_index": idx,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[x0, y0], [x1, y1]],
                },
            }
        )

    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeseries", required=True, help="Segment timeseries table with segment_id")
    parser.add_argument("--output", required=True, help="Output GeoJSON path")
    parser.add_argument("--center-lon", type=float, default=-81.1637, help="Map center longitude")
    parser.add_argument("--center-lat", type=float, default=33.8361, help="Map center latitude")
    args = parser.parse_args()

    ts = _read_table(Path(args.timeseries))
    if "segment_id" not in ts.columns:
        raise ValueError("timeseries table must include segment_id")
    segment_ids = sorted(ts["segment_id"].astype(str).unique().tolist())
    geojson = build_mock_geojson(segment_ids, args.center_lon, args.center_lat)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(geojson, f)

    print(f"Wrote mock segment geometry: {out} ({len(segment_ids)} segments)")


if __name__ == "__main__":
    main()
