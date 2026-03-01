"""Build complete mock demo data for frontend testing.

Outputs:
- pixel_samples.parquet
- segment_timeseries.parquet
- segments.geojson
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Allow running as either:
# - python scripts/build_mock_demo_data.py
# - python -m scripts.build_mock_demo_data
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.generate_mock_pixels import build_mock_pixels
    from scripts.generate_mock_segments import build_mock_geojson
except ModuleNotFoundError:
    from generate_mock_pixels import build_mock_pixels
    from generate_mock_segments import build_mock_geojson
from vegetation_vision.pipeline import build_timeseries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate full mock dataset for Vegetation Vision frontend.")
    parser.add_argument("--outdir", default="data", help="Output directory")
    parser.add_argument("--segments", type=int, default=160)
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--pixels", type=int, default=30)
    parser.add_argument("--center-lon", type=float, default=-81.1637)
    parser.add_argument("--center-lat", type=float, default=33.8361)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pixel_path = outdir / "pixel_samples.parquet"
    ts_path = outdir / "segment_timeseries.parquet"
    geom_path = outdir / "segments.geojson"

    pixel_df = build_mock_pixels(
        segments=args.segments,
        months=args.months,
        pixels_per_segment_month=args.pixels,
    )
    pixel_df.to_parquet(pixel_path, index=False)

    ts_df = build_timeseries(pixel_df)
    ts_df.to_parquet(ts_path, index=False)

    segment_ids = sorted(ts_df["segment_id"].astype(str).unique().tolist())
    geojson = build_mock_geojson(segment_ids, args.center_lon, args.center_lat)
    with geom_path.open("w", encoding="utf-8") as f:
        json.dump(geojson, f)

    print(f"Wrote {pixel_path} ({len(pixel_df)} rows)")
    print(f"Wrote {ts_path} ({len(ts_df)} rows)")
    print(f"Wrote {geom_path} ({len(segment_ids)} segments)")
    print("Run: uvicorn backend.main:app --reload")
    print("Open: http://127.0.0.1:8000/app")


if __name__ == "__main__":
    main()
