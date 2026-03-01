"""Extract per-segment monthly NDVI stats from raster composites."""

from __future__ import annotations

import argparse
from glob import glob
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.transform import xy
from shapely.geometry import Point


def _write_table(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        df.to_parquet(output, index=False)
    elif output.suffix.lower() == ".csv":
        df.to_csv(output, index=False)
    else:
        raise ValueError("Output must end with .parquet or .csv")


def _parse_month_from_name(name: str, pattern: re.Pattern[str]) -> pd.Timestamp:
    match = pattern.search(name)
    if not match:
        raise ValueError(f"Could not parse month from filename: {name}")
    return pd.to_datetime(match.group(1)).to_period("M").to_timestamp()


def _compute_stats(arr: np.ndarray, threshold: float) -> dict[str, float | int]:
    flat = arr.reshape(-1)
    valid = np.isfinite(flat)
    values = flat[valid]
    expected = int(flat.size)
    obs = int(values.size)

    if obs == 0:
        return {
            "obs_count": 0,
            "expected_pixels": expected,
            "valid_pixel_frac": 0.0,
            "cloud_free_ratio": 0.0,
            "ndvi_median": np.nan,
            "ndvi_p90": np.nan,
            "frac_ndvi_gt_0_6": np.nan,
            "ndvi_iqr": np.nan,
        }

    return {
        "obs_count": obs,
        "expected_pixels": expected,
        "valid_pixel_frac": obs / max(expected, 1),
        "cloud_free_ratio": obs / max(expected, 1),
        "ndvi_median": float(np.nanmedian(values)),
        "ndvi_p90": float(np.nanpercentile(values, 90)),
        "frac_ndvi_gt_0_6": float(np.mean(values > threshold)),
        "ndvi_iqr": float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25)),
    }


def _min_distance_to_line(
    arr: np.ndarray,
    out_transform,
    line_geom,
    ndvi_threshold: float,
) -> float:
    veg_mask = np.isfinite(arr) & (arr > ndvi_threshold)
    rows, cols = np.where(veg_mask)
    if rows.size == 0:
        return float("nan")

    xs, ys = xy(out_transform, rows, cols, offset="center")
    min_dist = float("inf")
    for x, y in zip(xs, ys):
        dist = line_geom.distance(Point(float(x), float(y)))
        if dist < min_dist:
            min_dist = dist
    return float(min_dist) if np.isfinite(min_dist) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-segment NDVI stats from monthly rasters.")
    parser.add_argument("--segments", required=True, help="Segment buffer polygons GeoJSON/Shapefile/GPKG")
    parser.add_argument(
        "--centerlines",
        default=None,
        help="Optional segment centerline file with same segment ids; enables vegetation distance-to-line calculation.",
    )
    parser.add_argument("--segment-id-col", default="segment_id")
    parser.add_argument("--raster-glob", required=True, help="Glob path for monthly NDVI rasters")
    parser.add_argument(
        "--date-regex",
        default=r"(\d{4}-\d{2})",
        help="Regex used to parse YYYY-MM from raster filename.",
    )
    parser.add_argument("--ndvi-threshold", type=float, default=0.60)
    parser.add_argument("--output", required=True, help="Output feature table (.parquet/.csv)")
    args = parser.parse_args()

    segments = gpd.read_file(args.segments)
    if args.segment_id_col not in segments.columns:
        raise ValueError(f"Missing segment id column: {args.segment_id_col}")
    if segments.empty:
        raise ValueError("No segment polygons found.")

    centerlines = None
    if args.centerlines:
        centerlines = gpd.read_file(args.centerlines)
        if args.segment_id_col not in centerlines.columns:
            raise ValueError(f"Missing segment id column in centerlines: {args.segment_id_col}")
        centerlines[args.segment_id_col] = centerlines[args.segment_id_col].astype(str)

    raster_paths = sorted(Path(p) for p in glob(args.raster_glob))
    if not raster_paths:
        raise ValueError(f"No rasters matched: {args.raster_glob}")
    date_pattern = re.compile(args.date_regex)

    rows: list[dict[str, object]] = []
    for raster_path in raster_paths:
        date = _parse_month_from_name(raster_path.name, date_pattern)
        with rasterio.open(raster_path) as ds:
            seg = segments.to_crs(ds.crs)
            nodata = ds.nodata
            if centerlines is not None:
                centerline_local = centerlines.to_crs(ds.crs)
                line_lookup_local = centerline_local.set_index(args.segment_id_col)["geometry"].to_dict()
            else:
                line_lookup_local = None

            if ds.crs and ds.crs.is_geographic and line_lookup_local is not None:
                print(
                    f"Warning: raster {raster_path.name} CRS is geographic; "
                    "distance output will be in degree units, not meters."
                )

            for _, row in seg.iterrows():
                out, out_transform = mask(ds, [row.geometry], crop=True, filled=True)
                arr = out[0].astype("float32")
                if nodata is not None:
                    arr[arr == nodata] = np.nan

                stats = _compute_stats(arr, args.ndvi_threshold)
                segment_id = str(row[args.segment_id_col])
                if line_lookup_local is not None and segment_id in line_lookup_local:
                    stats["vegetation_distance_m"] = _min_distance_to_line(
                        arr,
                        out_transform,
                        line_lookup_local[segment_id],
                        args.ndvi_threshold,
                    )
                else:
                    stats["vegetation_distance_m"] = np.nan
                rows.append(
                    {
                        "segment_id": segment_id,
                        "date": date,
                        **stats,
                    }
                )

    feature_df = pd.DataFrame(rows).sort_values(["segment_id", "date"]).reset_index(drop=True)
    out = Path(args.output)
    _write_table(feature_df, out)
    print(f"Wrote feature table: {out} ({len(feature_df)} rows)")


if __name__ == "__main__":
    main()
