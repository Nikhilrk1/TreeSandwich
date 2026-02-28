"""Build fixed-length corridor segments from centerline geometries."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd

from vegetation_vision.segments import (
    SegmentingConfig,
    build_segment_buffers,
    build_segments,
    write_segment_geojson,
    write_segment_metadata_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split corridor centerlines into fixed-length segments.")
    parser.add_argument("--input", required=True, help="Input corridors file (GeoJSON/Shapefile/GPKG)")
    parser.add_argument("--corridor-id-col", default="corridor_id", help="Column name for corridor id")
    parser.add_argument("--segment-length-m", type=float, default=500.0)
    parser.add_argument("--buffer-m", type=float, default=50.0)
    parser.add_argument("--out-segments-geojson", required=True, help="Output segmented centerlines GeoJSON")
    parser.add_argument("--out-buffers-geojson", required=True, help="Output buffered segment polygons GeoJSON")
    parser.add_argument("--out-metadata-csv", required=True, help="Output segment metadata CSV")
    args = parser.parse_args()

    in_path = Path(args.input)
    corridors = gpd.read_file(in_path)
    cfg = SegmentingConfig(segment_length_m=args.segment_length_m, buffer_m=args.buffer_m)

    segments = build_segments(corridors, corridor_id_column=args.corridor_id_col, config=cfg)
    buffers = build_segment_buffers(segments, buffer_m=args.buffer_m)

    write_segment_geojson(segments, args.out_segments_geojson)
    write_segment_geojson(buffers, args.out_buffers_geojson)
    write_segment_metadata_csv(segments, args.out_metadata_csv)

    print(f"Input corridors: {len(corridors)}")
    print(f"Output segments: {len(segments)}")
    print(f"Wrote centerlines to {args.out_segments_geojson}")
    print(f"Wrote buffers to {args.out_buffers_geojson}")
    print(f"Wrote metadata to {args.out_metadata_csv}")


if __name__ == "__main__":
    main()
