"""Corridor segmentation utilities for Vegetation Vision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString


@dataclass(frozen=True)
class SegmentingConfig:
    segment_length_m: float = 500.0
    buffer_m: float = 50.0
    input_crs: str = "EPSG:4326"
    metric_crs: str = "EPSG:3857"


def _iter_lines(geom: LineString | MultiLineString) -> list[LineString]:
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [line for line in geom.geoms if not line.is_empty]
    return []


def _split_line_by_length(line: LineString, segment_length_m: float) -> list[LineString]:
    if line.is_empty or line.length <= 0:
        return []
    if line.length <= segment_length_m:
        return [line]

    cuts = [0.0]
    dist = segment_length_m
    while dist < line.length:
        cuts.append(dist)
        dist += segment_length_m
    cuts.append(line.length)

    segments: list[LineString] = []
    for start, end in zip(cuts[:-1], cuts[1:]):
        a = line.interpolate(start)
        b = line.interpolate(end)
        seg = LineString([a, b])
        if seg.length > 0:
            segments.append(seg)
    return segments


def build_segments(
    corridors: gpd.GeoDataFrame,
    corridor_id_column: str = "corridor_id",
    config: SegmentingConfig = SegmentingConfig(),
) -> gpd.GeoDataFrame:
    """Split corridor centerlines into fixed-length segments."""

    if corridors.empty:
        raise ValueError("Input corridors GeoDataFrame is empty.")
    if corridor_id_column not in corridors.columns:
        raise ValueError(f"Missing corridor id column: {corridor_id_column}")

    gdf = corridors.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(config.input_crs)
    gdf = gdf.to_crs(config.metric_crs)

    rows: list[dict[str, object]] = []
    for _, row in gdf.iterrows():
        corridor_id = str(row[corridor_id_column])
        lines = _iter_lines(row.geometry)
        seg_idx = 0
        for line in lines:
            for seg in _split_line_by_length(line, config.segment_length_m):
                segment_id = f"{corridor_id}_s{seg_idx:04d}"
                rows.append(
                    {
                        "segment_id": segment_id,
                        "corridor_id": corridor_id,
                        "segment_index": seg_idx,
                        "length_m": float(seg.length),
                        "geometry": seg,
                    }
                )
                seg_idx += 1

    if not rows:
        raise ValueError("No line segments were produced from input geometry.")

    segments = gpd.GeoDataFrame(rows, geometry="geometry", crs=config.metric_crs)
    return segments


def build_segment_buffers(
    segments: gpd.GeoDataFrame,
    buffer_m: float = 50.0,
) -> gpd.GeoDataFrame:
    """Build corridor buffer polygons around segment centerlines."""

    if segments.empty:
        raise ValueError("Input segments GeoDataFrame is empty.")
    out = segments.copy()
    out["geometry"] = out.geometry.buffer(buffer_m)
    return out


def write_segment_geojson(
    segments: gpd.GeoDataFrame,
    output_path: str | Path,
    out_crs: str = "EPSG:4326",
) -> None:
    out = segments.to_crs(out_crs).copy()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_file(output_path, driver="GeoJSON")


def write_segment_metadata_csv(
    segments: gpd.GeoDataFrame,
    output_path: str | Path,
    out_crs: str = "EPSG:4326",
) -> None:
    out = segments.to_crs(out_crs).copy()
    centroids = out.geometry.centroid
    df = pd.DataFrame(
        {
            "segment_id": out["segment_id"].astype(str),
            "corridor_id": out["corridor_id"].astype(str),
            "length_m": out["length_m"].astype(float),
            "centroid_lon": centroids.x.astype(float),
            "centroid_lat": centroids.y.astype(float),
        }
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
