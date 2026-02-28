"""Vegetation Vision MVP modules."""

from .forecast import build_forecast_rows
from .features import add_temporal_features, build_segment_feature_table
from .risk import compute_risk_scores

try:
    from .segments import (
        SegmentingConfig,
        build_segment_buffers,
        build_segments,
        write_segment_geojson,
        write_segment_metadata_csv,
    )
except Exception:  # pragma: no cover - optional geospatial dependency path
    SegmentingConfig = None
    build_segment_buffers = None
    build_segments = None
    write_segment_geojson = None
    write_segment_metadata_csv = None

__all__ = [
    "add_temporal_features",
    "build_forecast_rows",
    "build_segment_feature_table",
    "build_segment_buffers",
    "build_segments",
    "compute_risk_scores",
    "SegmentingConfig",
    "write_segment_geojson",
    "write_segment_metadata_csv",
]
