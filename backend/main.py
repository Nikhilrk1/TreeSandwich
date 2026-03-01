"""FastAPI service for TreeSandwich frontend driven by future_predictions.csv."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles


FUTURE_CSV_PATH = os.getenv("VV_FUTURE_PREDICTIONS_CSV", "future_predictions.csv")
POWERLINES_GEOJSON_PATH = os.getenv(
    "VV_POWERLINES_GEOJSON",
    "US_Electric_Power_Transmission_Lines_-6976209181916424225.geojson",
)
DEFAULT_BBOX_HALF_DEG = float(os.getenv("VV_DEFAULT_BBOX_HALF_DEG", "0.004"))
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIT_DIR = PROJECT_ROOT / "ViT"
YEARS = list(range(2025, 2031))

app = FastAPI(title="TreeSandwich API", version="0.2.0")
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
if PROJECT_ROOT.exists():
    app.mount("/files", StaticFiles(directory=str(PROJECT_ROOT), html=False), name="files")
if VIT_DIR.exists():
    app.mount("/ViT", StaticFiles(directory=str(VIT_DIR), html=False), name="vit")


def _normalize_bbox(value: str) -> tuple[float, float, float, float]:
    try:
        minx, miny, maxx, maxy = [float(x.strip()) for x in value.split(",")]
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid bbox. Expected 'minx,miny,maxx,maxy'.",
        ) from exc
    if minx >= maxx or miny >= maxy:
        raise HTTPException(status_code=400, detail="Invalid bbox extent values.")
    return minx, miny, maxx, maxy


def _bbox_intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _feature_bbox(geom: dict[str, object]) -> tuple[float, float, float, float]:
    coords: list[tuple[float, float]] = []

    def collect(obj: object) -> None:
        if isinstance(obj, list):
            if len(obj) == 2 and all(isinstance(v, (float, int)) for v in obj):
                coords.append((float(obj[0]), float(obj[1])))
            else:
                for part in obj:
                    collect(part)

    collect(geom.get("coordinates"))
    if not coords:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return (min(xs), min(ys), max(xs), max(ys))


def _jsonable(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if pd.isna(value):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if value is pd.NaT:
        return None
    return value


def _to_json_records(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        clean: dict[str, object] = {}
        for key, value in row.items():
            clean[key] = _jsonable(value)
        out.append(clean)
    return out


def _first_existing_column(df: pd.DataFrame, candidates: list[str], label: str, required: bool = False) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"Could not find required column for {label}. Checked: {candidates}")
    return None


def _prepare_segment_ids(df: pd.DataFrame) -> pd.Series:
    if "segment_id" in df.columns:
        return df["segment_id"].astype(str)
    if "feature_id" in df.columns and "section_id" in df.columns:
        return df["feature_id"].astype(str) + "_" + df["section_id"].astype(str)
    if "feature_id" in df.columns:
        return df["feature_id"].astype(str)
    return pd.Series([f"segment_{i:05d}" for i in range(len(df))], index=df.index)


def _prepare_predictions_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        raise ValueError("future_predictions.csv is empty.")

    df = raw_df.copy()
    df["segment_id"] = _prepare_segment_ids(df)

    # Distance columns (support both described schema and current CSV schema).
    orig_col = _first_existing_column(
        df,
        ["original_distance_m", "original_distance", "distance_2025_m", "now_distance_m"],
        label="original distance",
        required=True,
    )
    new_col = _first_existing_column(
        df,
        ["new_distance_m", "new_distance", "distance_2026_m"],
        label="new distance (2026)",
        required=False,
    )
    growth_col = _first_existing_column(
        df,
        ["growth_rate", "growth_rate_m", "growth_rate_2y_m", "predicted_future_delta_distance_m"],
        label="growth rate over two years",
        required=False,
    )

    df["original_distance_m"] = pd.to_numeric(df[orig_col], errors="coerce")
    df["new_distance_m"] = pd.to_numeric(df[new_col], errors="coerce") if new_col else np.nan
    df["growth_rate_2y_m"] = pd.to_numeric(df[growth_col], errors="coerce") if growth_col else np.nan

    # Fill whichever of new/growth is missing using the provided formula relationship.
    missing_new = df["new_distance_m"].isna() & df["growth_rate_2y_m"].notna()
    df.loc[missing_new, "new_distance_m"] = (
        df.loc[missing_new, "original_distance_m"] - df.loc[missing_new, "growth_rate_2y_m"] / 2.0
    )
    missing_growth = df["growth_rate_2y_m"].isna() & df["new_distance_m"].notna()
    df.loc[missing_growth, "growth_rate_2y_m"] = (
        (df.loc[missing_growth, "original_distance_m"] - df.loc[missing_growth, "new_distance_m"]) * 2.0
    )

    if df["new_distance_m"].isna().all():
        raise ValueError("Could not derive 2026 distance values from CSV.")
    if df["growth_rate_2y_m"].isna().all():
        raise ValueError("Could not derive growth-rate values from CSV.")

    # Bounding box pair of points support.
    bbox_sets = [
        ("min_lon", "min_lat", "max_lon", "max_lat"),
        ("bbox_min_lon", "bbox_min_lat", "bbox_max_lon", "bbox_max_lat"),
        ("lon1", "lat1", "lon2", "lat2"),
        ("west", "south", "east", "north"),
    ]
    bbox_cols: tuple[str, str, str, str] | None = None
    for cols in bbox_sets:
        if all(col in df.columns for col in cols):
            bbox_cols = cols
            break

    if bbox_cols is not None:
        min_lon_col, min_lat_col, max_lon_col, max_lat_col = bbox_cols
        df["bbox_min_lon"] = pd.to_numeric(df[min_lon_col], errors="coerce")
        df["bbox_min_lat"] = pd.to_numeric(df[min_lat_col], errors="coerce")
        df["bbox_max_lon"] = pd.to_numeric(df[max_lon_col], errors="coerce")
        df["bbox_max_lat"] = pd.to_numeric(df[max_lat_col], errors="coerce")
    else:
        lon_col = _first_existing_column(df, ["coord_lon", "center_lon", "lon", "longitude"], label="longitude", required=True)
        lat_col = _first_existing_column(df, ["coord_lat", "center_lat", "lat", "latitude"], label="latitude", required=True)
        lon = pd.to_numeric(df[lon_col], errors="coerce")
        lat = pd.to_numeric(df[lat_col], errors="coerce")
        df["bbox_min_lon"] = lon - DEFAULT_BBOX_HALF_DEG
        df["bbox_max_lon"] = lon + DEFAULT_BBOX_HALF_DEG
        df["bbox_min_lat"] = lat - DEFAULT_BBOX_HALF_DEG
        df["bbox_max_lat"] = lat + DEFAULT_BBOX_HALF_DEG

    # Ensure bbox direction is valid.
    df["bbox_min_lon"], df["bbox_max_lon"] = np.minimum(df["bbox_min_lon"], df["bbox_max_lon"]), np.maximum(
        df["bbox_min_lon"], df["bbox_max_lon"]
    )
    df["bbox_min_lat"], df["bbox_max_lat"] = np.minimum(df["bbox_min_lat"], df["bbox_max_lat"]), np.maximum(
        df["bbox_min_lat"], df["bbox_max_lat"]
    )

    df["center_lon"] = (df["bbox_min_lon"] + df["bbox_max_lon"]) / 2.0
    df["center_lat"] = (df["bbox_min_lat"] + df["bbox_max_lat"]) / 2.0
    df["annual_growth_m_per_year"] = df["growth_rate_2y_m"] / 2.0

    if "feature_id" in df.columns and "section_id" in df.columns:
        df["label"] = "feature " + df["feature_id"].astype(str) + " sec " + df["section_id"].astype(str)
    else:
        df["label"] = df["segment_id"]
    image_col = _first_existing_column(df, ["image_path", "image", "img_path"], label="image path", required=False)
    df["image_path"] = df[image_col] if image_col else None

    keep_cols = [
        "segment_id",
        "label",
        "center_lon",
        "center_lat",
        "bbox_min_lon",
        "bbox_min_lat",
        "bbox_max_lon",
        "bbox_max_lat",
        "original_distance_m",
        "new_distance_m",
        "growth_rate_2y_m",
        "annual_growth_m_per_year",
        "image_path",
    ]
    out = df[keep_cols].copy()
    out = out.dropna(subset=["segment_id", "center_lon", "center_lat", "original_distance_m", "new_distance_m", "growth_rate_2y_m"])
    out["segment_id"] = out["segment_id"].astype(str)
    return out.reset_index(drop=True)


def _distance_for_year(frame: pd.DataFrame, year: int) -> pd.Series:
    if year <= 2025:
        return frame["original_distance_m"]
    if year == 2026:
        return frame["new_distance_m"]
    years_passed = year - 2026
    return frame["new_distance_m"] - (frame["annual_growth_m_per_year"] * years_passed)


def _year_frame(year: int) -> pd.DataFrame:
    if year not in YEARS:
        raise HTTPException(status_code=400, detail=f"Year must be in {YEARS[0]}-{YEARS[-1]}.")
    base = _load_predictions().copy()
    base["year"] = int(year)
    base["distance_m"] = _distance_for_year(base, year).clip(lower=0.0)
    base["is_projected"] = year >= 2027
    base["distance_change_vs_2025_m"] = base["original_distance_m"] - base["distance_m"]
    return base


def _filter_bbox(df: pd.DataFrame, bbox: tuple[float, float, float, float] | None) -> pd.DataFrame:
    if bbox is None:
        return df

    minx, miny, maxx, maxy = bbox
    mask = ~(
        (df["bbox_max_lon"] < minx)
        | (df["bbox_min_lon"] > maxx)
        | (df["bbox_max_lat"] < miny)
        | (df["bbox_min_lat"] > maxy)
    )
    return df[mask].copy()


@lru_cache(maxsize=1)
def _load_predictions() -> pd.DataFrame:
    if not os.path.exists(FUTURE_CSV_PATH):
        raise FileNotFoundError(
            f"Future predictions CSV not found at {FUTURE_CSV_PATH}. "
            "Set VV_FUTURE_PREDICTIONS_CSV to your input file."
        )
    raw = pd.read_csv(FUTURE_CSV_PATH)
    return _prepare_predictions_frame(raw)


@lru_cache(maxsize=1)
def _load_powerline_index() -> dict[str, object]:
    if not os.path.exists(POWERLINES_GEOJSON_PATH):
        raise FileNotFoundError(
            f"Powerline GeoJSON not found at {POWERLINES_GEOJSON_PATH}. "
            "Set VV_POWERLINES_GEOJSON to your US lines dataset."
        )

    with open(POWERLINES_GEOJSON_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("type") != "FeatureCollection":
        raise ValueError("Powerline GeoJSON must be a FeatureCollection.")

    indexed: list[tuple[tuple[float, float, float, float], dict[str, object]]] = []
    for feature in payload.get("features", []):
        indexed.append((_feature_bbox(feature.get("geometry", {})), feature))

    return {
        "type": "FeatureCollection",
        "crs": payload.get("crs"),
        "features": indexed,
    }


@app.get("/health")
def health() -> dict[str, object]:
    try:
        df = _load_predictions()
    except FileNotFoundError as exc:
        return {"status": "degraded", "detail": str(exc)}

    return {
        "status": "ok",
        "rows": int(len(df)),
        "segments": int(df["segment_id"].nunique()),
        "years": YEARS,
    }


@app.get("/timeline/years")
def timeline_years() -> dict[str, object]:
    return {"years": YEARS, "default": YEARS[-1]}


@app.post("/reload")
def reload_data() -> dict[str, str]:
    _load_predictions.cache_clear()
    _load_powerline_index.cache_clear()
    return {"status": "reloaded"}


@app.get("/map/layer")
def map_layer(
    year: int = Query(..., ge=YEARS[0], le=YEARS[-1]),
    bbox: str | None = Query(
        None,
        description="Optional extent filter as 'minx,miny,maxx,maxy' in EPSG:4326",
    ),
    max_features: int = Query(12000, ge=100, le=250000),
) -> dict[str, object]:
    data = _year_frame(year)
    bbox_filter = _normalize_bbox(bbox) if bbox else None
    data = _filter_bbox(data, bbox_filter).sort_values("distance_m", ascending=True)

    truncated = len(data) > max_features
    data = data.head(max_features)

    features: list[dict[str, object]] = []
    for _, row in data.iterrows():
        props = {
            "segment_id": row["segment_id"],
            "label": row["label"],
            "year": int(row["year"]),
            "distance_m": float(row["distance_m"]),
            "original_distance_m": float(row["original_distance_m"]),
            "new_distance_m": float(row["new_distance_m"]),
            "growth_rate_2y_m": float(row["growth_rate_2y_m"]),
            "annual_growth_m_per_year": float(row["annual_growth_m_per_year"]),
            "distance_change_vs_2025_m": float(row["distance_change_vs_2025_m"]),
            "image_path": row["image_path"] if pd.notna(row["image_path"]) else None,
            "bbox_min_lon": float(row["bbox_min_lon"]),
            "bbox_min_lat": float(row["bbox_min_lat"]),
            "bbox_max_lon": float(row["bbox_max_lon"]),
            "bbox_max_lat": float(row["bbox_max_lat"]),
            "is_projected": bool(row["is_projected"]),
        }
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row["center_lon"]), float(row["center_lat"])],
                },
                "properties": props,
            }
        )

    return {
        "type": "FeatureCollection",
        "features": features,
        "meta": {
            "year": year,
            "returned": len(features),
            "truncated": truncated,
        },
    }


@app.get("/segments/top")
def top_segments(
    year: int = Query(..., ge=YEARS[0], le=YEARS[-1]),
    n: Annotated[int | None, Query(ge=1, le=500000)] = None,
) -> list[dict[str, object]]:
    cur = _year_frame(year)
    prev = _year_frame(max(YEARS[0], year - 1))[["segment_id", "distance_m"]].rename(
        columns={"distance_m": "prev_distance_m"}
    )
    cur = cur.merge(prev, on="segment_id", how="left")

    cur["trend_arrow"] = np.where(
        cur["distance_m"] < cur["prev_distance_m"] - 0.05,
        "up",
        np.where(cur["distance_m"] > cur["prev_distance_m"] + 0.05, "down", "flat"),
    )

    sorted_rows = cur.sort_values("distance_m", ascending=True)
    if n is not None:
        sorted_rows = sorted_rows.head(n)

    out = (
        sorted_rows[
            [
                "segment_id",
                "label",
                "year",
                "distance_m",
                "original_distance_m",
                "new_distance_m",
                "growth_rate_2y_m",
                "annual_growth_m_per_year",
                "distance_change_vs_2025_m",
                "trend_arrow",
                "image_path",
                "center_lon",
                "center_lat",
            ]
        ]
        .to_dict(orient="records")
    )
    return _to_json_records(out)


@app.get("/segments/{segment_id}/timeseries")
def segment_timeseries(segment_id: str) -> list[dict[str, object]]:
    base = _load_predictions()
    row = base[base["segment_id"] == segment_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"segment_id not found: {segment_id}")
    row = row.iloc[0]

    out: list[dict[str, object]] = []
    for year in YEARS:
        if year == 2025:
            dist = float(row["original_distance_m"])
        elif year == 2026:
            dist = float(row["new_distance_m"])
        else:
            dist = float(row["new_distance_m"] - row["annual_growth_m_per_year"] * (year - 2026))
        out.append(
            {
                "segment_id": segment_id,
                "label": row["label"],
                "year": int(year),
                "distance_m": max(0.0, dist),
                "is_projected": year >= 2027,
                "original_distance_m": float(row["original_distance_m"]),
                "new_distance_m": float(row["new_distance_m"]),
                "growth_rate_2y_m": float(row["growth_rate_2y_m"]),
                "annual_growth_m_per_year": float(row["annual_growth_m_per_year"]),
            }
        )
    return _to_json_records(out)


@app.get("/segments/{segment_id}/forecast")
def segment_forecast(segment_id: str) -> list[dict[str, object]]:
    # Compatibility endpoint for existing frontend patterns.
    series = segment_timeseries(segment_id)
    out: list[dict[str, object]] = []
    for row in series:
        year = int(row["year"])
        if year < 2027:
            continue
        annual = float(row["annual_growth_m_per_year"])
        uncertainty = abs(annual) * max(1.0, year - 2026) * 0.25
        out.append(
            {
                "segment_id": row["segment_id"],
                "target_year": year,
                "predicted_distance_m": float(row["distance_m"]),
                "lower_distance_m": max(0.0, float(row["distance_m"]) - uncertainty),
                "upper_distance_m": float(row["distance_m"]) + uncertainty,
            }
        )
    return _to_json_records(out)


@app.get("/powerlines/layer")
def powerlines_layer(
    bbox: str | None = Query(
        None,
        description="Optional extent filter as 'minx,miny,maxx,maxy' in EPSG:4326",
    ),
    max_features: int = Query(18000, ge=100, le=200000),
) -> dict[str, object]:
    try:
        powerlines = _load_powerline_index()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    bbox_filter = _normalize_bbox(bbox) if bbox else None

    matched = 0
    truncated = False
    out_features: list[dict[str, object]] = []
    for fb, feature in powerlines["features"]:
        if bbox_filter and not _bbox_intersects(fb, bbox_filter):
            continue
        matched += 1
        out_features.append(feature)
        if len(out_features) >= max_features:
            truncated = True
            break

    payload: dict[str, object] = {
        "type": "FeatureCollection",
        "features": out_features,
        "meta": {
            "matched": matched,
            "returned": len(out_features),
            "truncated": truncated,
        },
    }
    if powerlines.get("crs") is not None:
        payload["crs"] = powerlines.get("crs")
    return payload
