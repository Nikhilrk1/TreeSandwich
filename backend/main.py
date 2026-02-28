"""FastAPI service for Vegetation Vision MVP."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles


DATA_PATH = os.getenv("VV_DATA_PATH", "data/segment_timeseries.parquet")
SEGMENT_GEOJSON_PATH = os.getenv("VV_SEGMENT_GEOJSON", "data/segments.geojson")
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Vegetation Vision API", version="0.1.0")
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


def _normalize_month(value: str) -> pd.Timestamp:
    try:
        # Accepts YYYY-MM or full date.
        return pd.to_datetime(value).to_period("M").to_timestamp()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid date/month: {value}") from exc


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


def _bbox_intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _to_iso_strings(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def normalize(value: object) -> object:
        if isinstance(value, (pd.Timestamp, pd.Period)):
            return pd.Timestamp(value).strftime("%Y-%m")
        if isinstance(value, (np.floating, float)):
            if pd.isna(value):
                return None
            return float(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
        if value is pd.NaT:
            return None
        return value

    out: list[dict[str, object]] = []
    for row in rows:
        clean: dict[str, object] = {}
        for key, value in row.items():
            clean[key] = normalize(value)
        out.append(clean)
    return out


@lru_cache(maxsize=1)
def _load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Set VV_DATA_PATH or generate data first."
        )

    if DATA_PATH.endswith(".parquet"):
        df = pd.read_parquet(DATA_PATH)
    elif DATA_PATH.endswith(".csv"):
        df = pd.read_csv(DATA_PATH)
    else:
        raise ValueError("VV_DATA_PATH must point to .parquet or .csv")

    if df.empty:
        return df

    if "segment_id" in df.columns:
        df["segment_id"] = df["segment_id"].astype(str)

    for col in ["date", "as_of_date", "target_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.to_period("M").dt.to_timestamp()

    for col in ["risk_score", "risk_confidence", "ndvi_median", "ndvi_anomaly"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "horizon_months" in df.columns:
        df["horizon_months"] = pd.to_numeric(df["horizon_months"], errors="coerce").fillna(0).astype(int)

    if "is_forecast" in df.columns:
        if df["is_forecast"].dtype == bool:
            pass
        else:
            df["is_forecast"] = (
                df["is_forecast"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"1", "true", "yes", "y"})
            )

    return df


@lru_cache(maxsize=1)
def _load_segment_geojson() -> dict[str, object]:
    if not os.path.exists(SEGMENT_GEOJSON_PATH):
        raise FileNotFoundError(
            f"Segment GeoJSON not found at {SEGMENT_GEOJSON_PATH}. "
            "Set VV_SEGMENT_GEOJSON or generate geometry first."
        )
    with open(SEGMENT_GEOJSON_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("type") != "FeatureCollection":
        raise ValueError("Segment GeoJSON must be a FeatureCollection.")
    return payload


def _historical_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["horizon_months"] == 0) & (~df["is_forecast"].astype(bool))].copy()


@app.get("/health")
def health() -> dict[str, object]:
    try:
        df = _load_data()
    except FileNotFoundError:
        return {"status": "degraded", "detail": f"Missing dataset: {DATA_PATH}"}

    return {
        "status": "ok",
        "rows": int(len(df)),
        "segments": int(df["segment_id"].nunique()) if "segment_id" in df.columns else 0,
    }


@app.get("/timeline/months")
def timeline_months() -> dict[str, object]:
    df = _load_data()
    if df.empty:
        return {"months": [], "default": None}
    hist = _historical_rows(df)
    months = sorted(hist["target_date"].dropna().unique().tolist())
    month_labels = [pd.Timestamp(x).strftime("%Y-%m") for x in months]
    default = month_labels[-1] if month_labels else None
    return {"months": month_labels, "default": default}


@app.post("/reload")
def reload_data() -> dict[str, str]:
    _load_data.cache_clear()
    _load_segment_geojson.cache_clear()
    return {"status": "reloaded"}


@app.get("/segments/top")
def top_segments(
    date: str = Query(..., description="Month (YYYY-MM)"),
    n: int = Query(25, ge=1, le=200),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
) -> list[dict[str, object]]:
    df = _load_data()
    if df.empty:
        return []

    month = _normalize_month(date)
    hist = _historical_rows(df)
    current = hist[(hist["target_date"] == month) & (hist["risk_confidence"] >= min_confidence)].copy()
    if current.empty:
        return []

    prev_month = (month - pd.DateOffset(months=1)).to_period("M").to_timestamp()
    previous = hist[hist["target_date"] == prev_month][["segment_id", "risk_score"]].rename(
        columns={"risk_score": "prev_risk_score"}
    )

    current = current.merge(previous, on="segment_id", how="left")

    if "as_of_date" in df.columns:
        forecast = df[
            (df["is_forecast"].astype(bool))
            & (df["as_of_date"] == month)
            & (df["horizon_months"].isin([3, 6]))
        ][["segment_id", "horizon_months", "risk_score"]].copy()
        if not forecast.empty:
            pivot = (
                forecast.pivot_table(
                    index="segment_id",
                    columns="horizon_months",
                    values="risk_score",
                    aggfunc="max",
                )
                .rename(columns={3: "forecast_risk_3m", 6: "forecast_risk_6m"})
                .reset_index()
            )
            current = current.merge(pivot, on="segment_id", how="left")

    delta = current["risk_score"] - current["prev_risk_score"]
    current["trend_arrow"] = np.where(
        delta >= 2,
        "up",
        np.where(delta <= -2, "down", "flat"),
    )
    for col in ["forecast_risk_3m", "forecast_risk_6m"]:
        if col not in current.columns:
            current[col] = np.nan

    out_records = (
        current.sort_values("risk_score", ascending=False)
        .head(n)[
            [
                "segment_id",
                "target_date",
                "risk_score",
                "risk_level",
                "risk_confidence",
                "trend_arrow",
                "ndvi_anomaly",
                "ndvi_median",
                "forecast_risk_3m",
                "forecast_risk_6m",
            ]
        ]
        .to_dict(orient="records")
    )
    return _to_iso_strings(out_records)


@app.get("/map/layer")
def map_layer(
    date: str = Query(..., description="Month (YYYY-MM)"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    min_risk: int = Query(0, ge=0, le=100),
    bbox: str | None = Query(
        None,
        description="Optional extent filter as 'minx,miny,maxx,maxy' in EPSG:4326",
    ),
) -> dict[str, object]:
    df = _load_data()
    geojson = _load_segment_geojson()

    month = _normalize_month(date)
    hist = _historical_rows(df)
    month_rows = hist[
        (hist["target_date"] == month)
        & (hist["risk_confidence"] >= min_confidence)
        & (hist["risk_score"] >= min_risk)
    ].copy()

    if month_rows.empty:
        return {"type": "FeatureCollection", "features": []}

    risk_map = month_rows.set_index("segment_id")[
        ["risk_score", "risk_level", "risk_confidence", "ndvi_median", "ndvi_anomaly", "target_date"]
    ].to_dict(orient="index")

    bbox_filter = _normalize_bbox(bbox) if bbox else None
    features: list[dict[str, object]] = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {}) or {}
        segment_id = props.get("segment_id")
        if segment_id not in risk_map:
            continue

        if bbox_filter:
            fb = _feature_bbox(feature.get("geometry", {}))
            if not _bbox_intersects(fb, bbox_filter):
                continue

        segment_risk = risk_map[segment_id]
        score = segment_risk.get("risk_score", np.nan)
        confidence = segment_risk.get("risk_confidence", np.nan)
        ndvi_median = segment_risk.get("ndvi_median", np.nan)
        ndvi_anomaly = segment_risk.get("ndvi_anomaly", np.nan)

        merged_props = dict(props)
        merged_props.update(
            {
                "risk_score": None if pd.isna(score) else float(score),
                "risk_level": segment_risk.get("risk_level"),
                "risk_confidence": None if pd.isna(confidence) else float(confidence),
                "ndvi_median": None if pd.isna(ndvi_median) else float(ndvi_median),
                "ndvi_anomaly": None if pd.isna(ndvi_anomaly) else float(ndvi_anomaly),
                "date": pd.Timestamp(segment_risk.get("target_date")).strftime("%Y-%m"),
            }
        )
        features.append(
            {
                "type": "Feature",
                "geometry": feature.get("geometry"),
                "properties": merged_props,
            }
        )

    return {"type": "FeatureCollection", "features": features}


@app.get("/segments/{segment_id}/timeseries")
def segment_timeseries(segment_id: str) -> list[dict[str, object]]:
    df = _load_data()
    seg = _historical_rows(df)
    seg = seg[seg["segment_id"] == segment_id].copy()
    if seg.empty:
        raise HTTPException(status_code=404, detail=f"segment_id not found: {segment_id}")

    cols = [
        "segment_id",
        "target_date",
        "ndvi_median",
        "ndvi_p90",
        "frac_ndvi_gt_0_6",
        "ndvi_anomaly",
        "risk_score",
        "risk_level",
        "risk_confidence",
        "obs_count",
        "valid_pixel_frac",
    ]
    keep = [c for c in cols if c in seg.columns]
    records = seg.sort_values("target_date")[keep].to_dict(orient="records")
    return _to_iso_strings(records)


@app.get("/segments/{segment_id}/forecast")
def segment_forecast(segment_id: str) -> list[dict[str, object]]:
    df = _load_data()
    seg = df[(df["segment_id"] == segment_id) & (df["is_forecast"].astype(bool))].copy()
    if seg.empty:
        return []

    latest_as_of = seg["as_of_date"].max()
    seg = seg[seg["as_of_date"] == latest_as_of].sort_values("horizon_months")

    cols = [
        "segment_id",
        "as_of_date",
        "target_date",
        "horizon_months",
        "ndvi_median",
        "forecast_lower",
        "forecast_upper",
        "risk_score",
        "risk_level",
        "risk_confidence",
    ]
    keep = [c for c in cols if c in seg.columns]
    records = seg[keep].to_dict(orient="records")
    return _to_iso_strings(records)
