# Vegetation Vision API Contract (MVP)

Base URL: `http://127.0.0.1:8000`

## `GET /health`
Returns service status and loaded row/segment counts.

## `GET /timeline/months`
Returns available historical months for slider playback.

Example response:
```json
{
  "months": ["2024-01", "2024-02", "2024-03"],
  "default": "2024-03"
}
```

## `GET /map/layer`
Returns GeoJSON line features with risk properties merged for a selected month.

Query params:
- `date` (required): `YYYY-MM`
- `min_confidence` (optional, default `0.0`)
- `min_risk` (optional, default `0`)
- `bbox` (optional): `"minx,miny,maxx,maxy"` in EPSG:4326

Feature properties include:
- `segment_id`
- `corridor_id`
- `risk_score`
- `risk_level`
- `risk_confidence`
- `ndvi_median`
- `ndvi_anomaly`
- `date`

## `GET /segments/top`
Top-N ranked segments for month panel.

Query params:
- `date` (required): `YYYY-MM`
- `n` (optional, default `25`, max `200`)
- `min_confidence` (optional)

Fields include:
- `segment_id`, `risk_score`, `risk_level`, `risk_confidence`
- `trend_arrow` (`up`, `down`, `flat`)
- `forecast_risk_3m`, `forecast_risk_6m`

## `GET /segments/{segment_id}/timeseries`
Historical rows (`horizon_months = 0`) for charting.

## `GET /segments/{segment_id}/forecast`
Latest available forecast rows (for 3/6 month lines and confidence band).

## `POST /reload`
Clears in-memory caches and reloads parquet/GeoJSON on next request.
