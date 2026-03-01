# Vegetation Vision API Contract (MVP)

Base URL: `http://127.0.0.1:8000`

## `GET /health`
Returns service status and loaded row/segment counts.

## `GET /timeline/years`
Returns available historical years for slider playback.

Example response:
```json
{
  "years": [2023, 2024, 2025],
  "default": 2025
}
```

## `GET /map/layer`
Returns GeoJSON line features with annual (or monthly) risk/hazard properties merged.

Query params:
- `year` (recommended): `YYYY`
- `date` (optional alternative): `YYYY-MM`
- `min_confidence` (optional, default `0.0`)
- `min_risk` (optional, default `0`)
- `bbox` (optional): `"minx,miny,maxx,maxy"` in EPSG:4326

Feature properties include:
- `segment_id`
- `corridor_id`
- `risk_score`
- `risk_level`
- `hazard_level`
- `risk_confidence`
- `growth_amount_m`
- `growth_rate_m_per_month`
- `vegetation_distance_m`

## `GET /segments/top`
Top-N ranked segments for panel view (year or month scope).

Query params:
- `year` (recommended): `YYYY`
- `date` (optional alternative): `YYYY-MM`
- `n` (optional, default `25`, max `200`)
- `min_confidence` (optional)

Fields include:
- `segment_id`, `risk_score`, `risk_level`, `hazard_level`, `risk_confidence`
- `trend_arrow` (`up`, `down`, `flat`)
- `forecast_growth_3m_m`, `forecast_growth_6m_m`
- `forecast_risk_3m`, `forecast_risk_6m`

## `GET /segments/{segment_id}/timeseries`
Historical rows (`horizon_months = 0`) for charting.

Important fields:
- `growth_amount_m`
- `growth_rate_m_per_month`
- `vegetation_distance_m`
- `hazard_level`
- `risk_score`

## `GET /segments/{segment_id}/forecast`
Latest available forecast rows.

Important fields:
- `predicted_growth_amount_m`
- `growth_lower_m`, `growth_upper_m`
- `predicted_distance_m`
- `hazard_level`
- `risk_score`

## `POST /reload`
Clears in-memory caches and reloads parquet/GeoJSON on next request.
