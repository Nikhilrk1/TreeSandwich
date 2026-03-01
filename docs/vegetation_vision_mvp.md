# Vegetation Vision MVP Plan

## Problem Formulation
- Goal: prioritize vegetation maintenance along power corridors using NDVI
  trends and anomalies plus calculated vegetation distance-to-line.
- Segment unit: split line centerlines into fixed 500m segments.
- Analysis footprint: 50m buffer on each side of segment centerline.
- Cadence: monthly composite observations.
- Forecast horizons: 3 and 6 months.
- Risk output: `risk_score` in range 0-100 per segment/month.
- Hazard output: `hazard_level` derived from distance thresholds.

What the score means:
- Higher values indicate stronger vegetation growth pressure in corridor
  footprint relative to that segment's seasonal baseline.

What it does not mean:
- Not tree height or immediate outage probability.
- Not direct compliance evidence.

## Data Pipeline
- Primary signal: Sentinel-2 L2A NDVI (10m), monthly composites.
- Masking: cloud/snow/shadow/water removed with QA layers.
- Storage:
- Monthly image artifacts as COGs (optional for demo reproducibility).
- Segment features as parquet.
- Scored + forecast table as parquet.

Pixel-to-segment mapping:
- Use segment buffer polygons and zonal stats.
- Calculate nearest vegetation distance to segment centerline from vegetated
  NDVI pixels.

Per-segment monthly features:
- `ndvi_median`, `ndvi_p90`, `frac_ndvi_gt_0_6`, `ndvi_iqr`
- `vegetation_distance_m`, `growth_amount_m`, `growth_rate_m_per_month`
- `obs_count`, `valid_pixel_frac`, `cloud_free_ratio`
- `ndvi_rolling_mean_3m`, `ndvi_slope_3m`
- `segment_month_baseline_median`, `segment_month_baseline_iqr`
- `ndvi_anomaly`, `ndvi_anomaly_z`

## Baseline Risk Index
1. Compute per-segment month-of-year baseline and IQR.
2. Compute anomaly and recent trend.
3. Compute predicted growth pressure proxy (3m/6m).
4. Apply weighted formula and sigmoid scaling to 0-100.
5. Assign hazard level from distance (`critical <=2m`, `high <=5m`,
   `medium <=10m`, else `low`).

High risk rule:
- `risk_score >= 70` and `risk_confidence >= 0.5`

## Optional Learned Model
- No labels: forecast vegetation growth amount (meters) and projected
  distance-to-line at 3/6 months, then map to risk/hazard.
- Labels available: train classifier for maintenance-within-H-months.
- Train strategy: time-based split only, no random shuffle.

## Evaluation
No labels:
- Forecast MAE/RMSE.
- Ranking stability month-to-month.
- Qualitative hotspot sanity checks.

With labels:
- ROC-AUC / PR-AUC.
- Calibration and top-k capture.

Sanity checks:
- Seasonality leakage baseline comparison.
- Cloud/snow artifact spikes under low observation counts.
- Persistent evergreen corridors should not stay high risk without trend.

## Product/UI
- Time slider map (yearly) with segment color by risk.
- Click segment for historical + forecast chart with confidence band.
- Side panel sorted by current risk, hazard level, and forecasted growth amount.

## MVP Stack
- Frontend: React + MapLibre + chart library.
- Backend: FastAPI.
- Processing: Python + pandas/numpy/geospatial libs.
- Output: parquet + precomputed map layer assets.

## Delivery Phases
1. Setup: AOI, segmenting rules, contracts.
2. Ingestion: imagery pull + masking + monthly composites.
3. Features: zonal stats and temporal features.
4. Risk: baseline index and thresholds.
5. Forecast: 3/6 month baseline model.
6. API + frontend integration.
7. Polish and demo story.

## Demo Narrative (2-3 Minutes)
1. Show timeline slider and moving hotspots.
2. Open top-risk panel and explain ranking logic.
3. Click a segment and show history + forecast.
4. Explain crew prioritization workflow and constraints.
