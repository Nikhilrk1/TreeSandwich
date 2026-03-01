# TreeSandwich

TreeSandwich is a map-based frontend for visualizing vegetation-to-powerline
distance risk from a CSV input and projected yearly distances.

## Current Input Contract
Primary input file:
- `future_predictions.csv` (override with `VV_FUTURE_PREDICTIONS_CSV`)

Supported columns (backend auto-detects available names):
- Segment identity:
  - `segment_id` or (`feature_id` + `section_id`)
- Location (one of):
  - Bounding box pair columns:
    - `min_lon,min_lat,max_lon,max_lat`
    - or `bbox_min_lon,bbox_min_lat,bbox_max_lon,bbox_max_lat`
    - or `lon1,lat1,lon2,lat2`
    - or `west,south,east,north`
  - Fallback center point:
    - `coord_lon,coord_lat` (or `center_lon,center_lat` / `lon,lat`)
- Distances:
  - Original distance for 2025:
    - `original_distance_m` (or `now_distance_m`)
  - New distance for 2026:
    - `new_distance_m` (optional if growth exists)
  - Growth over 2 years:
    - `growth_rate` / `growth_rate_2y_m`
    - or `predicted_future_delta_distance_m`

If `new_distance_m` is missing, backend computes:
- `new_distance_m = original_distance_m - (growth_rate_2y_m / 2)`

## Year Projection Logic
Slider years are fixed to **2025–2030**.

- `2025`: original distance
- `2026`: new distance
- `2027+`: `new_distance - ((growth_rate_2y / 2) * years_passed_since_2026)`

Distances are clipped at `0`.

## Run
1. Install deps:
```bash
pip install -r requirements.txt
```

2. Start API + frontend:
```bash
python -m uvicorn backend.main:app --reload
```

3. Open:
- `http://127.0.0.1:8000/app`

## Overlay Data
All-US transmission lines overlay is loaded from:
- `VV_POWERLINES_GEOJSON`
- default:
  - `US_Electric_Power_Transmission_Lines_-6976209181916424225.geojson`

## API Endpoints Used by Frontend
- `GET /timeline/years`
- `GET /map/layer?year=YYYY[&bbox=minx,miny,maxx,maxy]`
- `GET /segments/top?year=YYYY&n=20`
- `GET /segments/{segment_id}/timeseries`
- `GET /powerlines/layer?bbox=minx,miny,maxx,maxy`
