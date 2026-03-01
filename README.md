# TreeSandwich
USC Gridhacks powerline vegetation prediction project.

## Vegetation Vision MVP
This repository now includes an executable MVP scaffold for NDVI-assisted,
distance-aware vegetation hazard scoring along power corridors.

### Structure
- `docs/vegetation_vision_mvp.md`: product + implementation plan.
- `vegetation_vision/features.py`: per-segment feature engineering.
- `vegetation_vision/risk.py`: distance-driven hazard + risk index.
- `vegetation_vision/forecast.py`: 3/6 month vegetation growth amount forecast.
- `vegetation_vision/pipeline.py`: CLI to build scored timeseries parquet.
- `vegetation_vision/segments.py`: corridor segmentation + buffer utilities.
- `backend/main.py`: FastAPI endpoints for map/panel/chart consumption.
- `frontend/`: static interactive map app served at `/app`.
- `schema/segment_timeseries.sql`: minimal table schema.

### Quick Start
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. One-command mock demo data generation (fastest way to test frontend):
```bash
python scripts/build_mock_demo_data.py --outdir data --segments 160 --months 24
```

3. Build feature/risk/forecast output from pixel-level monthly data:
```bash
python -m vegetation_vision.pipeline \
  --input data/pixel_samples.parquet \
  --input-grain pixel \
  --output data/segment_timeseries.parquet
```

Expected input columns:
- `segment_id`
- `date` (monthly date)
- `ndvi`
- `is_valid` (1/0)

Optional input columns:
- `is_cloud_free` (1/0)
- `expected_pixels` (int)
- `vegetation_distance_m` (nearest vegetation distance to line in meters)

4. Generate mock segment geometry for map playback (if you do not yet have
   real corridor geometry):
```bash
python scripts/generate_mock_segments.py \
  --timeseries data/segment_timeseries.parquet \
  --output data/segments.geojson
```

5. Run API:
```bash
python -m uvicorn backend.main:app --reload
```

6. Open app:
- `http://127.0.0.1:8000/app`

7. Example endpoints:
- `/health`
- `/timeline/years`
- `/map/layer?year=2025`
- `/powerlines/layer?bbox=-125,24,-66,49`
- `/segments/top?year=2025&n=25`
- `/segments/{segment_id}/timeseries`
- `/segments/{segment_id}/forecast`

### Real Corridor Segmenting
Given a real line dataset with `corridor_id`:
```bash
python scripts/build_segments.py \
  --input data/corridors.geojson \
  --corridor-id-col corridor_id \
  --segment-length-m 500 \
  --buffer-m 50 \
  --out-segments-geojson data/segments_centerline.geojson \
  --out-buffers-geojson data/segments_buffer.geojson \
  --out-metadata-csv data/segment_metadata.csv
```

Given monthly NDVI rasters (for example `ndvi_2025-07.tif`) and segment buffers:
```bash
python scripts/extract_segment_ndvi_stats.py \
  --segments data/segments_buffer.geojson \
  --centerlines data/segments_centerline.geojson \
  --segment-id-col segment_id \
  --raster-glob 'data/ndvi/ndvi_*.tif' \
  --date-regex '(\\d{4}-\\d{2})' \
  --output data/segment_features.parquet

python -m vegetation_vision.pipeline \
  --input data/segment_features.parquet \
  --input-grain feature \
  --output data/segment_timeseries.parquet
```

Precompute map layers for caching/performance:
```bash
python scripts/precompute_monthly_layers.py \
  --timeseries data/segment_timeseries.parquet \
  --segments data/segments.geojson \
  --outdir data/layers
```

### Notes
- `hazard_level` is based on vegetation distance-to-line thresholds
  (`critical <=2m`, `high <=5m`, `medium <=10m`, else `low`).
- Forecast output focuses on `predicted_growth_amount_m` and projected
  `predicted_distance_m`, not forecast NDVI.
- NDVI quality masking and monthly compositing are assumed upstream.
- Geometry for map playback is loaded from `VV_SEGMENT_GEOJSON`
  (default `data/segments.geojson`).
- US powerline overlay is loaded from `VV_POWERLINES_GEOJSON`
  (default `US_Electric_Power_Transmission_Lines_-6976209181916424225.geojson`).
