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


  ## How To Generate Satellite Images
  - Download the Fhsh and wildlife power line dataset from https://gis-fws.opendata.arcgis.com/datasets/fws::us-electric-power-transmission-lines/explore?location=34.140142%2C-81.587457%2C9. Name it lines.geojson 
  - Run createSatteliteImages.py
  - example for SC:  python createSatteliteImages.py --input lines.geojson --outdir out_tiles_gee --size 1024 --dpi 150 --dataset naip --lookback 5 --sc-only

## How to Train and Use the Custom Vision Transformer
- Use a corridor CSV (for example `ViT/tree_line_distance_compare.csv`) plus the matching tiles in `ViT/images`.
- Train the model with:
  - `bash ViT/train_vit.sh`
- Run inference on new CSV data with:
  - `bash ViT/vit_infer.sh`
- Script notes:
  - `ViT/train_vit.sh` trains and saves the checkpoint (currently `ViT/models`).
  - `ViT/vit_infer.sh` can build/update metadata cache and then run `--mode infer` to write prediction CSV output.
