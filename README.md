# TreeSandwich

TreeSandwich is a vegetation-to-powerline distance risk visualization platform.
It combines satellite imagery, a custom Vision Transformer, and a map-based web
interface to highlight corridors where vegetation is encroaching on power lines.

---

## Quick Start — Running the Website

```bash
# 1. Install dependencies and get the data from this google drive link: https://drive.google.com/drive/folders/1pqmy3RHQbmAuysHPyjF1ydSU-k6zv7y7?usp=drive_link
pip install -r requirements.txt

# 2. Start the API + frontend server
python -m uvicorn backend.main:app --reload

# 3. Open in a Chromium-based browser (Chrome, Edge, Brave, etc.)
#    http://127.0.0.1:8000/app
```

The website reads from `future_predictions.csv` in the project root.
No additional pipeline steps are required to view the map.

### Environment Variables (optional)

| Variable | Default | Purpose |
|---|---|---|
| `VV_FUTURE_PREDICTIONS_CSV` | `future_predictions.csv` | Path to the predictions CSV |
| `VV_POWERLINES_GEOJSON` | `US_Electric_Power_Transmission_Lines_-6976209181916424225.geojson` | US transmission lines overlay |

---

## End-to-End Pipeline

The full pipeline has four stages. Each stage feeds the next.

### Stage 1 — Generate Satellite Imagery

Download the US Electric Power Transmission Lines dataset from
[FWS ArcGIS](https://gis-fws.opendata.arcgis.com/datasets/fws::us-electric-power-transmission-lines/explore)
and save it as `lines.geojson`, then run:

```bash
python createSatteliteImages.py \
  --input lines.geojson \
  --outdir out_tiles_gee \
  --size 1024 --dpi 150 \
  --dataset naip --lookback 5 \
  --sc-only          # example: South Carolina only
```

This produces satellite tiles in `ViT/images/` used by later stages.

### Stage 2 — Train / Run the Vision Transformer

Use the corridor CSV (e.g. `ViT/tree_line_distance_compare.csv`) and the
matching tiles from Stage 1.

```bash
# Train the model (saves checkpoint to ViT/models/)
bash ViT/train_vit.sh

# Run inference to produce future_predictions.csv
bash ViT/vit_infer.sh
```

The inference step writes `future_predictions.csv`, which is the primary input
for the website.

### Stage 3 — (Optional) Real Corridor Segmenting & NDVI Features

If you have real corridor geometry and NDVI rasters, you can build fine-grained
segment features:

```bash
# Segment corridors into fixed-length sections
python scripts/build_segments.py \
  --input data/corridors.geojson \
  --corridor-id-col corridor_id \
  --segment-length-m 500 \
  --buffer-m 50 \
  --out-segments-geojson data/segments_centerline.geojson \
  --out-buffers-geojson data/segments_buffer.geojson \
  --out-metadata-csv data/segment_metadata.csv

# Extract NDVI statistics per segment
python scripts/extract_segment_ndvi_stats.py \
  --segments data/segments_buffer.geojson \
  --centerlines data/segments_centerline.geojson \
  --segment-id-col segment_id \
  --raster-glob 'data/ndvi/ndvi_*.tif' \
  --date-regex '(\\d{4}-\\d{2})' \
  --output data/segment_features.parquet

# Run the vegetation vision pipeline
python -m vegetation_vision.pipeline \
  --input data/segment_features.parquet \
  --input-grain feature \
  --output data/segment_timeseries.parquet
```

### Stage 4 — (Optional) Precompute Map Layers

For large datasets, precompute cached layers:

```bash
python scripts/precompute_monthly_layers.py \
  --timeseries data/segment_timeseries.parquet \
  --segments data/segments.geojson \
  --outdir data/layers
```

---

## Input Contract — `future_predictions.csv`

The backend auto-detects column names. Supported columns:

**Segment identity:**
- `segment_id`, or `feature_id` + `section_id`

**Location** (one of):
- Bounding box: `min_lon,min_lat,max_lon,max_lat` (several naming variants accepted)
- Center point fallback: `coord_lon,coord_lat` (or `center_lon,center_lat`, `lon,lat`)

**Distances:**
- Original distance (2025): `original_distance_m` or `now_distance_m`
- New distance (2026): `new_distance_m` (optional if growth rate exists)
- Growth over 2 years: `growth_rate`, `growth_rate_2y_m`, or `predicted_future_delta_distance_m`

**Image path:**
- `image_path` — relative path to the satellite tile (e.g. `ViT/images/tile_20_6_now_2023.png`)

If `new_distance_m` is missing, the backend computes:
`new_distance_m = original_distance_m - (growth_rate_2y_m / 2)`

## Year Projection Logic

The slider covers **2025–2030**:
- **2025**: original distance
- **2026**: new distance (from ViT prediction)
- **2027+**: `new_distance - (annual_growth × years_since_2026)`, clipped at 0

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /timeline/years` | Available years and default |
| `GET /map/layer?year=&bbox=&max_features=` | GeoJSON prediction points for map |
| `GET /segments/top?year=` | All segments ranked by risk |
| `GET /segments/{id}/timeseries` | Year-by-year distance for a segment |
| `GET /powerlines/layer?bbox=&max_features=` | Powerline overlay GeoJSON |
| `POST /reload` | Force-reload CSV/GeoJSON data |

## Notes

- Hazard levels: `critical ≤2m`, `high ≤5m`, `medium ≤10m`, else `low`
- Satellite images are served via `/ViT/images/` and `/files/` static mounts
- The frontend works best in Chromium-based browsers (Chrome, Edge, Brave)
