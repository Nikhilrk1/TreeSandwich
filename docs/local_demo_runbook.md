# Local Demo Runbook

## 1) Install dependencies
```bash
pip install -r requirements.txt
```

## 2) Create synthetic NDVI pixel samples
```bash
python scripts/generate_mock_pixels.py \
  --output data/pixel_samples.parquet \
  --segments 120 \
  --months 18 \
  --pixels 30
```

## 3) Build segment timeseries (risk + forecast)
```bash
python -m vegetation_vision.pipeline \
  --input data/pixel_samples.parquet \
  --input-grain pixel \
  --output data/segment_timeseries.parquet
```

## 4) Build mock segment geometry for map
```bash
python scripts/generate_mock_segments.py \
  --timeseries data/segment_timeseries.parquet \
  --output data/segments.geojson
```

## 5) Run backend + app
```bash
uvicorn backend.main:app --reload
```
Open:
- `http://127.0.0.1:8000/app`

## 6) Demo flow
1. Drag slider month-by-month.
2. Show changing hotspots.
3. Open top risk panel and click top segment.
4. Explain historical vs forecast chart with confidence band.
