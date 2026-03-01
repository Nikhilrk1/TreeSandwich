# Frontend Shell

This is a static map UI for the Vegetation Vision MVP.

## Served by FastAPI
When running `python -m uvicorn backend.main:app --reload`, open:
- `http://127.0.0.1:8000/app`

The page calls backend endpoints:
- `/timeline/years`
- `/map/layer?year=YYYY`
- `/powerlines/layer?bbox=minx,miny,maxx,maxy`
- `/segments/top?year=YYYY`
- `/segments/{segment_id}/timeseries`
- `/segments/{segment_id}/forecast`

## Notes
- The map expects segment geometry from `VV_SEGMENT_GEOJSON`
  (default: `data/segments.geojson`).
- The all-US line overlay expects a GeoJSON at `VV_POWERLINES_GEOJSON`
  (default: `US_Electric_Power_Transmission_Lines_-6976209181916424225.geojson`).
- Use `scripts/generate_mock_segments.py` for a fast local demo dataset.
