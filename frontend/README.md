# Frontend Shell

This is a static map UI for the Vegetation Vision MVP.

## Served by FastAPI
When running `uvicorn backend.main:app --reload`, open:
- `http://127.0.0.1:8000/app`

The page calls backend endpoints:
- `/timeline/years`
- `/map/layer?year=YYYY`
- `/segments/top?year=YYYY`
- `/segments/{segment_id}/timeseries`
- `/segments/{segment_id}/forecast`

## Notes
- The map expects segment geometry from `VV_SEGMENT_GEOJSON`
  (default: `data/segments.geojson`).
- Use `scripts/generate_mock_segments.py` for a fast local demo dataset.
