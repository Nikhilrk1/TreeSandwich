# Frontend Shell

Static map UI served by FastAPI at:
- `http://127.0.0.1:8000/app`

## Data Source
The map now uses `future_predictions.csv` via backend endpoints.
No dummy segment parquet/geojson pipeline is required for frontend operation.

## Endpoints Consumed
- `/timeline/years`
- `/map/layer?year=YYYY&bbox=minx,miny,maxx,maxy`
- `/segments/top?year=YYYY`
- `/segments/{segment_id}/timeseries`
- `/powerlines/layer?bbox=minx,miny,maxx,maxy`

## Notes
- Circle color is based on distance-to-powerline (`distance_m`).
- Circle minimum radius is constrained by each record's bounding-box footprint.
- Circles stay circles at all zoom levels (no line mode).
