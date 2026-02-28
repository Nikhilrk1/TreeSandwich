# Implementation Timeline (Hackathon)

## Team Roles
- PM: scope control, demo flow, acceptance checks.
- ML/Geo: NDVI ingestion, feature engineering, risk/forecast modeling.
- Full-stack: API, map UI, charts, panel interactions.

## Phase Plan
| Phase | Tasks | Owner | Effort | Acceptance Criteria |
|---|---|---|---|---|
| Setup | AOI + corridor subset, segmenting rules, config and contracts | PM + Full-stack | 0.5 day | Fixed config and example data loaded |
| Ingestion | Pull Sentinel-2 monthly, cloud/snow mask, NDVI composites | ML/Geo | 1 day | Monthly NDVI outputs for pilot region |
| Features | Segment buffers + zonal stats + temporal features | ML/Geo | 0.75 day | Feature parquet with all required columns |
| Risk | Baseline risk index + confidence + thresholds | ML | 0.5 day | Ranked segments and QA plots |
| Forecast | 3/6 month baseline forecast with intervals | ML | 0.5 day | Forecast rows + error report |
| Backend | FastAPI endpoints for top list, history, forecast | Full-stack | 0.5 day | Endpoints respond in local demo |
| Frontend | Map slider, top-risk panel, per-segment chart | Full-stack | 1 day | End-to-end interaction flow works |
| Polish | Storyline, labels, final QA, fallback screenshots | PM + All | 0.25 day | Stable 2-3 minute demo |

## Critical Path
- Segment definition -> NDVI composites -> feature table -> risk score -> slider map + top-risk panel.

## Stretch Goals
- Label-based supervised model (if work orders are available).
- Weather covariates for better forecasts.
- Export prioritized segment list for operations.
