CREATE TABLE IF NOT EXISTS segment_metadata (
  segment_id TEXT PRIMARY KEY,
  corridor_id TEXT NOT NULL,
  line_class TEXT,
  length_m REAL NOT NULL,
  buffer_m REAL NOT NULL,
  geom_wkt TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS segment_timeseries (
  segment_id TEXT NOT NULL,
  as_of_date DATE NOT NULL,
  target_date DATE NOT NULL,
  horizon_months INTEGER NOT NULL,
  is_forecast BOOLEAN NOT NULL,

  ndvi_median REAL,
  ndvi_p90 REAL,
  frac_ndvi_gt_0_6 REAL,
  ndvi_iqr REAL,
  ndvi_rolling_mean_3m REAL,
  ndvi_slope_3m REAL,
  segment_month_baseline_median REAL,
  segment_month_baseline_iqr REAL,
  ndvi_anomaly REAL,
  ndvi_anomaly_z REAL,
  obs_count INTEGER,
  expected_pixels INTEGER,
  valid_pixel_frac REAL,
  cloud_free_ratio REAL,

  risk_score REAL NOT NULL,
  risk_level TEXT NOT NULL,
  risk_confidence REAL NOT NULL,

  forecast_lower REAL,
  forecast_upper REAL,
  model_version TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
