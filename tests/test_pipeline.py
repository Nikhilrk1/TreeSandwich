from __future__ import annotations

import numpy as np
import pandas as pd

from vegetation_vision.pipeline import build_timeseries, build_timeseries_from_features


def _make_synthetic_pixels() -> pd.DataFrame:
    rows = []
    months = pd.date_range("2025-01-01", periods=8, freq="MS")

    for month_idx, date in enumerate(months):
        for pixel in range(24):
            rows.append(
                {
                    "segment_id": "seg_a",
                    "date": date,
                    "ndvi": 0.35 + 0.03 * month_idx + (pixel % 4) * 0.005,
                    "is_valid": 1,
                    "is_cloud_free": 1,
                    "expected_pixels": 24,
                    "vegetation_distance_m": 8.0 - 0.45 * month_idx,
                }
            )
            rows.append(
                {
                    "segment_id": "seg_b",
                    "date": date,
                    "ndvi": 0.45 + (pixel % 3) * 0.003,
                    "is_valid": 1,
                    "is_cloud_free": 1,
                    "expected_pixels": 24,
                    "vegetation_distance_m": 16.0 - 0.05 * month_idx,
                }
            )

    return pd.DataFrame(rows)


def test_build_timeseries_contains_forecast_and_risk():
    pixel_df = _make_synthetic_pixels()
    out = build_timeseries(pixel_df)

    assert not out.empty
    assert {"risk_score", "risk_level", "hazard_level", "is_forecast", "horizon_months"}.issubset(out.columns)

    forecast = out[out["is_forecast"]]
    assert len(forecast) == 4  # 2 segments * {3, 6} horizons

    hist = out[(~out["is_forecast"]) & (out["horizon_months"] == 0)].copy()
    latest = hist.sort_values("target_date").groupby("segment_id").tail(1).set_index("segment_id")
    assert latest.loc["seg_a", "risk_score"] >= latest.loc["seg_b", "risk_score"]
    assert latest.loc["seg_a", "hazard_level"] in {"medium", "high", "critical"}
    assert np.isfinite(latest.loc["seg_a", "risk_confidence"])


def test_build_timeseries_from_feature_rows():
    pixel_df = _make_synthetic_pixels()
    # Aggregate manually to exercise feature-grain path.
    feature_df = (
        pixel_df.groupby(["segment_id", "date"], as_index=False)
        .agg(
            ndvi_median=("ndvi", "median"),
            ndvi_p90=("ndvi", lambda s: np.percentile(s, 90)),
            frac_ndvi_gt_0_6=("ndvi", lambda s: float((s > 0.6).mean())),
            ndvi_iqr=("ndvi", lambda s: np.percentile(s, 75) - np.percentile(s, 25)),
            obs_count=("ndvi", "count"),
            expected_pixels=("expected_pixels", "max"),
            valid_pixel_frac=("is_valid", "mean"),
            cloud_free_ratio=("is_cloud_free", "mean"),
            vegetation_distance_m=("vegetation_distance_m", "min"),
        )
        .sort_values(["segment_id", "date"])
    )

    out = build_timeseries_from_features(feature_df)
    assert not out.empty
    assert out["is_forecast"].any()
    assert out["risk_score"].between(0, 100).all()
    assert "predicted_growth_amount_m" in out.columns
