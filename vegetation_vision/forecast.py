"""Simple seasonal-trend forecast baseline for Vegetation Vision."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .risk import compute_risk_scores


def _trend_from_tail(values: pd.Series, tail: int = 6) -> float:
    clean = values.dropna().tail(tail).to_numpy(dtype=float)
    if clean.size < 2:
        return 0.0
    x = np.arange(clean.size, dtype=float)
    slope, _ = np.polyfit(x, clean, 1)
    return float(slope)


def _month_baseline_maps(seg_df: pd.DataFrame) -> tuple[dict[int, float], dict[int, float]]:
    baseline = seg_df.groupby(seg_df["date"].dt.month)["ndvi_median"].median().to_dict()
    q75 = seg_df.groupby(seg_df["date"].dt.month)["ndvi_median"].quantile(0.75)
    q25 = seg_df.groupby(seg_df["date"].dt.month)["ndvi_median"].quantile(0.25)
    iqr = (q75 - q25).to_dict()
    return baseline, iqr


def build_forecast_rows(
    scored_df: pd.DataFrame,
    horizons_months: tuple[int, int] = (3, 6),
) -> pd.DataFrame:
    """Generate per-segment forecast rows with heuristic uncertainty."""

    if scored_df.empty:
        return pd.DataFrame()

    df = scored_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    df = df.sort_values(["segment_id", "date"])

    all_rows: list[dict[str, object]] = []

    for segment_id, seg_df in df.groupby("segment_id", sort=False):
        seg_df = seg_df.sort_values("date").copy()
        latest = seg_df.iloc[-1]
        last_date = latest["date"]
        last_anom = float(latest.get("ndvi_anomaly", 0.0) or 0.0)
        trend = float(latest.get("ndvi_anomaly_slope_3m", np.nan))
        if np.isnan(trend):
            trend = _trend_from_tail(seg_df["ndvi_anomaly"])

        baseline_map, iqr_map = _month_baseline_maps(seg_df)
        fallback_baseline = float(seg_df["ndvi_median"].median())
        fallback_iqr = float(seg_df["ndvi_median"].quantile(0.75) - seg_df["ndvi_median"].quantile(0.25))
        fallback_iqr = max(fallback_iqr, 0.1)

        density = float(seg_df["frac_ndvi_gt_0_6"].dropna().tail(3).mean()) if "frac_ndvi_gt_0_6" in seg_df.columns else 0.0
        density = float(np.clip(np.nan_to_num(density, nan=0.0), 0.0, 1.0))

        residual_scale = float(seg_df["ndvi_anomaly"].diff().dropna().abs().median()) if "ndvi_anomaly" in seg_df.columns else 0.2
        residual_scale = max(np.nan_to_num(residual_scale, nan=0.2), 0.1)

        for horizon in horizons_months:
            target_date = (pd.Timestamp(last_date) + pd.DateOffset(months=int(horizon))).to_period("M").to_timestamp()
            month = int(target_date.month)
            baseline = float(baseline_map.get(month, fallback_baseline))
            month_iqr = float(iqr_map.get(month, fallback_iqr))
            month_iqr = max(month_iqr, 0.1)

            pred_anom = last_anom + (trend * horizon)
            pred_ndvi = baseline + pred_anom * (month_iqr + 0.05)

            band = residual_scale * np.sqrt(horizon / 3.0)
            pred_low = pred_ndvi - (1.96 * band)
            pred_high = pred_ndvi + (1.96 * band)

            all_rows.append(
                {
                    "segment_id": segment_id,
                    "date": target_date,
                    "as_of_date": last_date,
                    "target_date": target_date,
                    "horizon_months": int(horizon),
                    "is_forecast": True,
                    "ndvi_median": pred_ndvi,
                    "ndvi_p90": np.nan,
                    "frac_ndvi_gt_0_6": density,
                    "ndvi_iqr": month_iqr,
                    "ndvi_anomaly": pred_anom,
                    "ndvi_anomaly_slope_3m": trend,
                    "obs_count": 0,
                    "expected_pixels": int(latest.get("expected_pixels", latest.get("obs_count", 1)) or 1),
                    "valid_pixel_frac": 0.0,
                    "cloud_free_ratio": 0.0,
                    "forecast_lower": pred_low,
                    "forecast_upper": pred_high,
                }
            )

    forecast_df = pd.DataFrame(all_rows)
    if forecast_df.empty:
        return forecast_df

    forecast_df = compute_risk_scores(forecast_df)
    return forecast_df
