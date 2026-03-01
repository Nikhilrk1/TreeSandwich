"""Forecast vegetation growth amount and projected distance-to-line."""

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


def _growth_rate(seg_df: pd.DataFrame) -> float:
    if "growth_rate_m_per_month" in seg_df.columns:
        recent = pd.to_numeric(seg_df["growth_rate_m_per_month"], errors="coerce").dropna().tail(3)
        if not recent.empty:
            return float(recent.mean())
    if "growth_amount_m" in seg_df.columns:
        return max(0.0, _trend_from_tail(pd.to_numeric(seg_df["growth_amount_m"], errors="coerce")))
    return 0.0


def build_forecast_rows(
    scored_df: pd.DataFrame,
    horizons_months: tuple[int, int] = (3, 6),
) -> pd.DataFrame:
    """Generate per-segment forecast rows with growth amount outputs."""

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

        current_distance = pd.to_numeric(latest.get("vegetation_distance_m", np.nan), errors="coerce")
        current_distance = float(current_distance) if pd.notna(current_distance) else np.nan
        growth_rate = max(0.0, _growth_rate(seg_df))

        baseline_map, iqr_map = _month_baseline_maps(seg_df)
        fallback_baseline = float(seg_df["ndvi_median"].median())
        fallback_iqr = float(seg_df["ndvi_median"].quantile(0.75) - seg_df["ndvi_median"].quantile(0.25))
        fallback_iqr = max(fallback_iqr, 0.1)

        density = float(seg_df["frac_ndvi_gt_0_6"].dropna().tail(3).mean()) if "frac_ndvi_gt_0_6" in seg_df.columns else 0.0
        density = float(np.clip(np.nan_to_num(density, nan=0.0), 0.0, 1.0))

        growth_residual = (
            float(pd.to_numeric(seg_df.get("growth_amount_m"), errors="coerce").diff().dropna().abs().median())
            if "growth_amount_m" in seg_df.columns
            else np.nan
        )
        growth_residual = max(np.nan_to_num(growth_residual, nan=0.35), 0.15)

        for horizon in horizons_months:
            target_date = (pd.Timestamp(last_date) + pd.DateOffset(months=int(horizon))).to_period("M").to_timestamp()
            month = int(target_date.month)
            baseline = float(baseline_map.get(month, fallback_baseline))
            month_iqr = float(iqr_map.get(month, fallback_iqr))
            month_iqr = max(month_iqr, 0.1)

            pred_anom = last_anom + (trend * horizon)
            seasonal_growth_boost = max(0.0, pred_anom) * 0.08
            predicted_growth_amount_m = max(0.0, (growth_rate * horizon) + seasonal_growth_boost)

            band = growth_residual * np.sqrt(horizon / 3.0)
            growth_low = max(0.0, predicted_growth_amount_m - (1.96 * band))
            growth_high = max(0.0, predicted_growth_amount_m + (1.96 * band))

            if np.isnan(current_distance):
                predicted_distance_m = np.nan
            else:
                predicted_distance_m = max(0.0, current_distance - predicted_growth_amount_m)

            all_rows.append(
                {
                    "segment_id": segment_id,
                    "date": target_date,
                    "as_of_date": last_date,
                    "target_date": target_date,
                    "horizon_months": int(horizon),
                    "is_forecast": True,
                    "ndvi_median": np.nan,
                    "ndvi_p90": np.nan,
                    "frac_ndvi_gt_0_6": density,
                    "ndvi_iqr": month_iqr,
                    "ndvi_anomaly": pred_anom,
                    "ndvi_anomaly_slope_3m": trend,
                    "growth_amount_m": predicted_growth_amount_m,
                    "growth_rate_m_per_month": growth_rate,
                    "predicted_growth_amount_m": predicted_growth_amount_m,
                    "growth_lower_m": growth_low,
                    "growth_upper_m": growth_high,
                    "vegetation_distance_m": predicted_distance_m,
                    "predicted_distance_m": predicted_distance_m,
                    "obs_count": 0,
                    "expected_pixels": int(latest.get("expected_pixels", latest.get("obs_count", 1)) or 1),
                    "valid_pixel_frac": 0.0,
                    "cloud_free_ratio": 0.0,
                    "forecast_lower": growth_low,
                    "forecast_upper": growth_high,
                }
            )

    forecast_df = pd.DataFrame(all_rows)
    if forecast_df.empty:
        return forecast_df

    forecast_df = compute_risk_scores(forecast_df)
    return forecast_df
