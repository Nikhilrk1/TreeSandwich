"""Feature engineering for Vegetation Vision segment-month records."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


REQUIRED_PIXEL_COLUMNS = {"segment_id", "date", "ndvi", "is_valid"}


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _safe_percentile(values: np.ndarray, pct: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.nanpercentile(values, pct))


def _safe_iqr(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25))


def _rolling_slope(values: np.ndarray) -> float:
    if values.size < 3 or np.isnan(values).any():
        return float("nan")
    x = np.arange(values.size, dtype=float)
    slope, _ = np.polyfit(x, values, 1)
    return float(slope)


def build_segment_feature_table(
    pixel_df: pd.DataFrame,
    ndvi_threshold: float = 0.60,
) -> pd.DataFrame:
    """Aggregate pixel-level NDVI into segment-month features.

    Expected input grain: one row per pixel per segment-month.
    """

    _require_columns(pixel_df, REQUIRED_PIXEL_COLUMNS)
    df = pixel_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    df["is_valid"] = df["is_valid"].astype(bool)
    if "is_cloud_free" in df.columns:
        df["is_cloud_free"] = df["is_cloud_free"].astype(bool)

    rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    for (segment_id, date), group in df.groupby(["segment_id", "date"], sort=True):
        total_pixels = int(len(group))
        valid_vals = group.loc[group["is_valid"], "ndvi"].dropna().to_numpy(dtype=float)
        obs_count = int(valid_vals.size)
        valid_pixel_frac = obs_count / total_pixels if total_pixels else 0.0

        if "is_cloud_free" in group.columns:
            cloud_free_ratio = float(group["is_cloud_free"].mean())
        else:
            # Fallback when cloud-specific QA isn't provided.
            cloud_free_ratio = valid_pixel_frac

        if "expected_pixels" in group.columns:
            expected_pixels = int(np.nanmax(group["expected_pixels"].to_numpy(dtype=float)))
            if expected_pixels <= 0:
                expected_pixels = total_pixels
        else:
            expected_pixels = total_pixels

        row = {
            "segment_id": segment_id,
            "date": date,
            "obs_count": obs_count,
            "expected_pixels": expected_pixels,
            "valid_pixel_frac": float(valid_pixel_frac),
            "cloud_free_ratio": float(cloud_free_ratio),
            "ndvi_median": float(np.nanmedian(valid_vals)) if obs_count else float("nan"),
            "ndvi_p90": _safe_percentile(valid_vals, 90),
            "frac_ndvi_gt_0_6": float(np.mean(valid_vals > ndvi_threshold)) if obs_count else float("nan"),
            "ndvi_iqr": _safe_iqr(valid_vals),
        }
        if "vegetation_distance_m" in group.columns:
            dist_vals = group["vegetation_distance_m"].dropna().to_numpy(dtype=float)
            row["vegetation_distance_m"] = float(np.nanmin(dist_vals)) if dist_vals.size else float("nan")
        rows.append(row)

    features = pd.DataFrame(rows).sort_values(["segment_id", "date"]).reset_index(drop=True)
    return features


def add_temporal_features(feature_df: pd.DataFrame, anomaly_eps: float = 0.05) -> pd.DataFrame:
    """Add rolling, baseline, and anomaly features to segment-month table."""

    required = {"segment_id", "date", "ndvi_median"}
    _require_columns(feature_df, required)

    df = feature_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    df = df.sort_values(["segment_id", "date"]).reset_index(drop=True)
    df["month"] = df["date"].dt.month

    monthly_baseline = (
        df.groupby(["segment_id", "month"])["ndvi_median"]
        .agg(
            segment_month_baseline_median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )
    monthly_baseline["segment_month_baseline_iqr"] = (
        monthly_baseline["q75"] - monthly_baseline["q25"]
    )
    monthly_baseline = monthly_baseline.drop(columns=["q25", "q75"])
    df = df.merge(monthly_baseline, on=["segment_id", "month"], how="left")

    denom = df["segment_month_baseline_iqr"].fillna(0.0) + anomaly_eps
    df["ndvi_anomaly"] = (df["ndvi_median"] - df["segment_month_baseline_median"]) / denom

    seg_std = df.groupby("segment_id")["ndvi_median"].transform("std").replace(0.0, np.nan)
    df["ndvi_anomaly_z"] = (
        (df["ndvi_median"] - df["segment_month_baseline_median"]) / seg_std
    )

    grouped = df.groupby("segment_id", sort=False)["ndvi_median"]
    df["ndvi_rolling_mean_3m"] = grouped.transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )
    df["ndvi_slope_3m"] = grouped.transform(
        lambda s: s.rolling(window=3, min_periods=3).apply(
            lambda arr: _rolling_slope(np.asarray(arr, dtype=float)),
            raw=True,
        )
    )

    if "vegetation_distance_m" in df.columns:
        dist_group = df.groupby("segment_id", sort=False)["vegetation_distance_m"]
        # Positive values represent vegetation moving toward the line.
        df["growth_amount_m"] = dist_group.transform(lambda s: s.diff() * -1.0).clip(lower=0.0)
        df["growth_rate_m_per_month"] = (
            df.groupby("segment_id", sort=False)["growth_amount_m"]
            .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
            .fillna(0.0)
        )
        df["distance_change_m_3m"] = dist_group.transform(lambda s: s.diff(periods=3) * -1.0)
    else:
        df["growth_amount_m"] = np.nan
        df["growth_rate_m_per_month"] = np.nan
        df["distance_change_m_3m"] = np.nan

    return df.drop(columns=["month"])
