"""Baseline seasonality-aware vegetation growth pressure index."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _rolling_slope(values: np.ndarray) -> float:
    if values.size < 3 or np.isnan(values).any():
        return float("nan")
    x = np.arange(values.size, dtype=float)
    slope, _ = np.polyfit(x, values, 1)
    return float(slope)


def compute_risk_scores(
    feature_df: pd.DataFrame,
    horizons_months: tuple[int, int] = (3, 6),
    high_threshold: int = 70,
    medium_threshold: int = 50,
) -> pd.DataFrame:
    """Compute risk score and categorical level for each segment-month row."""

    _require_columns(feature_df, {"segment_id", "date", "ndvi_anomaly"})
    df = feature_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    df = df.sort_values(["segment_id", "date"]).reset_index(drop=True)

    if "ndvi_anomaly_slope_3m" not in df.columns:
        df["ndvi_anomaly_slope_3m"] = (
            df.groupby("segment_id", sort=False)["ndvi_anomaly"]
            .transform(
                lambda s: s.rolling(window=3, min_periods=3).apply(
                    lambda arr: _rolling_slope(np.asarray(arr, dtype=float)),
                    raw=True,
                )
            )
            .fillna(0.0)
        )

    anom = df["ndvi_anomaly"].fillna(0.0).clip(-2.0, 3.0)
    trend = df["ndvi_anomaly_slope_3m"].fillna(0.0).clip(-1.0, 1.0)
    density = df.get("frac_ndvi_gt_0_6", pd.Series(0.0, index=df.index)).fillna(0.0).clip(0.0, 1.0)

    pred_candidates = [anom + (h * trend) for h in horizons_months]
    pred_growth_pressure = np.maximum.reduce([np.zeros(len(df), dtype=float), *pred_candidates]).clip(0.0, 3.0)

    raw = (
        (0.40 * anom.to_numpy())
        + (0.30 * (trend.to_numpy() * 2.0))
        + (0.20 * density.to_numpy())
        + (0.10 * pred_growth_pressure)
    )

    score = np.round(100.0 * _sigmoid(raw)).astype(int)
    df["risk_score"] = np.clip(score, 0, 100)

    expected = df.get("expected_pixels", df.get("obs_count", pd.Series(1, index=df.index))).fillna(1)
    obs = df.get("obs_count", expected).fillna(0)
    expected = expected.replace(0, 1)
    df["risk_confidence"] = np.clip((obs / expected).to_numpy(dtype=float), 0.0, 1.0)

    df["risk_level"] = np.where(
        df["risk_score"] >= high_threshold,
        "high",
        np.where(df["risk_score"] >= medium_threshold, "medium", "low"),
    )

    return df
