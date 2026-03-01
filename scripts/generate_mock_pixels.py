"""Generate synthetic pixel-level NDVI samples for local MVP testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_mock_pixels(
    segments: int = 50,
    months: int = 18,
    pixels_per_segment_month: int = 30,
) -> pd.DataFrame:
    start = pd.Timestamp("2024-01-01")
    dates = pd.date_range(start, periods=months, freq="MS")
    rng = np.random.default_rng(42)

    rows: list[dict[str, object]] = []
    for s in range(segments):
        segment_id = f"seg_{s:04d}"
        base = rng.uniform(0.25, 0.65)
        trend = rng.uniform(-0.005, 0.02)
        phase = rng.uniform(0.0, np.pi)
        initial_distance_m = rng.uniform(4.0, 18.0)
        encroachment_rate = rng.uniform(0.0, 0.45)

        for i, date in enumerate(dates):
            seasonal = 0.08 * np.sin((2 * np.pi * (date.month - 1) / 12.0) + phase)
            monthly_center = base + seasonal + trend * i
            seasonal_push = max(0.0, np.sin((2 * np.pi * (date.month - 1) / 12.0) + phase)) * 0.15
            distance_m = max(0.5, initial_distance_m - (encroachment_rate * i) - seasonal_push)

            for _ in range(pixels_per_segment_month):
                cloud_free = rng.random() > 0.18
                valid = cloud_free and (rng.random() > 0.04)
                ndvi = monthly_center + rng.normal(0.0, 0.03)
                ndvi = float(np.clip(ndvi, -0.1, 0.95))
                rows.append(
                    {
                        "segment_id": segment_id,
                        "date": date,
                        "ndvi": ndvi,
                        "is_valid": int(valid),
                        "is_cloud_free": int(cloud_free),
                        "expected_pixels": pixels_per_segment_month,
                        "vegetation_distance_m": distance_m,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output .parquet or .csv file")
    parser.add_argument("--segments", type=int, default=50)
    parser.add_argument("--months", type=int, default=18)
    parser.add_argument("--pixels", type=int, default=30)
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = build_mock_pixels(args.segments, args.months, args.pixels)
    if out.suffix.lower() == ".parquet":
        df.to_parquet(out, index=False)
    elif out.suffix.lower() == ".csv":
        df.to_csv(out, index=False)
    else:
        raise ValueError("Output must end with .parquet or .csv")

    print(f"Wrote mock pixel sample table: {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
