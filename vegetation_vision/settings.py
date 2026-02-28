"""Project defaults for Vegetation Vision MVP."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MVPConfig:
    segment_length_m: int = 500
    corridor_buffer_m: int = 50
    cadence: str = "monthly"
    ndvi_threshold: float = 0.60
    anomaly_eps: float = 0.05
    risk_high_threshold: int = 70
    risk_medium_threshold: int = 50
    forecast_horizons_months: tuple[int, int] = (3, 6)


DEFAULT_CONFIG = MVPConfig()
