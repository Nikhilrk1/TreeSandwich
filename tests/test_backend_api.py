from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient


def _write_fixture_tables(tmp_path: Path) -> tuple[Path, Path]:
    rows = [
        {
            "segment_id": "seg_0001",
            "as_of_date": "2025-06-01",
            "target_date": "2025-06-01",
            "horizon_months": 0,
            "is_forecast": False,
            "ndvi_median": 0.55,
            "ndvi_anomaly": 0.6,
            "growth_amount_m": 0.8,
            "growth_rate_m_per_month": 0.26,
            "vegetation_distance_m": 4.2,
            "risk_score": 74,
            "risk_level": "high",
            "hazard_level": "high",
            "risk_confidence": 0.9,
            "obs_count": 20,
            "expected_pixels": 24,
        },
        {
            "segment_id": "seg_0002",
            "as_of_date": "2025-06-01",
            "target_date": "2025-06-01",
            "horizon_months": 0,
            "is_forecast": False,
            "ndvi_median": 0.42,
            "ndvi_anomaly": 0.1,
            "growth_amount_m": 0.2,
            "growth_rate_m_per_month": 0.05,
            "vegetation_distance_m": 12.0,
            "risk_score": 48,
            "risk_level": "low",
            "hazard_level": "low",
            "risk_confidence": 0.85,
            "obs_count": 21,
            "expected_pixels": 24,
        },
        {
            "segment_id": "seg_0001",
            "as_of_date": "2025-06-01",
            "target_date": "2025-09-01",
            "horizon_months": 3,
            "is_forecast": True,
            "ndvi_median": None,
            "ndvi_anomaly": 0.75,
            "predicted_growth_amount_m": 1.1,
            "growth_lower_m": 0.8,
            "growth_upper_m": 1.4,
            "predicted_distance_m": 3.1,
            "risk_score": 79,
            "risk_level": "high",
            "hazard_level": "high",
            "risk_confidence": 0.9,
            "obs_count": 0,
            "expected_pixels": 24,
            "forecast_lower": 0.51,
            "forecast_upper": 0.65,
        },
    ]
    df = pd.DataFrame(rows)
    table_path = tmp_path / "segment_timeseries.parquet"
    df.to_parquet(table_path, index=False)

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"segment_id": "seg_0001", "corridor_id": "corr_001"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-81.2, 33.8], [-81.19, 33.81]],
                },
            },
            {
                "type": "Feature",
                "properties": {"segment_id": "seg_0002", "corridor_id": "corr_001"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-81.21, 33.79], [-81.20, 33.80]],
                },
            },
        ],
    }
    geom_path = tmp_path / "segments.geojson"
    with geom_path.open("w", encoding="utf-8") as f:
        json.dump(geojson, f)
    return table_path, geom_path


def _client_with_env(monkeypatch, data_path: Path, segment_path: Path) -> TestClient:
    monkeypatch.setenv("VV_DATA_PATH", str(data_path))
    monkeypatch.setenv("VV_SEGMENT_GEOJSON", str(segment_path))
    mod = importlib.import_module("backend.main")
    mod = importlib.reload(mod)
    return TestClient(mod.app)


def test_timeline_years(monkeypatch, tmp_path: Path):
    data_path, segment_path = _write_fixture_tables(tmp_path)
    client = _client_with_env(monkeypatch, data_path, segment_path)

    resp = client.get("/timeline/years")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["years"] == [2025]
    assert payload["default"] == 2025


def test_map_layer(monkeypatch, tmp_path: Path):
    data_path, segment_path = _write_fixture_tables(tmp_path)
    client = _client_with_env(monkeypatch, data_path, segment_path)

    resp = client.get("/map/layer", params={"year": 2025, "min_risk": 70})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["type"] == "FeatureCollection"
    assert len(payload["features"]) == 1
    props = payload["features"][0]["properties"]
    assert props["segment_id"] == "seg_0001"
    assert props["risk_score"] == 74.0
    assert props["hazard_level"] == "high"


def test_top_segments_includes_forecast_columns(monkeypatch, tmp_path: Path):
    data_path, segment_path = _write_fixture_tables(tmp_path)
    client = _client_with_env(monkeypatch, data_path, segment_path)

    resp = client.get("/segments/top", params={"year": 2025, "n": 5})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload
    top = payload[0]
    assert "forecast_growth_3m_m" in top
    assert "forecast_growth_6m_m" in top
    assert "hazard_level" in top
