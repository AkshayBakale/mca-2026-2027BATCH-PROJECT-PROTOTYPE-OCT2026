"""
test_inference.py - Unit & Monotonicity Tests for Quantile LightGBM
"""

import sys
import os
import pytest

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
backend_dir = os.path.join(root_dir, "03_backend_api")
for p in [root_dir, backend_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.services.ml_engine import ml_engine

def test_models_loaded():
    """Verify LightGBM quantile models and metadata are initialized."""
    assert ml_engine.q10_model is not None
    assert ml_engine.q50_model is not None
    assert ml_engine.q90_model is not None
    assert len(ml_engine.feature_columns) > 0

def test_quantile_monotonicity():
    """Verify mathematical guarantee: q10 <= q50 <= q90 across synthetic telemetry."""
    sample_payload = {
        "age": 28,
        "height_cm": 165.0,
        "weight_kg": 58.0,
        "last_period_start": "2026-03-01",
        "previous_cycle_length": 29.0,
        "cycle_length_2": 28.0,
        "cycle_length_3": 30.0,
        "cycle_length_4": 29.0,
        "cycle_length_5": 28.0,
        "average_cycle_length_5": 28.8,
        "cycle_std_dev": 1.2,
        "usual_period_duration": 5,
        "flow_intensity": "Medium",
        "stress_level": 5,
        "sleep_hours": 7.5,
        "pcos_diagnosis": 0
    }
    
    result = ml_engine.predict(sample_payload)
    pred = result["prediction"]
    
    q10 = pred["q10_cycle_length"]
    q50 = pred["q50_cycle_length"]
    q90 = pred["q90_cycle_length"]
    
    assert q10 <= q50, f"Monotonicity violation: q10 ({q10}) > q50 ({q50})"
    assert q50 <= q90, f"Monotonicity violation: q50 ({q50}) > q90 ({q90})"
    assert pred["uncertainty_window_days"] == pytest.approx(q90 - q10, abs=0.1)

def test_pcos_cycle_lengthening():
    """PCOS diagnosis should shift predictions higher or widen uncertainty."""
    base_payload = {
        "age": 25,
        "height_cm": 160.0,
        "weight_kg": 55.0,
        "last_period_start": "2026-03-01",
        "previous_cycle_length": 30.0,
        "average_cycle_length_5": 30.0,
        "cycle_std_dev": 2.0,
        "stress_level": 4,
        "sleep_hours": 7.5,
        "pcos_diagnosis": 0
    }
    
    pcos_payload = dict(base_payload)
    pcos_payload["pcos_diagnosis"] = 1
    pcos_payload["cycle_std_dev"] = 4.5
    pcos_payload["previous_cycle_length"] = 36.0
    
    res_base = ml_engine.predict(base_payload)
    res_pcos = ml_engine.predict(pcos_payload)
    
    assert res_pcos["prediction"]["q50_cycle_length"] >= res_base["prediction"]["q50_cycle_length"]
    assert res_pcos["prediction"]["q10_cycle_length"] <= res_pcos["prediction"]["q50_cycle_length"] <= res_pcos["prediction"]["q90_cycle_length"]
