"""
test_bayesian.py - Unit Tests for Bayesian Latent Cycle De-Aliasing
"""

import sys
import os
import pytest

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
backend_dir = os.path.join(root_dir, "03_backend_api")
for p in [root_dir, backend_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.services.bayesian_cleaner import (
    dealias_cycle_gap,
    clean_cycle_time_series,
    compute_gap_posterior,
    gaussian_pdf
)

def test_normal_cycle_gap_no_split():
    """A standard 29-day interval should resolve to K=1 with no split."""
    res = dealias_cycle_gap(gap_days=29.0, mean_cycle=28.5, std_cycle=2.0)
    assert res["optimal_k"] == 1
    assert not res["split_triggered"]
    assert not res["is_artifact"]
    assert len(res["sub_cycle_lengths"]) == 1
    assert res["sub_cycle_lengths"][0] == 29.0

def test_double_cycle_tracking_gap_split():
    """A ~58-day tracking gap should resolve to K=2 (two latent cycles)."""
    res = dealias_cycle_gap(gap_days=58.0, mean_cycle=28.5, std_cycle=2.0)
    assert res["optimal_k"] == 2
    assert res["split_triggered"]
    assert len(res["sub_cycle_lengths"]) == 2
    assert pytest.approx(sum(res["sub_cycle_lengths"]), abs=0.1) == 58.0
    assert 28.0 <= res["sub_cycle_lengths"][0] <= 30.0

def test_triple_cycle_tracking_gap():
    """An 86-day gap should split into K=3 cycles."""
    res = dealias_cycle_gap(gap_days=86.0, mean_cycle=28.5, std_cycle=2.0)
    assert res["optimal_k"] == 3
    assert res["split_triggered"]
    assert len(res["sub_cycle_lengths"]) == 3

def test_tracking_artifact_detection():
    """A gap < 15 days is mid-cycle spotting or erroneous entry."""
    res = dealias_cycle_gap(gap_days=10.0, mean_cycle=28.5, std_cycle=2.0)
    assert res["is_artifact"]
    assert not res["split_triggered"]

def test_clean_cycle_time_series():
    """Verify cleaning an array of mixed raw cycle lengths."""
    raw_history = [28.0, 29.0, 57.0, 28.5, 30.0]
    cleaned, summary = clean_cycle_time_series(raw_history, mean_cycle=28.5, std_cycle=2.0)
    
    assert summary["num_original"] == 5
    assert summary["num_cleaned"] == 6  # 57 split into two ~28.5 cycles
    assert summary["splits_performed"] == 1
    assert max(cleaned) < 35.0  # Outlier 57.0 decomposed
