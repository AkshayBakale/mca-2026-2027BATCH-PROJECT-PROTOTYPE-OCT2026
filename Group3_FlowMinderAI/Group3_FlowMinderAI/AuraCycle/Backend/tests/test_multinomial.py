"""
test_multinomial.py - Unit Tests for Dirichlet-Multinomial Phase Engine
"""

import sys
import os
import pytest
from importlib import import_module

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [root_dir, os.path.join(root_dir, "01_data_pipeline")]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    multi_mod = import_module("01_data_pipeline.multinomial_engine")
    MultinomialPhaseEngine = multi_mod.MultinomialPhaseEngine
except Exception:
    import multinomial_engine as multi_mod
    MultinomialPhaseEngine = multi_mod.MultinomialPhaseEngine

def test_multinomial_phase_probabilities_sum_to_one():
    """Verify continuous probability distributions across phases sum to 1.0."""
    engine = MultinomialPhaseEngine()
    analysis = engine.get_phase_analysis(
        current_cycle_day=14,
        cycle_length_estimate=28.0
    )
    
    probs = analysis["distribution"]
    assert pytest.approx(sum(probs.values()), abs=1e-3) == 1.0
    assert "Ovulatory" in probs
    assert probs["Ovulatory"] > probs["Menstrual"]  # Day 14 should have high ovulatory/follicular

def test_early_cycle_menstrual_phase():
    """Day 2 of cycle should have dominant Menstrual phase probability."""
    engine = MultinomialPhaseEngine()
    analysis = engine.get_phase_analysis(
        current_cycle_day=2,
        cycle_length_estimate=28.0
    )
    
    probs = analysis["distribution"]
    assert probs["Menstrual"] > probs["Luteal"]
    assert analysis["dominant_phase"].lower() == "menstrual"

def test_multinomial_symptom_bayesian_update():
    """Pelvic cramps reported during cycle should evaluate properly."""
    engine = MultinomialPhaseEngine()
    symptoms = {"cramps_pelvic": 3, "fatigue_lethargy": 2}
    
    analysis = engine.get_phase_analysis(
        current_cycle_day=27,
        cycle_length_estimate=28.0,
        symptoms=symptoms
    )
    
    assert "dominant_phase" in analysis
    assert analysis["entropy_bits"] >= 0.0
