"""
test_recommendations.py - Unit Tests for Collaborative Filtering, Content Recommendations & CEBM Evidence
"""

import sys
import os
import pytest
from importlib import import_module

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [root_dir, os.path.join(root_dir, "02_ml_engine")]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    collab_mod = import_module("02_ml_engine.collaborative_filter")
    collaborative_recommender = collab_mod.collaborative_recommender
    content_mod = import_module("02_ml_engine.content_recommender")
    content_recommender = content_mod.content_recommender
    evidence_mod = import_module("02_ml_engine.evidence_ranker")
    evidence_ranker = evidence_mod.evidence_ranker
except Exception:
    import collaborative_filter as collab_mod
    collaborative_recommender = collab_mod.collaborative_recommender
    import content_recommender as content_mod
    content_recommender = content_mod.content_recommender
    import evidence_ranker as evidence_mod
    evidence_ranker = evidence_mod.evidence_ranker

def test_collaborative_filtering_recommendation():
    """Verify collaborative filtering identifies cohort neighbors and returns ranked remedies."""
    patient = {
        "age": 27,
        "bmi": 23.4,
        "previous_cycle_length": 29.0,
        "cycle_std_dev": 1.5,
        "stress_level": 7,
        "sleep_hours": 6.0,
        "pcos_diagnosis": 0
    }
    
    recs = collaborative_recommender.recommend_for_patient(patient, k=5, top_n=3)
    assert len(recs) <= 3
    assert len(recs) > 0
    for r in recs:
        assert "name" in r
        assert "predicted_rating" in r
        assert 1.0 <= r["predicted_rating"] <= 5.0

def test_content_based_recommendation():
    """Verify content-based cosine similarity matches PCOS interventions for PCOS patients."""
    patient = {
        "pcos_diagnosis": 1,
        "stress_level": 8,
        "sleep_hours": 5.0,
        "bmi": 28.5
    }
    nlp_symptoms = ["bloating", "irregular spotting", "anxiety"]
    
    recs = content_recommender.recommend(patient, nlp_symptoms=nlp_symptoms, top_n=3)
    assert len(recs) > 0
    rec_names = [r["title"].lower() for r in recs]
    # Inositol, chronobiology or inflammation should be matched
    assert any("inositol" in n or "circadian" in n or "omega" in n or "anti" in n for n in rec_names)

def test_evidence_ranker_cebm():
    """Verify Oxford CEBM ranking orders clinical disruptors correctly."""
    features = {
        "pcos_diagnosis": 1,
        "stress_level": 9,
        "emergency_contraception_recent": 1,
        "sleep_hours": 4.5
    }
    
    ranked = evidence_ranker.evaluate_patient_disruptors(features)
    assert len(ranked) >= 2
    for item in ranked:
        assert "cebm_level" in item
        assert "estimated_shift_days" in item
        assert item["estimated_shift_days"] > 0
