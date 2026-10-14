"""
test_nlp_parser.py - Unit Tests for NLP Symptom Entity & Sentiment Extraction
"""

import sys
import os
import pytest

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
backend_dir = os.path.join(root_dir, "03_backend_api")
for p in [root_dir, backend_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.services.nlp_parser import analyze_journal

def test_cramp_and_spotting_extraction():
    """Verify dysmenorrhea and spotting are detected with elevated severity."""
    text = "Severe pelvic cramps started this morning with light spotting and sharp pain."
    res = analyze_journal(text)
    
    assert len(res["entities"]) >= 2
    entity_names = [e["entity"] for e in res["entities"]]
    assert "dysmenorrhea" in entity_names
    assert "spotting" in entity_names
    
    cramp_entity = next(e for e in res["entities"] if e["entity"] == "dysmenorrhea")
    assert cramp_entity["severity"] >= 4
    assert res["feature_modifiers"]["cramp_severity"] >= 4

def test_sentiment_affect():
    """Negative sentiment in journal text should reflect in negative compound score."""
    distress_text = "I feel awful, terrible throbbing migraine, extreme exhaustion and misery."
    res = analyze_journal(distress_text)
    
    assert res["sentiment"]["compound"] < -0.2
    assert res["feature_modifiers"]["stress_boost"] >= 1.0

def test_clean_neutral_journal():
    """Text without gynecological symptoms should return 0 entities and benign modifiers."""
    neutral_text = "Had a great meeting with the team today, weather is sunny."
    res = analyze_journal(neutral_text)
    
    assert len(res["entities"]) == 0
    assert res["feature_modifiers"]["cramp_severity"] == 0
    assert res["feature_modifiers"]["onset_shift_days"] == 0.0
