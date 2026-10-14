"""
test_api.py - Comprehensive Flask REST API & Endpoint Tests
"""

import sys
import os
import json
import pytest

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
backend_dir = os.path.join(root_dir, "03_backend_api")
for p in [root_dir, backend_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app import create_app

@pytest.fixture
def client():
    app = create_app("testing")
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """GET /api/health should return 200 with model readiness status."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True

def test_dashboard_view(client):
    """GET / should render the HTML dashboard."""
    res = client.get("/")
    assert res.status_code == 200
    assert b"AuraCycle" in res.data

def test_predict_endpoint(client):
    """POST /api/predict with standard telemetry payload."""
    payload = {
        "age": 28,
        "height_cm": 162.0,
        "weight_kg": 57.5,
        "last_period_start": "2026-03-01",
        "previous_cycle_length": 29.0,
        "average_cycle_length_5": 28.8,
        "cycle_std_dev": 1.2,
        "stress_level": 5
    }
    res = client.post("/api/predict", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "prediction" in data["data"]
    assert "pdf_distribution" in data["data"]

def test_dealias_endpoint(client):
    """POST /api/dealias should compute posterior split for 58d gap."""
    payload = {
        "gap_days": 58.0,
        "mean_cycle": 28.5,
        "std_cycle": 2.0
    }
    res = client.post("/api/dealias", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["data"]["optimal_k"] == 2

def test_journal_nlp_endpoint(client):
    """POST /api/journal should extract symptoms and sentiment."""
    payload = {"text": "Bloating and painful lower back cramps started today"}
    res = client.post("/api/journal", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert len(data["data"]["entities"]) >= 1

def test_recommend_collaborative_endpoint(client):
    """POST /api/recommend/collaborative returns cohort remedies."""
    payload = {
        "age": 29,
        "stress_level": 7,
        "previous_cycle_length": 29
    }
    res = client.post("/api/recommend/collaborative", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "recommendations" in data

def test_recommend_content_endpoint(client):
    """POST /api/recommend/content returns targeted protocol matches."""
    payload = {
        "pcos_diagnosis": 1,
        "stress_level": 8
    }
    res = client.post("/api/recommend/content", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert len(data["recommendations"]) > 0

def test_evidence_ranking_endpoint(client):
    """POST /api/evidence/rank returns Oxford CEBM ranked factors."""
    payload = {
        "pcos_diagnosis": 1,
        "emergency_contraception_recent": 1,
        "stress_level": 9
    }
    res = client.post("/api/evidence/rank", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert data["total_disruptors_detected"] > 0

def test_multinomial_phase_endpoint(client):
    """POST /api/phase/multinomial returns continuous phase probabilities."""
    payload = {
        "current_cycle_day": 14,
        "cycle_length_estimate": 28.5
    }
    res = client.post("/api/phase/multinomial", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "analysis" in data

def test_literature_scrape_endpoint(client):
    """GET /api/literature/scrape returns clinical evidence from BeautifulSoup harvester."""
    res = client.get("/api/literature/scrape?query=pcos")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "evidence" in data
