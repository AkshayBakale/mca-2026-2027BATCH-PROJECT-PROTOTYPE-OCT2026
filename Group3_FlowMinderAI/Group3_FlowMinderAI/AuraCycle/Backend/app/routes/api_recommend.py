"""
api_recommend.py - Recommendation, Evidence Ranking & Multinomial Endpoints
Exposes REST endpoints for:
- Collaborative Filtering (Cohort Cycle Twins)
- Content-Based Recommendations (Cosine Similarity)
- Oxford CEBM Clinical Evidence Ranking
- Multinomial Cycle Phase & Symptom Distribution
- BeautifulSoup Clinical Literature Scraping
"""

import sys
import os
from flask import Blueprint, request, jsonify

# Add candidate root and engine directories to sys.path for cross-folder modular imports
current_dir = os.path.dirname(os.path.abspath(__file__))
root_candidates = [
    os.path.abspath(os.path.join(current_dir, "..", "..", "..")),
    os.path.abspath(os.path.join(current_dir, "..", "..")),
    os.path.abspath(os.path.join(current_dir, ".."))
]
for r in root_candidates:
    for sub in [r, os.path.join(r, "02_ml_engine"), os.path.join(r, "01_data_pipeline"), os.path.join(r, "Folder_2_ML_Engine"), os.path.join(r, "Folder_1_Data_Pipeline")]:
        if os.path.exists(sub) and sub not in sys.path:
            sys.path.insert(0, sub)

try:
    from importlib import import_module
    collab_mod = import_module("02_ml_engine.collaborative_filter")
    collaborative_recommender = collab_mod.collaborative_recommender

    content_mod = import_module("02_ml_engine.content_recommender")
    content_recommender = content_mod.content_recommender

    evidence_mod = import_module("02_ml_engine.evidence_ranker")
    evidence_ranker = evidence_mod.evidence_ranker

    multi_mod = import_module("01_data_pipeline.multinomial_engine")
    MultinomialPhaseEngine = multi_mod.MultinomialPhaseEngine

    scraper_mod = import_module("01_data_pipeline.literature_scraper")
    clinical_scraper = scraper_mod.clinical_scraper
except Exception:
    try:
        from ml_engine.collaborative_filter import collaborative_recommender
        from ml_engine.content_recommender import content_recommender
        from ml_engine.evidence_ranker import evidence_ranker
        from data_pipeline.multinomial_engine import MultinomialPhaseEngine
        from data_pipeline.literature_scraper import clinical_scraper
    except Exception:
        import collaborative_filter
        collaborative_recommender = collaborative_filter.collaborative_recommender
        import content_recommender
        content_recommender = content_recommender.content_recommender
        import evidence_ranker
        evidence_ranker = evidence_ranker.evidence_ranker
        from multinomial_engine import MultinomialPhaseEngine
        from literature_scraper import clinical_scraper

recommend_bp = Blueprint("recommend", __name__, url_prefix="/api")
multinomial_engine = MultinomialPhaseEngine()

@recommend_bp.route("/recommend/collaborative", methods=["POST"])
def recommend_collaborative():
    """
    POST /api/recommend/collaborative
    Runs memory-based User-User collaborative filtering against the 60k/user cohort.
    """
    data = request.get_json(silent=True) or {}
    k = int(request.args.get("k", 5))
    top_n = int(request.args.get("top_n", 4))
    
    recommendations = collaborative_recommender.recommend_for_patient(data, k=k, top_n=top_n)
    return jsonify({
        "status": "success",
        "method": "user_user_collaborative_filtering",
        "cohort_neighbors_evaluated": k,
        "recommendations": recommendations
    }), 200

@recommend_bp.route("/recommend/content", methods=["POST"])
def recommend_content():
    """
    POST /api/recommend/content
    Calculates cosine similarity between patient biological profile + NLP symptoms
    and evidence-backed intervention catalog.
    """
    payload = request.get_json(silent=True) or {}
    patient_data = payload.get("patient_data", payload)
    nlp_symptoms = payload.get("symptoms", [])
    top_n = int(request.args.get("top_n", 4))

    recommendations = content_recommender.recommend(patient_data, nlp_symptoms=nlp_symptoms, top_n=top_n)
    return jsonify({
        "status": "success",
        "method": "content_based_cosine_similarity",
        "clinical_vocabulary_size": len(content_recommender.vocab),
        "recommendations": recommendations
    }), 200

@recommend_bp.route("/evidence/rank", methods=["POST"])
def rank_evidence():
    """
    POST /api/evidence/rank
    Ranks patient-specific menstrual disruption drivers according to Oxford CEBM hierarchy.
    """
    patient_features = request.get_json(silent=True) or {}
    ranked_disruptors = evidence_ranker.evaluate_patient_disruptors(patient_features)
    
    total_impact_days = sum(item["estimated_shift_days"] for item in ranked_disruptors)
    
    return jsonify({
        "status": "success",
        "framework": "Oxford Centre for Evidence-Based Medicine (CEBM)",
        "total_disruptors_detected": len(ranked_disruptors),
        "cumulative_variance_impact_days": round(total_impact_days, 1),
        "ranked_evidence": ranked_disruptors
    }), 200

@recommend_bp.route("/phase/multinomial", methods=["POST"])
def phase_multinomial():
    """
    POST /api/phase/multinomial
    Calculates multinomial phase probability vector and symptom emission likelihoods.
    """
    payload = request.get_json(silent=True) or {}
    
    day = int(payload.get("current_cycle_day", 14))
    cycle_length = float(payload.get("cycle_length_estimate", 28.5))
    symptoms = payload.get("symptoms", None)
    has_pcos = bool(payload.get("pcos_diagnosis", False))
    stress = int(payload.get("stress_level", 4))

    analysis = multinomial_engine.get_phase_analysis(
        current_cycle_day=day,
        cycle_length_estimate=cycle_length,
        symptoms=symptoms,
        pcos=has_pcos,
        stress_level=stress
    )

    return jsonify({
        "status": "success",
        "distribution_type": "Dirichlet-Multinomial",
        "analysis": analysis
    }), 200

@recommend_bp.route("/literature/scrape", methods=["GET", "POST"])
def scrape_literature():
    """
    GET/POST /api/literature/scrape?query=pcos
    Uses BeautifulSoup to search or harvest clinical guidelines from medical corpus or URL.
    """
    query = request.args.get("query", "")
    target_url = request.args.get("url", "")
    
    if target_url:
        result = clinical_scraper.scrape_url(target_url)
        return jsonify(result), 200

    results = clinical_scraper.query_evidence_by_symptom_or_condition(query) if query else clinical_scraper.evidence_database
    
    return jsonify({
        "status": "success",
        "parser": "BeautifulSoup4 Clinical Guideline Harvester",
        "total_guidelines_indexed": len(results),
        "query": query,
        "evidence": results
    }), 200
