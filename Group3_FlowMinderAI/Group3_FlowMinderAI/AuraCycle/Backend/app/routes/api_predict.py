"""
api_predict.py - Quantile Prediction & Bayesian Inference Endpoints
Handles cycle length prediction, Bayesian de-aliasing, and health status.
"""

from flask import Blueprint, request, jsonify
from app.services.ml_engine import ml_engine
from app.services.nlp_parser import analyze_journal
from app.services.bayesian_cleaner import dealias_cycle_gap, clean_cycle_time_series

api_predict_bp = Blueprint("api_predict", __name__, url_prefix="/api")

@api_predict_bp.route("/health", methods=["GET"])
def health_check():
    """System health & model readiness status."""
    is_trained = (ml_engine.q10_model is not None and 
                  ml_engine.q50_model is not None and 
                  ml_engine.q90_model is not None)
    return jsonify({
        "status": "healthy",
        "system": "AuraCycle AI Engine",
        "models_loaded": is_trained,
        "feature_count": len(ml_engine.feature_columns),
        "supported_quantiles": [0.10, 0.50, 0.90]
    }), 200

@api_predict_bp.route("/predict", methods=["POST"])
def predict_cycle():
    """
    Inference endpoint:
    Accepts patient telemetry and optional journal text.
    Returns calibrated quantiles, dates, phase tracking, and PDF curve.
    """
    data = request.get_json(silent=True) or {}
    
    # Optional journal integration
    nlp_modifiers = None
    journal_text = data.get("journal_text", "")
    if journal_text and isinstance(journal_text, str) and journal_text.strip():
        journal_res = analyze_journal(journal_text)
        nlp_modifiers = journal_res.get("feature_modifiers")
        
    try:
        prediction_payload = ml_engine.predict(data, nlp_modifiers=nlp_modifiers)
        return jsonify({
            "success": True,
            "data": prediction_payload
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

@api_predict_bp.route("/dealias", methods=["POST"])
def dealias_gap():
    """
    Bayesian De-Aliasing Endpoint:
    Accepts gap_days, mean_cycle, std_cycle, and optional last_start_date.
    Returns P(K|G) and de-aliased sub-cycles.
    """
    data = request.get_json(silent=True) or {}
    gap_days = float(data.get("gap_days", 56.0))
    mean_cycle = float(data.get("mean_cycle", 28.5))
    std_cycle = float(data.get("std_cycle", 2.5))
    last_date = data.get("last_start_date")
    
    result = dealias_cycle_gap(
        gap_days=gap_days,
        mean_cycle=mean_cycle,
        std_cycle=std_cycle,
        last_start_date=last_date
    )
    return jsonify({
        "success": True,
        "data": result
    }), 200

@api_predict_bp.route("/clean-series", methods=["POST"])
def clean_series():
    """
    Cleans a sequence of past cycle lengths.
    """
    data = request.get_json(silent=True) or {}
    lengths = data.get("cycle_lengths", [])
    if not isinstance(lengths, list):
        return jsonify({"success": False, "error": "cycle_lengths must be a list of numbers"}), 400
        
    cleaned_lengths, summary = clean_cycle_time_series(lengths)
    return jsonify({
        "success": True,
        "cleaned_cycle_lengths": cleaned_lengths,
        "summary": summary
    }), 200
