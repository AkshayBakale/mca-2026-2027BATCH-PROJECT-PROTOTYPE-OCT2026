"""
api_journal.py - Free-Text Symptom Intake & NLP Parsing Endpoint
"""

from flask import Blueprint, request, jsonify
from app.services.nlp_parser import analyze_journal

api_journal_bp = Blueprint("api_journal", __name__, url_prefix="/api")

@api_journal_bp.route("/journal", methods=["POST"])
def parse_journal():
    """
    NLP Ingestion Endpoint:
    Accepts: { "text": "Cramps started this morning, feeling nauseous and bloated with spotting" }
    Returns: extracted entities, severity (1-5), VADER sentiment, feature modifiers.
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    
    if not text or not isinstance(text, str):
        return jsonify({
            "success": False,
            "error": "Field 'text' must be a non-empty string"
        }), 400
        
    analysis = analyze_journal(text)
    return jsonify({
        "success": True,
        "data": analysis
    }), 200
