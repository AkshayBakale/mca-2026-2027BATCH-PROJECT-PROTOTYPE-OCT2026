"""
api_llm.py - Clinical Report & Medical Explanation Endpoints
"""

from flask import Blueprint, request, jsonify, Response
from app.services.llm_explainer import llm_explainer
from app.services.ml_engine import ml_engine
from app.services.nlp_parser import analyze_journal

api_llm_bp = Blueprint("api_llm", __name__, url_prefix="/api/llm")

@api_llm_bp.route("/explain", methods=["POST"])
def explain_prediction():
    """
    Generates empathetic patient narrative and physician SOAP report.
    Accepts: { telemetry: {...}, journal_text: "..." }
    """
    data = request.get_json(silent=True) or {}
    telemetry = data.get("telemetry", data)
    journal_text = data.get("journal_text", "")
    
    journal_analysis = None
    if journal_text:
        journal_analysis = analyze_journal(journal_text)
        
    # Get predictions
    pred_result = ml_engine.predict(telemetry, nlp_modifiers=journal_analysis.get("feature_modifiers") if journal_analysis else None)
    
    synthesis = llm_explainer.generate_clinical_synthesis(
        prediction_result=pred_result,
        raw_telemetry=telemetry,
        journal_analysis=journal_analysis
    )
    
    return jsonify({
        "success": True,
        "data": {
            "synthesis": synthesis,
            "prediction": pred_result["prediction"],
            "phase_tracking": pred_result["phase_tracking"]
        }
    }), 200

@api_llm_bp.route("/export-pdf", methods=["POST"])
def export_soap_pdf():
    """
    Generates binary PDF file for download.
    """
    data = request.get_json(silent=True) or {}
    telemetry = data.get("telemetry", data)
    journal_text = data.get("journal_text", "")
    
    journal_analysis = None
    if journal_text:
        journal_analysis = analyze_journal(journal_text)
        
    pred_result = ml_engine.predict(telemetry, nlp_modifiers=journal_analysis.get("feature_modifiers") if journal_analysis else None)
    
    synthesis = llm_explainer.generate_clinical_synthesis(
        prediction_result=pred_result,
        raw_telemetry=telemetry,
        journal_analysis=journal_analysis
    )
    
    pdf_bytes = llm_explainer.generate_soap_pdf(
        prediction_result=pred_result,
        raw_telemetry=telemetry,
        clinical_synthesis=synthesis
    )
    
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={"Content-Disposition": "attachment; filename=AuraCycle_Clinical_Brief.pdf"}
    )
