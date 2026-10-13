"""
__init__.py - Flask Application Factory for AuraCycle AI
Registers CORS, Blueprints, rate limiting, and global error handlers.
"""

import os
import sys
from flask import Flask, jsonify
from flask_cors import CORS

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_root = os.path.dirname(current_dir)
project_root = os.path.dirname(backend_root)

for p in [project_root, backend_root, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Determine template and static folders
template_candidate = os.path.join(project_root, "04_frontend_dashboard", "templates")
if not os.path.exists(template_candidate):
    template_candidate = os.path.join(project_root, "auracycle_core", "app", "templates")
if not os.path.exists(template_candidate):
    template_candidate = os.path.join(current_dir, "templates")

static_candidate = os.path.join(project_root, "04_frontend_dashboard", "static")
if not os.path.exists(static_candidate):
    static_candidate = os.path.join(project_root, "auracycle_core", "app", "static")
if not os.path.exists(static_candidate):
    static_candidate = os.path.join(current_dir, "static")

from app.config import config_by_name
from app.routes.api_predict import api_predict_bp
from app.routes.api_journal import api_journal_bp
from app.routes.api_llm import api_llm_bp
from app.routes.api_recommend import recommend_bp
from app.routes.web_views import web_views_bp

def create_app(config_name: str = "default") -> Flask:
    """Application factory for AuraCycle AI."""
    app = Flask(
        __name__,
        template_folder=template_candidate,
        static_folder=static_candidate
    )
    
    # Load configuration
    cfg = config_by_name.get(config_name, config_by_name["default"])
    app.config.from_object(cfg)
    
    # Enable CORS
    CORS(app, resources={r"/*": {"origins": app.config.get("CORS_ORIGINS", "*")}})
    
    # Register Blueprints
    app.register_blueprint(api_predict_bp)
    app.register_blueprint(api_journal_bp)
    app.register_blueprint(api_llm_bp)
    app.register_blueprint(recommend_bp)
    app.register_blueprint(web_views_bp)
    
    # Global Error Handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({
            "success": False,
            "error": "Endpoint not found"
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            "success": False,
            "error": "Internal server error in AuraCycle AI engine"
        }), 500
        
    @app.errorhandler(400)
    def bad_request_error(error):
        return jsonify({
            "success": False,
            "error": "Bad request"
        }), 400

    return app
