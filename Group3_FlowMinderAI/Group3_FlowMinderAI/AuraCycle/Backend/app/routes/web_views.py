"""
web_views.py - Dashboard UI & Web Views Routes
"""

from flask import Blueprint, render_template

web_views_bp = Blueprint("web_views", __name__)

@web_views_bp.route("/")
def index():
    """Renders the AuraCycle AI Glassmorphic Dashboard."""
    return render_template("index.html")
