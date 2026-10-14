"""
run.py - Production & Development WSGI Entrypoint for AuraCycle AI
Master entrypoint to serve the full 4-folder modular architecture.
"""

import os
import sys

# Add project root and 03_backend_api to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(project_root, "03_backend_api")

for p in [project_root, backend_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app import create_app
from app.config import Config

# Create Flask application instance
env = os.environ.get("FLASK_ENV", "development")
app = create_app(env)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", Config.PORT))
    debug = Config.DEBUG
    print("=" * 68)
    print("   AuraCycle AI - Clinical Menstrual Health Intelligence System")
    print(f"   Listening on: http://localhost:{port} (Environment: {env})")
    print("=" * 68)
    app.run(host="0.0.0.0", port=port, debug=debug)
