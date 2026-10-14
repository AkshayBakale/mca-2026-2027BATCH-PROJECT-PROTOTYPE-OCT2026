"""
config.py - Centralized Configuration for AuraCycle AI
Handles environment configuration, paths, model artifacts, and service keys.
"""

import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "auracycle-ai-super-secret-production-key-2026")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ROOT_DIR = os.path.dirname(BASE_DIR)
    
    # Check 02_ml_engine/models first, fallback to models
    candidate_model_dirs = [
        os.path.join(ROOT_DIR, "02_ml_engine", "models"),
        os.path.join(ROOT_DIR, "auracycle_core", "models"),
        os.path.join(BASE_DIR, "models")
    ]
    MODEL_DIR = next((d for d in candidate_model_dirs if os.path.exists(d)), candidate_model_dirs[0])
    
    # Dataset dir
    candidate_data_dirs = [
        os.path.join(ROOT_DIR, "01_data_pipeline", "dataset"),
        os.path.join(ROOT_DIR, "auracycle_core", "data"),
        os.path.join(BASE_DIR, "data")
    ]
    DATA_DIR = next((d for d in candidate_data_dirs if os.path.exists(d)), candidate_data_dirs[0])
    
    # Model Artifact Paths
    MODEL_Q10_PATH = os.path.join(MODEL_DIR, "lgbm_quantile_10.joblib")
    MODEL_Q50_PATH = os.path.join(MODEL_DIR, "lgbm_quantile_50.joblib")
    MODEL_Q90_PATH = os.path.join(MODEL_DIR, "lgbm_quantile_90.joblib")
    FEATURE_METADATA_PATH = os.path.join(MODEL_DIR, "feature_metadata.joblib")
    
    # Bayesian Prior Parameters
    BAYESIAN_PRIORS = {
        1: 0.88,
        2: 0.08,
        3: 0.028,
        4: 0.010,
        5: 0.002
    }
    
    # LLM Settings
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:8b")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    
    # App Settings
    DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1")
    PORT = int(os.environ.get("PORT", 5000))
    CORS_ORIGINS = "*"

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

class TestingConfig(Config):
    TESTING = True
    DEBUG = True

config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig
}
