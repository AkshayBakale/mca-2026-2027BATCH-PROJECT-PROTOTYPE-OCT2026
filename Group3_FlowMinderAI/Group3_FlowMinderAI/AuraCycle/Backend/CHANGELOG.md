# Changelog

All notable changes to the **AuraCycle AI** project are documented in this file.

## [1.0.0] - 2026-09-26

### Added
- **4-Folder Enterprise Architecture**:
  - `01_data_pipeline`: Bayesian de-aliasing, Dirichlet-Multinomial phase engine, BeautifulSoup clinical guideline scraper, and 60k prior-based synthetic cohort dataset.
  - `02_ml_engine`: LightGBM quantile regression ($\tau \in [0.10, 0.50, 0.90]$), 43+ biological features, memory-based user collaborative filtering, content-based cosine recommendation, and Oxford CEBM evidence ranking.
  - `03_backend_api`: Flask application factory, REST blueprints (`api_predict`, `api_journal`, `api_llm`, `api_recommend`, `web_views`), and binary PDF consultation export.
  - `04_frontend_dashboard`: Glassmorphic UI with animated Menstrual Phase Dial, Gaussian prediction bell curve, recommendation cards, and evidence hierarchy panels.
- **Root Entrypoint & Requirements**:
  - Root `run.py` to start the production app directly.
  - Pinned `requirements.txt` including `beautifulsoup4` and scientific packages.
- **Comprehensive Test Suite**:
  - 29 unit and integration tests across all 4 modules (`test_api`, `test_bayesian`, `test_inference`, `test_multinomial`, `test_nlp_parser`, `test_recommendations`, `test_scraper`).
- **CI / CD**:
  - GitHub Actions automated testing workflow in `.github/workflows/ci.yml`.
