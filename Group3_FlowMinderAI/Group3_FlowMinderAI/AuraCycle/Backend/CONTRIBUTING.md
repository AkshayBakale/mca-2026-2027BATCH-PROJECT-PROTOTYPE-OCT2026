# Contributing to AuraCycle AI

Thank you for contributing to **AuraCycle AI**! This project is maintained collaboratively across 4 core modules:

1. **Folder 1: Data Pipeline (`01_data_pipeline/`)**
   - Responsible for biological cohort datasets, Bayesian de-aliasing filtering, and BeautifulSoup clinical literature harvesting.
2. **Folder 2: ML Engine (`02_ml_engine/`)**
   - Responsible for LightGBM quantile regression ($q_{10}, q_{50}, q_{90}$), feature transformations, collaborative filtering, and Oxford CEBM evidence ranking.
3. **Folder 3: Backend API (`03_backend_api/`)**
   - Responsible for Flask application factory, REST blueprints, LLM clinical reasoning orchestration, and PDF consultation export.
4. **Folder 4: Frontend Dashboard (`04_frontend_dashboard/`)**
   - Responsible for glassmorphic UI, Chart.js arrival density curves, phase dials, and interactive recommendation cards.

---

## Development Workflow

1. **Branching Model**:
   - Create a feature branch matching your folder assignment:
     - `feature/data-pipeline-<description>`
     - `feature/ml-engine-<description>`
     - `feature/backend-api-<description>`
     - `feature/frontend-<description>`
2. **Coding Standards**:
   - Write clean, type-hinted Python 3.10+ code with PEP8 styling.
   - Maintain non-leaking feature transformations (no target leakage from future cycle dates).
   - Document any clinical rationale citing Oxford CEBM evidence or gynecological guidelines.
3. **Testing**:
   - Run the automated test suite before opening a pull request:
     ```bash
     pytest tests/
     ```
   - Ensure all 29 tests pass with zero errors.
4. **Pull Requests**:
   - Describe changes made in your specific folder.
   - Attach test output or screenshots if UI modifications were introduced.
