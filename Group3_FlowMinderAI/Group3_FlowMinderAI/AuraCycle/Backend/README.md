# AuraCycle AI: Clinical Menstrual Health Intelligence System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Flask 3.0](https://img.shields.io/badge/Flask-3.0.0-emerald.svg)](https://flask.palletsprojects.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0.0-orange.svg)](https://lightgbm.readthedocs.io/)
[![BeautifulSoup4](https://img.shields.io/badge/BeautifulSoup-4.12.0-indigo.svg)](https://www.crummy.com/software/BeautifulSoup/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI Status](https://img.shields.io/badge/CI-passing%20(29%2F29%20tests)-brightgreen.svg)](.github/workflows/ci.yml)

AuraCycle AI is an enterprise-grade, clinical menstrual health intelligence system engineered to eliminate variance pollution in reproductive tracking, compute calibrated probabilistic arrival windows, and deliver evidence-backed hormonal wellness protocols.

---

## What Problem It Solves

1. **The Deterministic Precision Fallacy**: Traditional trackers predict an exact day (e.g., *"Period starts on Friday"*), ignoring biological variability. AuraCycle AI outputs an **80% Prediction Interval ($[q_{10}, q_{90}]$)** alongside a continuous Gaussian probability density function.
2. **Tracking Gap Variance Pollution**: When a user misses tracking for 2–3 months ($G > 1.8\mu$), commercial apps treat the multi-month gap as a single cycle, destroying prediction accuracy. AuraCycle AI uses **Bayesian inference** to de-alias and decompose gaps into latent cycles.
3. **Generic vs. Evidence-Based Interventions**: Standard trackers offer generic advice. AuraCycle AI maps patient disruptions (PCOS, acute stress, sleep deficits) to **clinically graded interventions** citing Oxford CEBM evidence.

---

## 4-Folder Architecture & Team Distribution

The project is structured into **4 distinct, production-ready modules** so that different team members can manage, contribute to, and upload their respective folders independently:

```text
auracycle_ai/
│
├── 01_data_pipeline/               # [FOLDER 1] Data Ingestion & Statistical Engines
│   ├── dataset/
│   │   ├── user_cohort_records.csv       # Clinical cohort dataset
│   │   ├── menstrual_cohort_60k.csv      # 60k prior-based synthetic cohort
│   │   ├── clinical_evidence_corpus.html # HTML clinical guideline documents
│   │   └── clinical_knowledge.json      # Structured medical knowledge base
│   ├── bayesian_cleaner.py         # Latent cycle de-aliasing & tracking gap filter P(K|G)
│   ├── dataset_generator.py        # 60k prior-based synthetic generator
│   ├── literature_scraper.py       # BeautifulSoup4 clinical guidelines harvester
│   └── multinomial_engine.py       # Dirichlet-Multinomial phase probability engine
│
├── 02_ml_engine/                   # [FOLDER 2] Machine Learning & Recommenders
│   ├── models/
│   │   ├── lgbm_quantile_10.joblib       # LightGBM tau = 0.10 regressor
│   │   ├── lgbm_quantile_50.joblib       # LightGBM tau = 0.50 median regressor
│   │   ├── lgbm_quantile_90.joblib       # LightGBM tau = 0.90 regressor
│   │   └── feature_metadata.joblib       # Feature mappings & metadata
│   ├── collaborative_filter.py     # User-User nearest-neighbor cohort recommender
│   ├── content_recommender.py      # TF-IDF / Cosine clinical protocol matcher
│   ├── evidence_ranker.py          # Oxford CEBM (Levels 1-5) clinical evidence ranker
│   ├── feature_transformer.py      # 43+ biological non-leaking features
│   ├── train_quantile_lgbm.py      # GroupKFold training for tau = 0.10, 0.50, 0.90
│   └── evaluate_metrics.py         # MAE, RMSE, PICP, Pinball Loss evaluation
│
├── 03_backend_api/                 # [FOLDER 3] Flask REST Application & Services
│   ├── app/
│   │   ├── __init__.py             # Application Factory & blueprint wiring
│   │   ├── config.py               # Environment configuration & model paths
│   │   ├── routes/
│   │   │   ├── api_predict.py      # /api/predict, /api/dealias, /api/clean-series
│   │   │   ├── api_journal.py      # /api/journal (NLP symptom & sentiment intake)
│   │   │   ├── api_llm.py          # /api/llm/explain, /api/llm/export-pdf (SOAP briefs)
│   │   │   ├── api_recommend.py    # /api/recommend/*, /api/evidence/rank, /api/phase/multinomial
│   │   │   └── web_views.py        # Serves dashboard UI (/)
│   │   └── services/
│   │       ├── ml_engine.py        # Model loading, monotonicity, arrival density curve
│   │       ├── nlp_parser.py       # Rule-based entity & sentiment modifier engine
│   │       ├── llm_explainer.py    # Dual-track patient narrative & SOAP generator
│   │       └── bayesian_cleaner.py # Real-time Bayesian de-aliasing service
│
├── 04_frontend_dashboard/          # [FOLDER 4] Glassmorphic User Interface
│   ├── static/
│   │   ├── css/style.css           # Glassmorphic dark styling, animations, neon glow
│   │   └── js/dashboard.js         # Chart.js density curve, SVG dial, API integrations
│   ├── templates/
│   │   └── index.html              # Responsive glassmorphic single-page application
│   └── index.html                  # Standalone entrypoint for Netlify
│
├── tests/                          # Automated Pytest Suite (29 tests)
│   ├── conftest.py
│   ├── test_api.py                 # REST API endpoints & dashboard tests
│   ├── test_bayesian.py            # Latent cycle decomposition & gap filter tests
│   ├── test_inference.py           # LightGBM quantile monotonicity & coverage tests
│   ├── test_multinomial.py         # Dirichlet-Multinomial phase distribution tests
│   ├── test_nlp_parser.py          # Symptom extraction & sentiment polarity tests
│   ├── test_recommendations.py     # Collaborative & content-based filtering tests
│   └── test_scraper.py             # BeautifulSoup clinical literature harvester tests
│
├── .github/workflows/ci.yml        # Automated CI workflow
├── .gitignore                      # Production gitignore
├── LICENSE                         # MIT License
├── CONTRIBUTING.md                 # Contribution guidelines
├── CHANGELOG.md                    # Release history
├── netlify.toml                    # Netlify deployment & API proxy configuration
├── requirements.txt                # Pinned production dependencies
└── run.py                          # Master WSGI & development server entrypoint
```

---

## Core Algorithms & Mathematical Formulations

### 1. Oxford CEBM Clinical Evidence Ranking
Patient-specific menstrual disruption drivers are classified and ranked strictly in accordance with the **Oxford Centre for Evidence-Based Medicine (CEBM)** hierarchy:
- **Level 1a/1b**: Systematic reviews and randomized controlled trials (e.g., *Inositol 40:1 ratio for PCOS ovulatory restoration*; *Magnesium Glycinate for primary dysmenorrhea*).
- **Level 2a/2b**: Systematic reviews of cohort studies (e.g., *Circadian rhythm entrainment for sleep deficit and GnRH regulation*).
- **Level 3–5**: Mechanistic, physiological, and expert consensus guidelines (e.g., *Metabolic insulin stabilization via low-glycemic dietary interventions*).

### 2. Dirichlet-Multinomial Continuous Phase Distribution
Menstrual cycle phases are modeled as a continuous probability vector over discrete phases:
$$\mathcal{P} = \{\text{Menstrual}, \text{Follicular}, \text{Ovulatory}, \text{Luteal}, \text{Late Luteal}\}$$
Conditional symptom emissions are calculated using a Dirichlet prior $\boldsymbol{\alpha}$ and Multinomial likelihood:
$$P(\text{Phase} \mid \mathbf{S}) \propto P(\mathbf{S} \mid \text{Phase}) \cdot P(\text{Phase})$$
Phase uncertainty is quantified via Shannon entropy in bits:
$$H(X) = -\sum_{i} P(\text{Phase}_i) \log_2 P(\text{Phase}_i)$$

### 3. BeautifulSoup Clinical Literature Harvester
Automated clinical guideline extractor that parses local medical documents (`clinical_evidence_corpus.html`) and live medical endpoints:
- Extracts guideline title, target gynecological condition, symptom target, Oxford evidence level, and clinical pearls.
- Indexes evidence into structured JSON catalogs for content-based matching.

### 4. Hybrid Recommendation Engine
- **Cohort Collaborative Filtering**: Identifies "Cycle Twins" among 60,000+ cohort records via cosine similarity:
  $$\text{Sim}(u, v) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$
  Predicts remedy utility based on real user satisfaction in identical biological clusters.
- **Content-Based Filtering**: Vectorizes patient biomarkers, clinical conditions, and unstructured journal entities into TF-IDF feature space, computing cosine similarity against evidence-backed clinical protocols.

### 5. Bayesian Latent Cycle De-Aliasing
Solves tracking gap variance pollution ($G > 1.8\mu$) without discarding valuable patient history:
$$P(K \mid G) = \frac{\mathcal{N}(G \mid K\mu, K\sigma^2) P(K)}{\sum_{k=1}^5 \mathcal{N}(G \mid k\mu, k\sigma^2) P(k)}$$

### 6. Tri-Quantile LightGBM Regression
Trained using pinball loss function:
$$\mathcal{L}_\tau(y, \hat{y}) = \max(\tau(y - \hat{y}), (\tau - 1)(y - \hat{y}))$$
Guarantees strict monotonicity: $q_{10} \le q_{50} \le q_{90}$.

---

## Detailed Steps for Execution

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.14
- Git

### Step 1: Install Dependencies
Open a terminal in the project root directory and run:
```bash
pip install -r requirements.txt
```

### Step 2: Run the Automated Test Suite
Verify that all 29 tests pass across every component:
```bash
pytest tests/
```
Output:
```text
tests/test_api.py ..........                                             [ 34%]
tests/test_bayesian.py .....                                             [ 51%]
tests/test_inference.py ...                                              [ 62%]
tests/test_multinomial.py ...                                            [ 72%]
tests/test_nlp_parser.py ...                                             [ 82%]
tests/test_recommendations.py ...                                        [ 93%]
tests/test_scraper.py ..                                                 [100%]
======================= 29 passed in 5.36s ========================
```

### Step 3: Start the Production Server
Run the master entrypoint script:
```bash
python run.py
```
Open your browser and navigate to:
```text
http://localhost:5000
```

---

## Deploying on Netlify

AuraCycle AI is pre-configured with `netlify.toml`:
1. Deploy the backend API on [Render.com](https://render.com) or [Railway.app](https://railway.app) (Python 3 environment running `python run.py`).
2. Update the backend URL in `netlify.toml`:
   ```toml
   [[redirects]]
     from = "/api/*"
     to = "https://your-api.onrender.com/api/:splat"
     status = 200
     force = true
   ```
3. Connect your repository to Netlify or drag-and-drop `04_frontend_dashboard` onto [app.netlify.com/drop](https://app.netlify.com/drop). Netlify serves the static UI and proxies all `/api/*` calls seamlessly.

---

## What to Upload to GitHub (Folder Allocation Guide)

Since different team members will upload the folders, here is the exact checklist:

### Person 1: Folder `01_data_pipeline/`
- `01_data_pipeline/dataset/` (`user_cohort_records.csv`, `menstrual_cohort_60k.csv`, `clinical_evidence_corpus.html`, `clinical_knowledge.json`)
- `01_data_pipeline/bayesian_cleaner.py`
- `01_data_pipeline/dataset_generator.py`
- `01_data_pipeline/literature_scraper.py`
- `01_data_pipeline/multinomial_engine.py`

### Person 2: Folder `02_ml_engine/`
- `02_ml_engine/models/` (`lgbm_quantile_10.joblib`, `lgbm_quantile_50.joblib`, `lgbm_quantile_90.joblib`, `feature_metadata.joblib`)
- `02_ml_engine/collaborative_filter.py`
- `02_ml_engine/content_recommender.py`
- `02_ml_engine/evidence_ranker.py`
- `02_ml_engine/feature_transformer.py`
- `02_ml_engine/train_quantile_lgbm.py`
- `02_ml_engine/evaluate_metrics.py`

### Person 3: Folder `03_backend_api/`
- `03_backend_api/app/__init__.py`
- `03_backend_api/app/config.py`
- `03_backend_api/app/routes/` (`api_predict.py`, `api_journal.py`, `api_llm.py`, `api_recommend.py`, `web_views.py`)
- `03_backend_api/app/services/` (`ml_engine.py`, `nlp_parser.py`, `llm_explainer.py`, `bayesian_cleaner.py`)

### Person 4: Folder `04_frontend_dashboard/`
- `04_frontend_dashboard/static/css/style.css`
- `04_frontend_dashboard/static/js/dashboard.js`
- `04_frontend_dashboard/templates/index.html`
- `04_frontend_dashboard/index.html`

### Shared / Root Level Files (Project Admin or Team Lead)
- `tests/` (all 7 test files + `conftest.py`)
- `.github/workflows/ci.yml`
- `.gitignore`
- `README.md` (This single comprehensive project README)
- `LICENSE`
- `CONTRIBUTING.md`
- `CHANGELOG.md`
- `netlify.toml`
- `requirements.txt`
- `run.py`

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
