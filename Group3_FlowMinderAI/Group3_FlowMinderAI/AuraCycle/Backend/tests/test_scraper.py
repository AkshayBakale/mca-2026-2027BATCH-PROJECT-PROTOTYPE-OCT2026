"""
test_scraper.py - Unit Tests for BeautifulSoup Clinical Literature Harvester
"""

import sys
import os
import pytest
from importlib import import_module

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [root_dir, os.path.join(root_dir, "01_data_pipeline")]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    scraper_mod = import_module("01_data_pipeline.literature_scraper")
    ClinicalLiteratureScraper = scraper_mod.ClinicalLiteratureScraper
    clinical_scraper = scraper_mod.clinical_scraper
except Exception:
    import literature_scraper as scraper_mod
    ClinicalLiteratureScraper = scraper_mod.ClinicalLiteratureScraper
    clinical_scraper = scraper_mod.clinical_scraper

SAMPLE_HTML = """
<html>
<body>
    <article class="clinical-guideline" data-source="ACOG Practice Bulletin" data-level="1a">
        <h3 class="guideline-title">Magnesium Glycinate in Primary Dysmenorrhea</h3>
        <span class="target-condition">dysmenorrhea</span>
        <span class="target-symptom">cramps_pelvic</span>
        <p class="clinical-pearl">Prostaglandin F2a inhibition reduces uterine myometrial hypercontractility.</p>
    </article>
    <article class="clinical-guideline" data-source="Endocrine Society" data-level="1b">
        <h3 class="guideline-title">Inositol Supplementation for Ovulatory Restoration</h3>
        <span class="target-condition">PCOS</span>
        <span class="target-symptom">anovulation</span>
        <p class="clinical-pearl">40:1 Myo to D-Chiro inositol ratio improves follicular maturation.</p>
    </article>
</body>
</html>
"""

def test_beautifulsoup_html_parsing():
    """Verify BeautifulSoup parses HTML guidelines into structured records."""
    scraper = ClinicalLiteratureScraper.__new__(ClinicalLiteratureScraper)
    scraper.corpus_path = "mock.html"
    guidelines = scraper.parse_html_evidence(SAMPLE_HTML, source_label="Test Suite")
    
    assert len(guidelines) == 2
    assert guidelines[0]["evidence_level"] == "1a"
    assert "Magnesium" in guidelines[0]["title"]
    assert guidelines[1]["target_condition"] == "PCOS"
    assert "Inositol" in guidelines[1]["title"]

def test_clinical_scraper_database_query():
    """Verify querying bundled evidence by symptom or condition keywords."""
    res = clinical_scraper.query_evidence_by_symptom_or_condition("pcos")
    assert isinstance(res, list)
    if len(res) > 0:
        assert any("pcos" in item["target_condition"].lower() or "pcos" in item["title"].lower() for item in res)
