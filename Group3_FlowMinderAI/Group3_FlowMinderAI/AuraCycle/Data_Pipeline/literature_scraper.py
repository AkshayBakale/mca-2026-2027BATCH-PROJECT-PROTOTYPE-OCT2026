"""
literature_scraper.py - BeautifulSoup Clinical Literature & Guideline Harvester
Parses clinical guidelines, medical articles, and evidence briefs from HTML documents
or live endpoints into structured evidence-based clinical intervention records.
"""

import os
import json
import logging
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
import requests

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_PATH = os.path.join(
    os.path.dirname(__file__), "dataset", "clinical_evidence_corpus.html"
)
OUTPUT_JSON_PATH = os.path.join(
    os.path.dirname(__file__), "dataset", "clinical_knowledge.json"
)

class ClinicalLiteratureScraper:
    """
    BeautifulSoup-powered parser that extracts structured clinical evidence,
    treatment recommendations, evidence grades, and biological mechanisms from HTML.
    """

    def __init__(self, corpus_html_path: Optional[str] = None):
        self.corpus_path = corpus_html_path or DEFAULT_CORPUS_PATH
        self.evidence_database: List[Dict] = []
        self._load_and_parse_local_corpus()

    def _load_and_parse_local_corpus(self) -> List[Dict]:
        """Loads and parses the bundled local HTML clinical evidence corpus using BeautifulSoup."""
        if not os.path.exists(self.corpus_path):
            logger.warning(f"Corpus file not found at {self.corpus_path}")
            return []

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        self.evidence_database = self.parse_html_evidence(html_content, source_label="Bundled Clinical Guidelines")
        self._export_knowledge_json()
        return self.evidence_database

    def parse_html_evidence(self, html_content: str, source_label: str = "Web Source") -> List[Dict]:
        """
        Uses BeautifulSoup to extract clinical guidelines and interventions from arbitrary HTML.
        """
        soup = BeautifulSoup(html_content, "html.parser")
        extracted_guidelines = []

        articles = soup.find_all("article", class_="clinical-guideline")
        # If no specific article tags found, look for general article, section, or card elements
        if not articles:
            articles = soup.find_all(["article", "section", "div"], attrs={"data-level": True})

        for art in articles:
            source = art.get("data-source", source_label)
            level = art.get("data-level", "2b")
            title_tag = art.find(["h1", "h2", "h3"], class_="guideline-title")
            title = title_tag.get_text(strip=True) if title_tag else "Clinical Practice Recommendation"
            
            cond_tag = art.find("span", class_="target-condition")
            condition = cond_tag.get_text(strip=True) if cond_tag else "general_menstrual"
            
            symp_tag = art.find("span", class_="target-symptom")
            symptom = symp_tag.get_text(strip=True) if symp_tag else "unspecified"
            
            summary_tag = art.find("section", class_="summary")
            summary = summary_tag.get_text(strip=True) if summary_tag else ""

            interventions = []
            intervention_items = art.find_all("div", class_="intervention-item")
            for item in intervention_items:
                int_id = item.get("data-id", f"INT-{len(interventions)+1:03d}")
                category = item.get("data-category", "lifestyle")
                strength = item.get("data-strength", "Moderate")
                
                int_title_tag = item.find(["h3", "h4", "h5", "b", "strong"])
                int_name = int_title_tag.get_text(strip=True) if int_title_tag else "Recommended Intervention"
                
                dosage_tag = item.find("p", class_="dosage")
                dosage = dosage_tag.get_text(strip=True) if dosage_tag else ""
                
                mech_tag = item.find("p", class_="mechanism")
                mechanism = mech_tag.get_text(strip=True) if mech_tag else ""
                
                conf_tag = item.find("span", class_="confidence")
                try:
                    confidence = float(conf_tag.get_text(strip=True)) if conf_tag else 0.80
                except ValueError:
                    confidence = 0.80

                interventions.append({
                    "intervention_id": int_id,
                    "name": int_name,
                    "category": category,
                    "strength": strength,
                    "dosage_instructions": dosage,
                    "biological_mechanism": mechanism,
                    "confidence_score": confidence
                })

            guideline_record = {
                "source": source,
                "evidence_level": level,
                "title": title,
                "target_condition": condition,
                "target_symptom": symptom,
                "summary": summary,
                "interventions": interventions
            }
            extracted_guidelines.append(guideline_record)

        return extracted_guidelines

    def scrape_url(self, url: str, timeout: int = 5) -> Dict:
        """
        Fetches an external HTML page and parses clinical evidence using BeautifulSoup.
        Falls back to local corpus if the network request fails or times out.
        """
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": "AuraCycle-AI-ClinicalHarvester/1.0"})
            resp.raise_for_status()
            parsed = self.parse_html_evidence(resp.text, source_label=url)
            return {
                "status": "success",
                "url": url,
                "records_extracted": len(parsed),
                "data": parsed
            }
        except Exception as e:
            logger.warning(f"Live scrape failed for {url} ({e}); serving local verified evidence corpus.")
            return {
                "status": "fallback_local",
                "error": str(e),
                "url": url,
                "records_extracted": len(self.evidence_database),
                "data": self.evidence_database
            }

    def query_evidence_by_symptom_or_condition(self, query: str) -> List[Dict]:
        """
        Searches the extracted clinical knowledge base for matching conditions or symptoms.
        """
        query_norm = query.lower().strip()
        matched = []
        for g in self.evidence_database:
            cond = g["target_condition"].lower()
            symp = g["target_symptom"].lower()
            title = g["title"].lower()
            summary = g["summary"].lower()

            if query_norm in cond or query_norm in symp or query_norm in title or query_norm in summary:
                matched.append(g)
                continue

            # Check individual interventions
            for item in g["interventions"]:
                if query_norm in item["name"].lower() or query_norm in item["biological_mechanism"].lower():
                    matched.append(g)
                    break

        return matched if matched else self.evidence_database

    def _export_knowledge_json(self):
        """Serializes parsed clinical database to JSON for rapid downstream consumption."""
        try:
            os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
            with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(self.evidence_database, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to export knowledge json: {e}")

# Global instance for shared API & recommendation usage
clinical_scraper = ClinicalLiteratureScraper()
