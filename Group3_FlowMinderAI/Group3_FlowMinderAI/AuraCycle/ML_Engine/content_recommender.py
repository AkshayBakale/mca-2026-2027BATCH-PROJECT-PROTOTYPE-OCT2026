"""
content_recommender.py - Content-Based Clinical Recommendation Engine
Calculates cosine similarity between patient biological profile & NLP symptom tags
and a vector space of evidence-backed clinical, nutritional, and lifestyle interventions.
"""

import math
from typing import Dict, List, Any, Optional
import numpy as np

# Clinical intervention feature space
INTERVENTION_PROFILES = [
    {
        "id": "CB-01",
        "title": "Dual-Phase Magnesium & Zinc Protocol",
        "type": "Nutraceutical",
        "description": "High-bioavailability Magnesium Bisglycinate (300mg) combined with Zinc Picolinate (15mg).",
        "clinical_tags": ["cramps_pelvic", "dysmenorrhea", "headache_migraine", "sleep_deficit", "stress"],
        "target_phase": ["Luteal", "Menstrual"],
        "evidence_grade": "A",
        "expected_benefit": "35-50% reduction in uterine cramping intensity via smooth muscle relaxation."
    },
    {
        "id": "CB-02",
        "title": "Targeted Myo-Inositol & Folate Therapy",
        "type": "Endocrine Protocol",
        "description": "40:1 ratio Myo-Inositol to D-Chiro-Inositol with active Methylfolate.",
        "clinical_tags": ["pcos_diagnosis", "cycle_irregularity", "insulin_resistance", "ovulatory_delay", "acne"],
        "target_phase": ["Follicular", "Ovulatory"],
        "evidence_grade": "A",
        "expected_benefit": "Restores spontaneous regular ovulatory cycles and reduces androgenic follicular arrest."
    },
    {
        "id": "CB-03",
        "title": "Anti-Inflammatory Omega-3 & Curcumin Protocol",
        "type": "Nutraceutical",
        "description": "2000mg EPA/DHA concentrated fish oil paired with 500mg bio-optimized Curcumin.",
        "clinical_tags": ["endometriosis", "fibroids", "cramps_pelvic", "bloating_digestive", "chronic_inflammation"],
        "target_phase": ["Luteal", "Menstrual"],
        "evidence_grade": "A",
        "expected_benefit": "Inhibits COX-2 and NF-kB inflammatory cascades, relieving cyclic pelvic pain."
    },
    {
        "id": "CB-04",
        "title": "Circadian Zeitgeber Reset & Sleep Hygiene",
        "type": "Chronobiology",
        "description": "Morning outdoor retinal blue-light exposure (15 min) + 0 lux nocturnal environment.",
        "clinical_tags": ["sleep_deficit", "circadian_delay", "fatigue_lethargy", "stress", "anxiety"],
        "target_phase": ["Follicular", "Luteal"],
        "evidence_grade": "B",
        "expected_benefit": "Synchronizes central GnRH pulsatility and lowers nocturnal cortisol spikes."
    },
    {
        "id": "CB-05",
        "title": "Vitex Agnus-Castus Luteal Support Protocol",
        "type": "Botanical Medicine",
        "description": "Standardized Chasteberry extract (20-40mg daily) taken in the morning.",
        "clinical_tags": ["mood_anxiety", "breast_tenderness", "luteal_phase_defect", "spotting_before_period", "headache_migraine"],
        "target_phase": ["Luteal", "Late Luteal"],
        "evidence_grade": "B",
        "expected_benefit": "Modulates pituitary prolactin secretion and stabilizes corpus luteum progesterone synthesis."
    },
    {
        "id": "CB-06",
        "title": "Ginger Rhizome (Zingiber) Acute Dysmenorrhea Protocol",
        "type": "Botanical Medicine",
        "description": "250mg standardized ginger rhizome capsule 4x daily starting 2 days before menses.",
        "clinical_tags": ["cramps_pelvic", "dysmenorrhea", "nausea", "bloating_digestive"],
        "target_phase": ["Late Luteal", "Menstrual"],
        "evidence_grade": "A",
        "expected_benefit": "Demonstrated non-inferiority to mefenamic acid and ibuprofen in clinical pain reduction."
    },
    {
        "id": "CB-07",
        "title": "Low-Glycemic Load Mediterranean Nutrition Framework",
        "type": "Dietary Pattern",
        "description": "High fiber (>35g/day), polyphenols, cruciferous indoles (DIM), and monounsaturated lipids.",
        "clinical_tags": ["pcos_diagnosis", "bmi_disruption", "bloating_digestive", "fatigue_lethargy"],
        "target_phase": ["Follicular", "Ovulatory", "Luteal"],
        "evidence_grade": "A",
        "expected_benefit": "Improves insulin sensitivity and assists hepatic clearance of excess estrogen metabolites."
    },
    {
        "id": "CB-08",
        "title": "Adaptogenic Cortisol Buffer (Withania & L-Theanine)",
        "type": "Nutraceutical",
        "description": "300mg KSM-66 Ashwagandha with 200mg pure L-Theanine.",
        "clinical_tags": ["stress", "anxiety", "acute_stress_velocity", "sleep_deficit"],
        "target_phase": ["Follicular", "Luteal"],
        "evidence_grade": "B",
        "expected_benefit": "Reduces serum cortisol velocity, preventing stress-mediated suppression of the LH surge."
    }
]

# Vocabulary of all clinical tags in the vector space
VOCABULARY = sorted(list({tag for item in INTERVENTION_PROFILES for tag in item["clinical_tags"]}))

class ContentBasedRecommender:
    """
    Computes vector cosine similarity between patient clinical profile / NLP tags
    and the intervention catalog.
    """

    def __init__(self):
        self.catalog = INTERVENTION_PROFILES
        self.vocab = VOCABULARY
        self.vocab_index = {word: i for i, word in enumerate(self.vocab)}
        self.item_vectors = self._build_item_vectors()

    def _build_item_vectors(self) -> np.ndarray:
        """Constructs one-hot/weighted tag vectors for each intervention."""
        vectors = np.zeros((len(self.catalog), len(self.vocab)))
        for i, item in enumerate(self.catalog):
            for tag in item["clinical_tags"]:
                if tag in self.vocab_index:
                    vectors[i, self.vocab_index[tag]] = 1.0
            # Normalize vector
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
        return vectors

    def build_patient_profile_vector(
        self,
        patient_data: Dict[str, Any],
        nlp_symptoms: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Builds a weighted clinical tag vector for the patient.
        Combines biometric flags and extracted NLP entities.
        """
        user_vector = np.zeros(len(self.vocab))

        # Check conditions
        if int(patient_data.get("pcos_diagnosis", 0)) == 1:
            self._add_weight(user_vector, "pcos_diagnosis", 2.0)
            self._add_weight(user_vector, "cycle_irregularity", 1.5)
            self._add_weight(user_vector, "insulin_resistance", 1.5)

        if int(patient_data.get("endometriosis", 0)) == 1:
            self._add_weight(user_vector, "endometriosis", 2.0)
            self._add_weight(user_vector, "chronic_inflammation", 1.5)
            self._add_weight(user_vector, "cramps_pelvic", 1.5)

        if int(patient_data.get("fibroids", 0)) == 1:
            self._add_weight(user_vector, "fibroids", 2.0)

        # Stress & Sleep
        stress = float(patient_data.get("stress_level", 5.0))
        if stress >= 6.5:
            self._add_weight(user_vector, "stress", stress / 5.0)
            self._add_weight(user_vector, "anxiety", (stress - 4.0) / 4.0)
            self._add_weight(user_vector, "acute_stress_velocity", 1.5)

        sleep = float(patient_data.get("sleep_hours", 7.5))
        if sleep < 7.0:
            self._add_weight(user_vector, "sleep_deficit", (7.5 - sleep) / 2.0)
            self._add_weight(user_vector, "circadian_delay", 1.2)

        bmi = float(patient_data.get("bmi", 22.0))
        if bmi < 18.5 or bmi > 30.0:
            self._add_weight(user_vector, "bmi_disruption", 1.5)

        if int(patient_data.get("spotting_before_period", 0)) == 1:
            self._add_weight(user_vector, "spotting_before_period", 1.5)
            self._add_weight(user_vector, "luteal_phase_defect", 1.2)

        # Incorporate NLP parsed symptoms
        if nlp_symptoms:
            for symp in nlp_symptoms:
                symp_lower = symp.lower().strip()
                if "cramp" in symp_lower or "pain" in symp_lower:
                    self._add_weight(user_vector, "cramps_pelvic", 2.0)
                    self._add_weight(user_vector, "dysmenorrhea", 2.0)
                elif "bloat" in symp_lower or "gas" in symp_lower:
                    self._add_weight(user_vector, "bloating_digestive", 2.0)
                elif "migraine" in symp_lower or "headache" in symp_lower:
                    self._add_weight(user_vector, "headache_migraine", 2.0)
                elif "fatigue" in symp_lower or "tired" in symp_lower:
                    self._add_weight(user_vector, "fatigue_lethargy", 1.8)
                elif "mood" in symp_lower or "anxiety" in symp_lower or "irritab" in symp_lower:
                    self._add_weight(user_vector, "mood_anxiety", 1.8)
                elif "breast" in symp_lower or "tender" in symp_lower:
                    self._add_weight(user_vector, "breast_tenderness", 2.0)

        # Normalize patient vector
        norm = np.linalg.norm(user_vector)
        if norm > 0:
            user_vector /= norm
        return user_vector

    def _add_weight(self, vector: np.ndarray, tag: str, weight: float):
        if tag in self.vocab_index:
            vector[self.vocab_index[tag]] += weight

    def recommend(
        self,
        patient_data: Dict[str, Any],
        nlp_symptoms: Optional[List[str]] = None,
        top_n: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Scores all catalog interventions via Cosine Similarity with patient profile vector:
        similarity = (u . i) / (||u|| ||i||)
        """
        user_vec = self.build_patient_profile_vector(patient_data, nlp_symptoms)
        
        # If user vector is zero (no specific disruptions logged), provide general balanced recommendations
        if np.linalg.norm(user_vec) == 0:
            user_vec = np.ones(len(self.vocab)) / math.sqrt(len(self.vocab))

        # Vector dot products
        similarities = np.dot(self.item_vectors, user_vec)
        
        ranked_indices = np.argsort(similarities)[::-1]
        
        recommendations = []
        for rank, idx in enumerate(ranked_indices[:top_n], start=1):
            item = self.catalog[idx]
            match_score = float(similarities[idx])
            recommendations.append({
                "rank": rank,
                "intervention_id": item["id"],
                "title": item["title"],
                "category": item["type"],
                "description": item["description"],
                "expected_benefit": item["expected_benefit"],
                "evidence_grade": item["evidence_grade"],
                "target_phases": item["target_phase"],
                "relevance_match": f"{max(45, int(match_score * 100))}% Profile Match",
                "matched_clinical_tags": [t for t in item["clinical_tags"] if t in self.vocab_index and user_vec[self.vocab_index[t]] > 0]
            })

        return recommendations

# Global shared instance
content_recommender = ContentBasedRecommender()
