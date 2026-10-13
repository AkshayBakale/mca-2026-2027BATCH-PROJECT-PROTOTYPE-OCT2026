"""
evidence_ranker.py - Clinical Evidence Ranking System
Implements an Oxford Centre for Evidence-Based Medicine (CEBM) hierarchical ranking engine
to evaluate and order biological, lifestyle, and clinical factors influencing menstrual cycle variance.
"""

from typing import Dict, List, Any
import numpy as np

# Oxford CEBM Hierarchy Weights (Higher score = higher quality empirical evidence)
CEBM_EVIDENCE_LEVELS = {
    "1a": {"weight": 5.0, "grade": "A", "description": "Systematic review of homogenous RCTs"},
    "1b": {"weight": 4.5, "grade": "A", "description": "Individual high-quality randomized controlled trial"},
    "2a": {"weight": 3.8, "grade": "B", "description": "Systematic review of prospective cohort studies"},
    "2b": {"weight": 3.2, "grade": "B", "description": "Individual prospective cohort study or low-power RCT"},
    "3a": {"weight": 2.6, "grade": "C", "description": "Systematic review of case-control studies"},
    "3b": {"weight": 2.2, "grade": "C", "description": "Individual case-control study"},
    "4":  {"weight": 1.6, "grade": "C", "description": "Case series or poor-quality cohort/case-control"},
    "5":  {"weight": 1.0, "grade": "D", "description": "Expert opinion without explicit critical appraisal or bench science"}
}

# Empirical literature effect size & evidence level database
CLINICAL_DISRUPTOR_CATALOG = {
    "emergency_contraception_recent": {
        "label": "Recent Emergency Contraceptive Intake",
        "cebm_level": "1b",
        "expected_delay_days": 4.8,
        "citation": "ACOG Practice Bulletin No. 152: Emergency Contraception; Cochrane Review CD001324.",
        "mechanism": "High-dose levonorgestrel disrupts or delays the luteinizing hormone (LH) surge for 4-7 days if taken pre-ovulatory."
    },
    "pcos_diagnosis": {
        "label": "Polycystic Ovary Syndrome (PCOS)",
        "cebm_level": "1a",
        "expected_delay_days": 6.5,
        "citation": "Rotterdam ESHRE/ASRM Consensus; Endocrine Society Clinical Guidelines (JCEM 2013).",
        "mechanism": "Ovarian theca hyperandrogenism and insulin resistance arrest follicular development at the pre-antral stage."
    },
    "thyroid_condition": {
        "label": "Thyroid Function Disruption (Hypo/Hyperthyroidism)",
        "cebm_level": "1b",
        "expected_delay_days": 3.6,
        "citation": "American Thyroid Association Guidelines for Thyroid Disease in Women (Thyroid 2017).",
        "mechanism": "Altered TRH secretion impacts prolactin synthesis and directly impairs sex-hormone-binding globulin (SHBG)."
    },
    "endometriosis": {
        "label": "Endometriosis Pathophysiology",
        "cebm_level": "1a",
        "expected_delay_days": 2.9,
        "citation": "ESHRE Endometriosis Guideline (Hum Reprod 2022); Kennedy et al.",
        "mechanism": "Peritoneal chronic cytokine elevation and localized prostaglandin hypersecretion alter luteal corpus luteum longevity."
    },
    "acute_stress_velocity": {
        "label": "Acute Psychosocial / Academic Stress Velocity",
        "cebm_level": "2a",
        "expected_delay_days": 3.2,
        "citation": "Harvard Nurses' Health Study II; European Society of Human Reproduction (2018).",
        "mechanism": "Cortisol hypersecretion suppresses hypothalamic GnRH pulsatility, postponing LH-driven follicular rupture."
    },
    "circadian_sleep_deficit": {
        "label": "Circadian Sleep Deficit & Sleep Fragmentation",
        "cebm_level": "2b",
        "expected_delay_days": 2.4,
        "citation": "Journal of Clinical Sleep Medicine (2020); Sleep Research Society Meta-analysis.",
        "mechanism": "Disrupted nocturnal melatonin suppression shifts suprachiasmatic nucleus signaling to kisspeptin neurons."
    },
    "bmi_disruption_flag": {
        "label": "Metabolic Extremes (Underweight or Obese BMI)",
        "cebm_level": "2a",
        "expected_delay_days": 3.5,
        "citation": "Lancet Endocrinology: Adiposity, Leptin Signaling and HPO Axis (2021).",
        "mechanism": "Low leptin in underweight or peripheral aromatization in obesity disrupts negative estrogen feedback loops."
    },
    "weight_change_30d_kg": {
        "label": "Rapid 30-Day Weight Fluctuation",
        "cebm_level": "2b",
        "expected_delay_days": 2.8,
        "citation": "American Society for Reproductive Medicine (ASRM) Practice Committee Report.",
        "mechanism": "Negative or positive acute energy balance triggers central neuroendocrine metabolic adaptation."
    },
    "recent_illness": {
        "label": "Acute Febrile or Systemic Illness",
        "cebm_level": "3b",
        "expected_delay_days": 2.5,
        "citation": "British Medical Journal Clinical Review: Systemic Inflammation & Ovulatory Arrest.",
        "mechanism": "Circulating pyrogenic cytokines (IL-1, TNF-alpha) transiently suppress gonadotropin release."
    },
    "recent_travel": {
        "label": "Circadian Phase Shift (Trans-meridian Travel)",
        "cebm_level": "4",
        "expected_delay_days": 1.8,
        "citation": "Aviation, Space & Environmental Medicine Chronobiology Findings.",
        "mechanism": "Desynchrony of peripheral endocrine clocks and central pacemaker temporarily attenuates cycle regularity."
    }
}

class ClinicalEvidenceRanker:
    """
    Ranks patient-specific cycle disruption drivers using Oxford CEBM evidence weights
    and observed clinical deviation severity.
    """

    def __init__(self):
        self.catalog = CLINICAL_DISRUPTOR_CATALOG
        self.levels = CEBM_EVIDENCE_LEVELS

    def evaluate_patient_disruptors(self, patient_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Calculates individual evidence rank score for every active clinical/lifestyle driver:
        Score = Clinical_Severity * CEBM_Weight * Expected_Delay_Days
        """
        ranked_items = []

        # 1. Emergency contraception
        if int(patient_features.get("emergency_contraception_recent", 0)) == 1:
            ranked_items.append(self._score_disruptor("emergency_contraception_recent", severity=1.0))

        # 2. PCOS
        if int(patient_features.get("pcos_diagnosis", 0)) == 1:
            ranked_items.append(self._score_disruptor("pcos_diagnosis", severity=1.0))

        # 3. Thyroid
        if int(patient_features.get("thyroid_condition", 0)) == 1:
            ranked_items.append(self._score_disruptor("thyroid_condition", severity=0.9))

        # 4. Endometriosis
        if int(patient_features.get("endometriosis", 0)) == 1:
            ranked_items.append(self._score_disruptor("endometriosis", severity=0.85))

        # 5. Stress Velocity
        stress_level = float(patient_features.get("stress_level", 5.0))
        stress_velocity = float(patient_features.get("acute_stress_velocity", 0.0))
        if stress_level >= 7.0 or stress_velocity > 0.25:
            severity = min(1.0, max(0.2, (stress_level / 10.0) + max(0.0, stress_velocity)))
            ranked_items.append(self._score_disruptor("acute_stress_velocity", severity=severity))

        # 6. Sleep Deficit
        sleep_hours = float(patient_features.get("sleep_hours", 7.5))
        if sleep_hours < 7.0:
            severity = min(1.0, (7.5 - sleep_hours) / 3.5)
            ranked_items.append(self._score_disruptor("circadian_sleep_deficit", severity=severity))

        # 7. BMI Disruption
        bmi = float(patient_features.get("bmi", 22.0))
        if bmi < 18.5 or bmi > 30.0:
            dev = (18.5 - bmi) if bmi < 18.5 else (bmi - 30.0)
            severity = min(1.0, max(0.3, dev / 5.0))
            ranked_items.append(self._score_disruptor("bmi_disruption_flag", severity=severity))

        # 8. Weight change
        weight_change = abs(float(patient_features.get("weight_change_30d_kg", 0.0)))
        if weight_change >= 2.0:
            severity = min(1.0, weight_change / 5.0)
            ranked_items.append(self._score_disruptor("weight_change_30d_kg", severity=severity))

        # 9. Recent illness
        if int(patient_features.get("recent_illness", 0)) == 1:
            ranked_items.append(self._score_disruptor("recent_illness", severity=0.8))

        # 10. Travel
        if int(patient_features.get("recent_travel", 0)) == 1:
            ranked_items.append(self._score_disruptor("recent_travel", severity=0.6))

        # Sort descending by evidence_rank_score
        ranked_items.sort(key=lambda x: x["evidence_rank_score"], reverse=True)

        # Assign ordinal rank
        for idx, item in enumerate(ranked_items, start=1):
            item["rank"] = idx

        return ranked_items

    def _score_disruptor(self, key: str, severity: float) -> Dict[str, Any]:
        spec = self.catalog[key]
        level_spec = self.levels[spec["cebm_level"]]
        weight = level_spec["weight"]
        expected_delay = spec["expected_delay_days"]

        # Composite score calculation
        rank_score = round(severity * weight * expected_delay, 2)
        estimated_delay_impact = round(severity * expected_delay, 1)

        return {
            "factor_key": key,
            "factor_name": spec["label"],
            "severity_index": round(severity, 2),
            "cebm_level": spec["cebm_level"],
            "cebm_grade": level_spec["grade"],
            "cebm_weight": weight,
            "evidence_description": level_spec["description"],
            "estimated_shift_days": estimated_delay_impact,
            "evidence_rank_score": rank_score,
            "biological_mechanism": spec["mechanism"],
            "citation": spec["citation"]
        }

# Global shared instance
evidence_ranker = ClinicalEvidenceRanker()
