"""
multinomial_engine.py - Multinomial Distribution Phase & Symptom Engine
Models menstrual cycle phase likelihoods and symptom manifestation profiles
using Dirichlet-Multinomial and discrete categorical probabilistic formulations.
"""

import math
from typing import Dict, List, Tuple, Optional
import numpy as np

CYCLE_PHASES = [
    "Menstrual",
    "Follicular",
    "Ovulatory",
    "Luteal",
    "Late Luteal"
]

SYMPTOM_CATEGORIES = [
    "cramps_pelvic",
    "bloating_digestive",
    "mood_anxiety",
    "fatigue_lethargy",
    "headache_migraine",
    "breast_tenderness"
]

# Baseline conditional symptom emission matrix: P(Symptom Category | Phase)
# Empirical distributions derived from clinical OB/GYN endocrinological literature
PHASE_SYMPTOM_EMISSIONS = {
    "Menstrual": {
        "cramps_pelvic": 0.42,
        "bloating_digestive": 0.18,
        "mood_anxiety": 0.12,
        "fatigue_lethargy": 0.20,
        "headache_migraine": 0.05,
        "breast_tenderness": 0.03
    },
    "Follicular": {
        "cramps_pelvic": 0.04,
        "bloating_digestive": 0.10,
        "mood_anxiety": 0.10,
        "fatigue_lethargy": 0.08,
        "headache_migraine": 0.05,
        "breast_tenderness": 0.03
    },
    "Ovulatory": {
        "cramps_pelvic": 0.15,  # Mittelschmerz
        "bloating_digestive": 0.12,
        "mood_anxiety": 0.08,
        "fatigue_lethargy": 0.07,
        "headache_migraine": 0.08,
        "breast_tenderness": 0.15
    },
    "Luteal": {
        "cramps_pelvic": 0.08,
        "bloating_digestive": 0.22,
        "mood_anxiety": 0.25,
        "fatigue_lethargy": 0.18,
        "headache_migraine": 0.12,
        "breast_tenderness": 0.25
    },
    "Late Luteal": {
        "cramps_pelvic": 0.25,
        "bloating_digestive": 0.28,
        "mood_anxiety": 0.32,
        "fatigue_lethargy": 0.26,
        "headache_migraine": 0.18,
        "breast_tenderness": 0.28
    }
}

class MultinomialPhaseEngine:
    """
    Computes multinomial distributions for phase classification
    and symptom count likelihoods across the menstrual timeline.
    """

    def __init__(self):
        self.phases = CYCLE_PHASES
        self.emissions = PHASE_SYMPTOM_EMISSIONS

    def compute_prior_phase_probabilities(
        self,
        current_cycle_day: int,
        cycle_length_estimate: float = 28.5,
        pcos: bool = False,
        stress_level: int = 3
    ) -> Dict[str, float]:
        """
        Computes the biological prior distribution P(Phase | Day, L)
        adjusting for follicular extension in PCOS and stress-induced ovulation delays.
        """
        L = max(cycle_length_estimate, 21.0)
        day = max(1, current_cycle_day)
        
        # In a standard cycle of length L:
        # Menstrual: days 1 to 5
        # Luteal phase is relatively fixed (typically 12-14 days before menses)
        # Follicular phase varies with cycle length and metabolic conditions
        luteal_duration = 14.0
        if pcos:
            luteal_duration = 12.0  # PCOS often has delayed ovulation with shorter/variable luteal phase
            
        ovulation_day = max(10.0, L - luteal_duration)
        if stress_level >= 7:
            ovulation_day += 2.0  # Acute stress delays LH surge and follicular maturation
            
        # Define Gaussian kernel density around physiological centers
        phase_centers = {
            "Menstrual": 2.5,
            "Follicular": 1.0 + (ovulation_day - 3.0) / 2.0,
            "Ovulatory": ovulation_day,
            "Luteal": ovulation_day + (luteal_duration / 2.0),
            "Late Luteal": L - 1.5
        }
        
        bandwidths = {
            "Menstrual": 1.8,
            "Follicular": max(2.5, (ovulation_day - 5.0) / 2.5),
            "Ovulatory": 1.8,
            "Luteal": max(2.5, luteal_duration / 3.0),
            "Late Luteal": 2.0
        }
        
        raw_scores = {}
        for phase, center in phase_centers.items():
            bw = bandwidths[phase]
            diff = (day - center) / bw
            raw_scores[phase] = math.exp(-0.5 * (diff ** 2))
            
        # If day exceeds predicted cycle length, shift mass to Late Luteal / Overdue
        if day > L:
            raw_scores["Late Luteal"] += 2.0 * math.exp(min(2.0, (day - L) / 3.0))
            
        total = sum(raw_scores.values())
        if total <= 1e-9:
            return {p: 1.0 / len(self.phases) for p in self.phases}
            
        return {p: round(score / total, 4) for p, score in raw_scores.items()}

    def multinomial_log_pmf(self, counts: List[int], probs: List[float]) -> float:
        """
        Evaluates the log probability mass function of a Multinomial distribution:
        log P(X = x) = log(n!) - sum(log(x_i!)) + sum(x_i * log(p_i))
        """
        n = sum(counts)
        if n == 0:
            return 0.0
            
        # Log factorials using lgamma
        log_n_fact = math.lgamma(n + 1)
        sum_log_xi_fact = sum(math.lgamma(x + 1) for x in counts)
        
        sum_xi_log_pi = 0.0
        for x, p in zip(counts, probs):
            if x > 0:
                p_safe = max(p, 1e-12)
                sum_xi_log_pi += x * math.log(p_safe)
                
        return log_n_fact - sum_log_xi_fact + sum_xi_log_pi

    def evaluate_symptom_evidence(
        self,
        symptom_counts: Dict[str, int],
        prior_probs: Dict[str, float]
    ) -> Dict[str, float]:
        r"""
        Applies Bayesian update with Multinomial likelihood:
        P(Phase | Symptoms) \propto P(Symptoms | Phase) * P(Phase)
        """
        active_counts = [symptom_counts.get(k, 0) for k in SYMPTOM_CATEGORIES]
        total_symptoms = sum(active_counts)
        
        if total_symptoms == 0:
            return prior_probs
            
        log_posteriors = {}
        for phase in self.phases:
            emission_dict = self.emissions[phase]
            # Normalize emission vector for the categories
            prob_vector = [emission_dict.get(k, 0.01) for k in SYMPTOM_CATEGORIES]
            sum_prob = sum(prob_vector)
            prob_vector = [p / sum_prob for p in prob_vector]
            
            # Compute multinomial log likelihood
            log_lik = self.multinomial_log_pmf(active_counts, prob_vector)
            prior_phase = max(prior_probs.get(phase, 1e-4), 1e-4)
            log_posteriors[phase] = log_lik + math.log(prior_phase)
            
        # Log-sum-exp trick for numerical stability
        max_log = max(log_posteriors.values())
        exp_posteriors = {p: math.exp(v - max_log) for p, v in log_posteriors.items()}
        total_exp = sum(exp_posteriors.values())
        
        return {p: round(v / total_exp, 4) for p, v in exp_posteriors.items()}

    def get_phase_analysis(
        self,
        current_cycle_day: int,
        cycle_length_estimate: float,
        symptoms: Optional[Dict[str, int]] = None,
        pcos: bool = False,
        stress_level: int = 3
    ) -> Dict:
        """
        Executes end-to-end multinomial phase analysis.
        Returns predicted phase, phase probability distribution, entropy, and hormonal state.
        """
        priors = self.compute_prior_phase_probabilities(
            current_cycle_day=current_cycle_day,
            cycle_length_estimate=cycle_length_estimate,
            pcos=pcos,
            stress_level=stress_level
        )
        
        if symptoms:
            posteriors = self.evaluate_symptom_evidence(symptoms, priors)
        else:
            posteriors = priors
            
        # Find dominant phase
        dominant_phase = max(posteriors.keys(), key=lambda k: posteriors[k])
        
        # Shannon Entropy as uncertainty metric: H(X) = -sum p log2(p)
        entropy = -sum(p * math.log2(p) for p in posteriors.values() if p > 0)
        
        hormonal_profiles = {
            "Menstrual": "Estrogen & Progesterone Nadir; Prostaglandin elevation",
            "Follicular": "Estrogen dominance, FSH stimulation, rising vitality",
            "Ovulatory": "LH surge peak, acute estrogen peak, basal body temperature nadir",
            "Luteal": "Progesterone peak (corpus luteum active), metabolic heat rise",
            "Late Luteal": "Progesterone withdrawal, PMS/PMDD neurosteroid fluctuation"
        }
        
        return {
            "current_cycle_day": current_cycle_day,
            "cycle_length_estimate": cycle_length_estimate,
            "dominant_phase": dominant_phase,
            "confidence": posteriors[dominant_phase],
            "distribution": posteriors,
            "entropy_bits": round(entropy, 3),
            "hormonal_profile": hormonal_profiles.get(dominant_phase, "Dynamic endocrine transition"),
            "is_fertile_window": dominant_phase in ["Follicular", "Ovulatory"] and (current_cycle_day >= cycle_length_estimate - 18 and current_cycle_day <= cycle_length_estimate - 12)
        }
