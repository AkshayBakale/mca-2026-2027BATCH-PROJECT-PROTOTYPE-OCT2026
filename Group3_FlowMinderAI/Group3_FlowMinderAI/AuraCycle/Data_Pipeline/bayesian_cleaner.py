"""
bayesian_cleaner.py - Latent Cycle De-Aliasing & Tracking Artifact Filter
Implements Bayesian inference P(K|G) to split prolonged gaps without variance inflation,
and prunes tracking noise/artifacts (intervals < 15 days).
"""

import math
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

DEFAULT_PRIORS = {
    1: 0.88,
    2: 0.08,
    3: 0.025,
    4: 0.010,
    5: 0.005
}

@dataclass
class LatentCycleResult:
    original_gap: float
    is_artifact: bool
    split_triggered: bool
    optimal_k: int
    sub_cycle_lengths: List[float]
    posteriors: Dict[int, float]
    imputed_dates: List[str]
    message: str

def gaussian_pdf(x: float, mean: float, var: float) -> float:
    """Computes Gaussian probability density N(x | mean, var)."""
    if var <= 1e-6:
        var = 1e-6
    coeff = 1.0 / math.sqrt(2.0 * math.pi * var)
    exponent = -((x - mean) ** 2) / (2.0 * var)
    return coeff * math.exp(exponent)

def compute_gap_posterior(
    gap_days: float,
    mean_cycle: float = 28.5,
    std_cycle: float = 2.5,
    max_k: int = 5,
    priors: Optional[Dict[int, float]] = None
) -> Dict[int, float]:
    """
    Computes posterior probability P(K | G) for K in [1, max_k]:
    P(K | G) = [N(G | K*mu, K*sigma^2) * P(K)] / sum_k [...]
    """
    if priors is None:
        priors = DEFAULT_PRIORS
        
    var_base = max(std_cycle ** 2, 0.5)
    likelihoods = {}
    
    for k in range(1, max_k + 1):
        prior_k = priors.get(k, 0.001 * (0.5 ** (k - 5)))
        mean_k = k * mean_cycle
        var_k = k * var_base  # Variance of sum of K independent cycles is K*sigma^2
        likelihood_k = gaussian_pdf(gap_days, mean_k, var_k)
        likelihoods[k] = likelihood_k * prior_k
        
    total_marginal = sum(likelihoods.values())
    
    if total_marginal <= 1e-15:
        estimated_k = max(1, min(max_k, int(round(gap_days / max(mean_cycle, 1.0)))))
        return {k: (1.0 if k == estimated_k else 0.0) for k in range(1, max_k + 1)}
        
    posteriors = {k: val / total_marginal for k, val in likelihoods.items()}
    return posteriors

def is_tracking_artifact(gap_days: float, min_valid_days: float = 15.0) -> bool:
    """Checks if a logged interval represents a tracking artifact (< 15 days)."""
    return gap_days < min_valid_days

def dealias_cycle_gap(
    gap_days: float,
    mean_cycle: float = 28.5,
    std_cycle: float = 2.5,
    last_start_date: Optional[Union[str, datetime.date]] = None,
    split_threshold_ratio: float = 1.8,
    max_k: int = 5
) -> Dict:
    """
    Evaluates a gap G. If G > 1.8 * mean_cycle, splits into optimal K latent sub-cycles.
    """
    if is_tracking_artifact(gap_days):
        return {
            "original_gap": gap_days,
            "is_artifact": True,
            "split_triggered": False,
            "optimal_k": 1,
            "sub_cycle_lengths": [gap_days],
            "posteriors": {1: 1.0},
            "imputed_dates": [],
            "message": "Tracking artifact: interval < 15 days is mid-cycle spotting or erroneous entry."
        }
        
    if gap_days <= split_threshold_ratio * mean_cycle:
        return {
            "original_gap": gap_days,
            "is_artifact": False,
            "split_triggered": False,
            "optimal_k": 1,
            "sub_cycle_lengths": [gap_days],
            "posteriors": {1: 1.0},
            "imputed_dates": [],
            "message": "Normal cycle interval within historical variation threshold."
        }
        
    posteriors = compute_gap_posterior(
        gap_days=gap_days,
        mean_cycle=mean_cycle,
        std_cycle=std_cycle,
        max_k=max_k
    )
    
    optimal_k = max(posteriors.keys(), key=lambda k: posteriors[k])
    sub_length = round(gap_days / optimal_k, 2)
    sub_cycle_lengths = [sub_length] * optimal_k
    
    imputed_dates = []
    if last_start_date:
        if isinstance(last_start_date, str):
            cur_date = datetime.datetime.strptime(last_start_date, "%Y-%m-%d").date()
        else:
            cur_date = last_start_date
            
        for step in range(1, optimal_k):
            sub_date = cur_date + datetime.timedelta(days=int(round(step * sub_length)))
            imputed_dates.append(sub_date.strftime("%Y-%m-%d"))
            
    return {
        "original_gap": gap_days,
        "is_artifact": False,
        "split_triggered": True,
        "optimal_k": optimal_k,
        "sub_cycle_lengths": sub_cycle_lengths,
        "posteriors": {k: round(v, 4) for k, v in posteriors.items()},
        "imputed_dates": imputed_dates,
        "message": f"Bayesian de-aliasing identified {optimal_k} latent cycles (Posterior: {posteriors[optimal_k]:.1%})."
    }

def clean_cycle_time_series(
    raw_lengths: List[float],
    mean_cycle: Optional[float] = None,
    std_cycle: Optional[float] = None
) -> Tuple[List[float], Dict]:
    """Cleans an entire time series of logged cycle lengths."""
    if not raw_lengths:
        return [], {"num_original": 0, "num_cleaned": 0, "artifacts_removed": 0, "splits_performed": 0}
        
    valid_lengths = [x for x in raw_lengths if x >= 15.0]
    artifacts_removed = len(raw_lengths) - len(valid_lengths)
    
    if not valid_lengths:
        return [], {"num_original": len(raw_lengths), "num_cleaned": 0, "artifacts_removed": artifacts_removed, "splits_performed": 0}
        
    if mean_cycle is None:
        mean_cycle = float(np.median(valid_lengths))
        if mean_cycle < 20.0 or mean_cycle > 45.0:
            mean_cycle = 28.5
            
    if std_cycle is None:
        std_cycle = max(float(np.std(valid_lengths)), 2.0)
        
    cleaned_lengths = []
    splits_performed = 0
    
    for length in valid_lengths:
        result = dealias_cycle_gap(length, mean_cycle, std_cycle)
        if result["split_triggered"]:
            cleaned_lengths.extend(result["sub_cycle_lengths"])
            splits_performed += 1
        else:
            cleaned_lengths.append(length)
            
    summary = {
        "num_original": len(raw_lengths),
        "num_cleaned": len(cleaned_lengths),
        "artifacts_removed": artifacts_removed,
        "splits_performed": splits_performed,
        "cleaned_mean": round(float(np.mean(cleaned_lengths)), 2) if cleaned_lengths else 0.0,
        "cleaned_std": round(float(np.std(cleaned_lengths)), 2) if cleaned_lengths else 0.0
    }
    
    return cleaned_lengths, summary

class BayesianCleaner:
    """Object-oriented interface for Bayesian cycle gap de-aliasing."""
    def __init__(self, default_mean: float = 28.5, default_std: float = 2.5, split_ratio: float = 1.8):
        self.default_mean = default_mean
        self.default_std = default_std
        self.split_ratio = split_ratio

    def dealias(self, gap_days: float, mean_cycle: Optional[float] = None, std_cycle: Optional[float] = None,
                last_start_date: Optional[str] = None) -> LatentCycleResult:
        mean_val = mean_cycle if mean_cycle is not None else self.default_mean
        std_val = std_cycle if std_cycle is not None else self.default_std
        res = dealias_cycle_gap(gap_days, mean_val, std_val, last_start_date, self.split_ratio)
        return LatentCycleResult(
            original_gap=res["original_gap"],
            is_artifact=res["is_artifact"],
            split_triggered=res["split_triggered"],
            optimal_k=res["optimal_k"],
            sub_cycle_lengths=res["sub_cycle_lengths"],
            posteriors=res["posteriors"],
            imputed_dates=res["imputed_dates"],
            message=res["message"]
        )

    def clean_series(self, raw_lengths: List[float], mean_cycle: Optional[float] = None, std_cycle: Optional[float] = None):
        return clean_cycle_time_series(raw_lengths, mean_cycle, std_cycle)
