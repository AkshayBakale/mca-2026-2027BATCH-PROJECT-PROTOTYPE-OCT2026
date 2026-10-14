"""
evaluate_metrics.py - Metric Evaluations for Quantile Regression
Calculates Pinball Loss, MAE, RMSE, and PICP (Prediction Interval Coverage Probability).
"""

import numpy as np
from typing import Dict

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """
    Computes Pinball (Quantile) loss for quantile alpha:
    L(y, q) = max(alpha * (y - q), (1 - alpha) * (q - y))
    """
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))

def prediction_interval_coverage_probability(y_true: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray) -> float:
    """
    Computes PICP: Percentage of actual values falling inside [lower_bound, upper_bound].
    For an 80% interval (tau=0.10 to tau=0.90), nominal target is 0.80.
    """
    covered = (y_true >= lower_bound) & (y_true <= upper_bound)
    return float(np.mean(covered))

def mean_prediction_interval_width(lower_bound: np.ndarray, upper_bound: np.ndarray) -> float:
    """Computes average width of prediction interval: E[upper - lower]."""
    return float(np.mean(upper_bound - lower_bound))

def evaluate_quantile_predictions(
    y_true: np.ndarray,
    q10_pred: np.ndarray,
    q50_pred: np.ndarray,
    q90_pred: np.ndarray
) -> Dict[str, float]:
    """
    Comprehensive evaluation of quantile regression calibration and accuracy.
    """
    y_true = np.asarray(y_true)
    q10 = np.asarray(q10_pred)
    q50 = np.asarray(q50_pred)
    q90 = np.asarray(q90_pred)
    
    mae_median = float(np.mean(np.abs(y_true - q50)))
    rmse_median = float(np.sqrt(np.mean((y_true - q50) ** 2)))
    
    loss_10 = pinball_loss(y_true, q10, alpha=0.10)
    loss_50 = pinball_loss(y_true, q50, alpha=0.50)
    loss_90 = pinball_loss(y_true, q90, alpha=0.90)
    mean_pinball = (loss_10 + loss_50 + loss_90) / 3.0
    
    picp_80 = prediction_interval_coverage_probability(y_true, q10, q90)
    mpiw = mean_prediction_interval_width(q10, q90)
    
    # Proportion below q10 (nominal 10%) and above q90 (nominal 10%)
    fraction_below_q10 = float(np.mean(y_true < q10))
    fraction_above_q90 = float(np.mean(y_true > q90))
    
    return {
        "mae_median": round(mae_median, 3),
        "rmse_median": round(rmse_median, 3),
        "pinball_loss_q10": round(loss_10, 4),
        "pinball_loss_q50": round(loss_50, 4),
        "pinball_loss_q90": round(loss_90, 4),
        "mean_pinball_loss": round(mean_pinball, 4),
        "picp_80_percent": round(picp_80, 4),
        "mean_interval_width_days": round(mpiw, 2),
        "fraction_below_q10": round(fraction_below_q10, 4),
        "fraction_above_q90": round(fraction_above_q90, 4)
    }

def print_calibration_table(metrics: Dict[str, float]):
    """Prints a formatted clinical calibration report."""
    print("==========================================================")
    print("      AURACYCLE AI - QUANTILE CALIBRATION REPORT          ")
    print("==========================================================")
    print(f" Median MAE                     : {metrics['mae_median']:.2f} days")
    print(f" Median RMSE                    : {metrics['rmse_median']:.2f} days")
    print(f" Pinball Loss (tau=0.10)        : {metrics['pinball_loss_q10']:.4f}")
    print(f" Pinball Loss (tau=0.50)        : {metrics['pinball_loss_q50']:.4f}")
    print(f" Pinball Loss (tau=0.90)        : {metrics['pinball_loss_q90']:.4f}")
    print(f" Mean Pinball Loss              : {metrics['mean_pinball_loss']:.4f}")
    print(f" 80% Prediction Coverage (PICP) : {metrics['picp_80_percent'] * 100:.2f}% (Nominal: 80.00%)")
    print(f" Mean Interval Width            : {metrics['mean_interval_width_days']:.2f} days")
    print(f" Fraction Below 10th Percentile : {metrics['fraction_below_q10'] * 100:.2f}% (Target: ~10.0%)")
    print(f" Fraction Above 90th Percentile : {metrics['fraction_above_q90'] * 100:.2f}% (Target: ~10.0%)")
    print("==========================================================")
