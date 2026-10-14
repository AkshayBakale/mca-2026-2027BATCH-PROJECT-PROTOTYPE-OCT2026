"""
train_quantile_lgbm.py - GroupKFold Quantile Regression Training Script
Trains three distinct LightGBM quantile models (tau = 0.10, 0.50, 0.90)
using 5-Fold GroupKFold cross-validation grouped by user_id.
Enforces monotonic calibration and outputs production model artifacts.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

try:
    from importlib import import_module
    data_mod = import_module("01_data_pipeline.dataset_generator")
    generate_cohort = data_mod.generate_cohort
    eval_mod = import_module("02_ml_engine.evaluate_metrics")
    evaluate_quantile_predictions = eval_mod.evaluate_quantile_predictions
    print_calibration_table = eval_mod.print_calibration_table
except Exception:
    try:
        from data_pipeline.dataset_generator import generate_cohort
        from ml_engine.evaluate_metrics import evaluate_quantile_predictions, print_calibration_table
    except Exception:
        from ml_pipeline.dataset_generator import generate_cohort
        from ml_pipeline.evaluate_metrics import evaluate_quantile_predictions, print_calibration_table

FLOW_MAP = {"Light": 1, "Medium": 2, "Heavy": 3}
EXERCISE_MAP = {"Low": 1, "Moderate": 2, "High": 3}

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes biological derived ratios and maps categorical features into numerical vectors.
    """
    data = df.copy()
    
    # 1. Categorical mappings
    if "flow_intensity" in data.columns and data["flow_intensity"].dtype == object:
        data["flow_intensity"] = data["flow_intensity"].map(FLOW_MAP).fillna(2).astype(int)
    if "exercise_intensity" in data.columns and data["exercise_intensity"].dtype == object:
        data["exercise_intensity"] = data["exercise_intensity"].map(EXERCISE_MAP).fillna(2).astype(int)
        
    # 2. Derived biological ratios
    # BMI disruption flag: 1 if BMI < 18.5 or BMI > 30.0, else 0
    data["bmi_disruption_flag"] = (
        (data["bmi"] < 18.5) | (data["bmi"] > 30.0)
    ).astype(int)
    
    # Acute stress velocity: scaled delta over baseline 5.0
    data["acute_stress_velocity"] = (
        (data["stress_level"] - 5.0) / 5.0 + 0.5 * (data["work_study_load"] - 5.0) / 5.0
    ).round(4)
    
    # Cycle variation coefficient: cycle_std_dev / average_cycle_length_5
    data["cycle_variation_coeff"] = (
        data["cycle_std_dev"] / (data["average_cycle_length_5"] + 1e-6)
    ).round(4)
    
    # Circadian sleep deficit: max(0, 7.5 - sleep_hours)
    data["circadian_sleep_deficit"] = np.maximum(
        0.0, 7.5 - data["sleep_hours"]
    ).round(2)
    
    # Metabolic load indicator
    data["metabolic_load_index"] = (
        data["meal_skipping"] * 1.5 + (np.abs(data["calorie_change_per_day"]) > 250).astype(int)
    ).astype(float)
    
    return data

def get_feature_columns() -> list:
    """List of all 43 non-leaking features used for inference."""
    return [
        "age",
        "height_cm",
        "weight_kg",
        "bmi",
        "previous_cycle_length",
        "cycle_length_2",
        "cycle_length_3",
        "cycle_length_4",
        "cycle_length_5",
        "average_cycle_length_3",
        "average_cycle_length_5",
        "cycle_std_dev",
        "previous_period_duration",
        "usual_period_duration",
        "flow_intensity",
        "spotting_before_period",
        "spotting_between_periods",
        "cycle_irregularity_history",
        "stress_level",
        "anxiety_level",
        "sleep_hours",
        "sleep_quality",
        "exercise_frequency_days",
        "exercise_intensity",
        "daily_steps",
        "work_study_load",
        "recent_travel",
        "major_life_change",
        "weight_change_30d_kg",
        "calorie_change_per_day",
        "meal_skipping",
        "diet_change_recently",
        "recent_illness",
        "pcos_diagnosis",
        "thyroid_condition",
        "endometriosis",
        "fibroids",
        "hormonal_contraception",
        "emergency_contraception_recent",
        "medication_change_recently",
        # Engineered ratios
        "bmi_disruption_flag",
        "acute_stress_velocity",
        "cycle_variation_coeff",
        "circadian_sleep_deficit",
        "metabolic_load_index"
    ]

def train_quantile_models(
    csv_path: str,
    models_dir: str,
    n_splits: int = 5,
    quantiles: list = [0.10, 0.50, 0.90]
) -> dict:
    """
    Executes GroupKFold cross-validation, computes metrics, and trains production models.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"[Trainer] Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"[Trainer] Loaded {len(df)} records across {df['user_id'].nunique()} unique cohorts.")
    
    # Feature Engineering
    df_feat = engineer_features(df)
    feature_cols = get_feature_columns()
    target_col = "actual_next_cycle_length"
    group_col = "user_id"
    
    X = df_feat[feature_cols].copy()
    y = df_feat[target_col].values
    groups = df_feat[group_col].values
    
    # 1. GroupKFold Cross-Validation
    print(f"\n[Trainer] Running {n_splits}-Fold GroupKFold Cross-Validation (Grouping on {group_col})...")
    gkf = GroupKFold(n_splits=n_splits)
    
    oof_preds = {alpha: np.zeros(len(df)) for alpha in quantiles}
    
    lgb_params_base = {
        "boosting_type": "gbdt",
        "objective": "quantile",
        "n_estimators": 180,
        "learning_rate": 0.06,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 25,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1
    }
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        for alpha in quantiles:
            model = lgb.LGBMRegressor(**lgb_params_base, alpha=alpha)
            model.fit(X_tr, y_tr)
            oof_preds[alpha][val_idx] = model.predict(X_val)
            
        print(f"  [Fold {fold}/{n_splits}] Complete.")
        
    # Enforce Monotonicity on OOF: q10 <= q50 <= q90
    oof_stacked = np.column_stack([oof_preds[0.10], oof_preds[0.50], oof_preds[0.90]])
    oof_monotonic = np.sort(oof_stacked, axis=1)
    q10_oof = oof_monotonic[:, 0]
    q50_oof = oof_monotonic[:, 1]
    q90_oof = oof_monotonic[:, 2]
    
    metrics = evaluate_quantile_predictions(y, q10_oof, q50_oof, q90_oof)
    print_calibration_table(metrics)
    
    # 2. Train final production models on full cohort
    print("\n[Trainer] Training final production models on all records...")
    prod_models = {}
    for alpha in quantiles:
        model = lgb.LGBMRegressor(**lgb_params_base, alpha=alpha)
        model.fit(X, y)
        prod_models[alpha] = model
        
        # Save model artifact
        filename = f"lgbm_quantile_{int(alpha * 100):02d}.joblib"
        filepath = os.path.join(models_dir, filename)
        joblib.dump(model, filepath)
        print(f"  Saved {alpha*100:.0f}th percentile model -> {filepath}")
        
    # Save feature metadata
    metadata = {
        "feature_columns": feature_cols,
        "quantiles": quantiles,
        "flow_map": FLOW_MAP,
        "exercise_map": EXERCISE_MAP,
        "calibration_metrics": metrics
    }
    meta_path = os.path.join(models_dir, "feature_metadata.joblib")
    joblib.dump(metadata, meta_path)
    print(f"  Saved metadata -> {meta_path}")
    
    return {"metrics": metrics, "models": prod_models}

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    csv_file = os.path.join(data_dir, "menstrual_cohort_60k.csv")
    if not os.path.exists(csv_file):
        print(f"[Dataset] {csv_file} not found. Generating 60,000 synthetic records...")
        generate_cohort(num_records=60000, output_path=csv_file)
        
    train_quantile_models(csv_path=csv_file, models_dir=models_dir)
