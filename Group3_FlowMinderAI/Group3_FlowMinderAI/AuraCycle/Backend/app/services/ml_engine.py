"""
ml_engine.py - Quantile Inference & Biological Feature Transformation Engine
Loads trained LightGBM models (q10, q50, q90), applies feature engineering,
guarantees quantile monotonicity, computes arrival probability densities (PDF),
and resolves current menstrual cycle phase.
"""

import os
import datetime
import math
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List

import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_root_candidates = [
    os.path.abspath(os.path.join(_current_dir, "..", "..", "..")),
    os.path.abspath(os.path.join(_current_dir, "..", "..")),
    os.path.abspath(os.path.join(_current_dir, ".."))
]
for _r in _root_candidates:
    for _sub in [_r, os.path.join(_r, "02_ml_engine"), os.path.join(_r, "01_data_pipeline"), os.path.join(_r, "Folder_2_ML_Engine"), os.path.join(_r, "Folder_1_Data_Pipeline")]:
        if os.path.exists(_sub) and _sub not in sys.path:
            sys.path.insert(0, _sub)

from app.config import Config

try:
    from importlib import import_module
    train_mod = import_module("02_ml_engine.train_quantile_lgbm")
    engineer_features = train_mod.engineer_features
    get_feature_columns = train_mod.get_feature_columns
    FLOW_MAP = train_mod.FLOW_MAP
    EXERCISE_MAP = train_mod.EXERCISE_MAP
except Exception:
    try:
        from ml_engine.train_quantile_lgbm import engineer_features, get_feature_columns, FLOW_MAP, EXERCISE_MAP
    except Exception:
        try:
            import train_quantile_lgbm as train_mod
            engineer_features = train_mod.engineer_features
            get_feature_columns = train_mod.get_feature_columns
            FLOW_MAP = train_mod.FLOW_MAP
            EXERCISE_MAP = train_mod.EXERCISE_MAP
        except Exception:
            from ml_pipeline.train_quantile_lgbm import engineer_features, get_feature_columns, FLOW_MAP, EXERCISE_MAP

class MLEngine:
    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or Config.MODEL_DIR
        self.q10_model = None
        self.q50_model = None
        self.q90_model = None
        self.metadata = None
        self.feature_columns = get_feature_columns()
        self._load_models()

    def _load_models(self):
        """Loads trained LightGBM models if present on disk."""
        q10_path = os.path.join(self.model_dir, "lgbm_quantile_10.joblib")
        q50_path = os.path.join(self.model_dir, "lgbm_quantile_50.joblib")
        q90_path = os.path.join(self.model_dir, "lgbm_quantile_90.joblib")
        meta_path = os.path.join(self.model_dir, "feature_metadata.joblib")

        try:
            if os.path.exists(q10_path) and os.path.exists(q50_path) and os.path.exists(q90_path):
                self.q10_model = joblib.load(q10_path)
                self.q50_model = joblib.load(q50_path)
                self.q90_model = joblib.load(q90_path)
                if os.path.exists(meta_path):
                    self.metadata = joblib.load(meta_path)
                    self.feature_columns = self.metadata.get("feature_columns", self.feature_columns)
                print("[MLEngine] Models successfully loaded into memory.")
            else:
                print("[MLEngine] Pre-trained models not found. Running in resilient clinical fallback mode.")
        except Exception as e:
            print(f"[MLEngine] Warning: Error loading model artifacts ({e}). Using resilient clinical fallback mode.")

    def transform_input(self, raw_data: Dict[str, Any], nlp_modifiers: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Validates, imputes, and transforms raw dictionary inputs into the 43+ feature vector.
        """
        d = dict(raw_data)
        
        # Calculate BMI if missing
        height = float(d.get("height_cm", 162.0))
        weight = float(d.get("weight_kg", 58.0))
        if "bmi" not in d or not d["bmi"]:
            bmi = round(weight / ((height / 100.0) ** 2), 1)
        else:
            bmi = float(d["bmi"])
            
        # Defaults for cycle history if partial
        prev_len = float(d.get("previous_cycle_length", 28.0))
        c2 = float(d.get("cycle_length_2", prev_len))
        c3 = float(d.get("cycle_length_3", prev_len))
        c4 = float(d.get("cycle_length_4", prev_len))
        c5 = float(d.get("cycle_length_5", prev_len))
        
        avg_3 = float(d.get("average_cycle_length_3", (prev_len + c2 + c3) / 3.0))
        avg_5 = float(d.get("average_cycle_length_5", (prev_len + c2 + c3 + c4 + c5) / 5.0))
        std_dev = float(d.get("cycle_std_dev", np.std([prev_len, c2, c3, c4, c5])))
        
        # Incorporate NLP modifiers if provided
        stress = float(d.get("stress_level", 5.0))
        spotting_flag = int(d.get("spotting_before_period", 0))
        
        if nlp_modifiers:
            stress += float(nlp_modifiers.get("stress_boost", 0.0))
            stress = min(10.0, max(1.0, stress))
            if nlp_modifiers.get("spotting_flag", 0) == 1:
                spotting_flag = 1
                
        row = {
            "age": int(d.get("age", 28)),
            "height_cm": height,
            "weight_kg": weight,
            "bmi": bmi,
            "previous_cycle_length": prev_len,
            "cycle_length_2": c2,
            "cycle_length_3": c3,
            "cycle_length_4": c4,
            "cycle_length_5": c5,
            "average_cycle_length_3": round(avg_3, 2),
            "average_cycle_length_5": round(avg_5, 2),
            "cycle_std_dev": round(std_dev, 2),
            "previous_period_duration": int(d.get("previous_period_duration", 5)),
            "usual_period_duration": int(d.get("usual_period_duration", 5)),
            "flow_intensity": d.get("flow_intensity", "Medium"),
            "spotting_before_period": spotting_flag,
            "spotting_between_periods": int(d.get("spotting_between_periods", 0)),
            "cycle_irregularity_history": int(d.get("cycle_irregularity_history", 0)),
            "stress_level": stress,
            "anxiety_level": float(d.get("anxiety_level", stress)),
            "sleep_hours": float(d.get("sleep_hours", 7.2)),
            "sleep_quality": float(d.get("sleep_quality", 7.0)),
            "exercise_frequency_days": int(d.get("exercise_frequency_days", 3)),
            "exercise_intensity": d.get("exercise_intensity", "Moderate"),
            "daily_steps": int(d.get("daily_steps", 7500)),
            "work_study_load": float(d.get("work_study_load", stress)),
            "recent_travel": int(d.get("recent_travel", 0)),
            "major_life_change": int(d.get("major_life_change", 0)),
            "weight_change_30d_kg": float(d.get("weight_change_30d_kg", 0.0)),
            "calorie_change_per_day": float(d.get("calorie_change_per_day", 0.0)),
            "meal_skipping": int(d.get("meal_skipping", 0)),
            "diet_change_recently": int(d.get("diet_change_recently", 0)),
            "recent_illness": int(d.get("recent_illness", 0)),
            "pcos_diagnosis": int(d.get("pcos_diagnosis", 0)),
            "thyroid_condition": int(d.get("thyroid_condition", 0)),
            "endometriosis": int(d.get("endometriosis", 0)),
            "fibroids": int(d.get("fibroids", 0)),
            "hormonal_contraception": int(d.get("hormonal_contraception", 0)),
            "emergency_contraception_recent": int(d.get("emergency_contraception_recent", 0)),
            "medication_change_recently": int(d.get("medication_change_recently", 0))
        }
        
        df_single = pd.DataFrame([row])
        df_engineered = engineer_features(df_single)
        return df_engineered[self.feature_columns]

    def _fallback_predict(self, features_df: pd.DataFrame, nlp_shift: float = 0.0) -> Tuple[float, float, float]:
        """
        Clinical biological heuristic fallback when ML models are not trained yet.
        """
        row = features_df.iloc[0]
        base = float(row["average_cycle_length_5"])
        if base < 20 or base > 45:
            base = float(row["previous_cycle_length"])
            
        shift = 0.0
        if row["stress_level"] >= 7:
            shift += (row["stress_level"] - 6) * 0.7
        if row["recent_illness"] == 1:
            shift += 2.0
        if row["emergency_contraception_recent"] == 1:
            shift += 3.5
        if row["bmi_disruption_flag"] == 1:
            shift += 1.5
        if row["pcos_diagnosis"] == 1:
            shift += 2.5
        if row["hormonal_contraception"] == 1:
            base = 28.0
            shift = 0.0
            
        shift += nlp_shift
        
        q50 = round(base + shift, 1)
        q10 = round(q50 - 2.1, 1)
        q90 = round(q50 + 2.8, 1)
        return q10, q50, q90

    def predict(self, raw_data: Dict[str, Any], nlp_modifiers: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Full inference pipeline:
        1. Feature transformation
        2. Quantile regression at tau = 0.10, 0.50, 0.90
        3. Monotonicity calibration & sorting
        4. Date conversion & arrival probability density calculation
        5. Current cycle phase identification
        """
        nlp_shift = 0.0
        if nlp_modifiers:
            nlp_shift = float(nlp_modifiers.get("onset_shift_days", 0.0))
            
        X = self.transform_input(raw_data, nlp_modifiers=nlp_modifiers)
        
        if self.q10_model is not None and self.q50_model is not None and self.q90_model is not None:
            raw_q10 = float(self.q10_model.predict(X)[0]) + nlp_shift
            raw_q50 = float(self.q50_model.predict(X)[0]) + nlp_shift
            raw_q90 = float(self.q90_model.predict(X)[0]) + nlp_shift
        else:
            raw_q10, raw_q50, raw_q90 = self._fallback_predict(X, nlp_shift=nlp_shift)
            
        # Enforce strict quantile monotonicity: q10 <= q50 <= q90
        sorted_quantiles = np.sort([raw_q10, raw_q50, raw_q90])
        q10 = round(float(sorted_quantiles[0]), 1)
        q50 = round(float(sorted_quantiles[1]), 1)
        q90 = round(float(sorted_quantiles[2]), 1)
        
        # Uncertainty bound & confidence metric
        uncertainty_window_days = round(q90 - q10, 1)
        confidence_score = round(max(0.60, min(0.98, 1.0 - (uncertainty_window_days / 20.0))), 2)
        
        # Date resolution
        last_period_str = raw_data.get("last_period_start", datetime.date.today().strftime("%Y-%m-%d"))
        try:
            last_date = datetime.datetime.strptime(last_period_str, "%Y-%m-%d").date()
        except ValueError:
            last_date = datetime.date.today()
            
        date_q10 = last_date + datetime.timedelta(days=int(round(q10)))
        date_q50 = last_date + datetime.timedelta(days=int(round(q50)))
        date_q90 = last_date + datetime.timedelta(days=int(round(q90)))
        
        today = datetime.date.today()
        days_since_start = (today - last_date).days
        current_cycle_day = max(1, days_since_start + 1)
        
        # Menstrual cycle phase resolution
        period_dur = int(raw_data.get("usual_period_duration", 5))
        ovulation_day = max(period_dur + 2, int(round(q50 - 14)))
        
        if current_cycle_day <= period_dur:
            phase = "Menstrual Phase"
            phase_desc = "Uterine lining shedding. Estrogen and progesterone are at baseline."
            phase_color = "#f43f5e" # Neon Rose
        elif current_cycle_day < ovulation_day - 1:
            phase = "Follicular Phase"
            phase_desc = "FSH stimulates follicle maturation. Estrogen rising, boosting energy."
            phase_color = "#8b5cf6" # Soft Violet
        elif current_cycle_day <= ovulation_day + 1:
            phase = "Ovulation Window"
            phase_desc = "LH surge triggers mature egg release. Peak fertility window."
            phase_color = "#10b981" # Emerald
        else:
            phase = "Luteal Phase"
            phase_desc = "Corpus luteum secretes progesterone. Preparing uterine lining for arrival."
            phase_color = "#f59e0b" # Warm Amber
            
        days_until_median = max(0, int(round(q50)) - days_since_start)
        
        # Arrival Probability Density Function (PDF) curve for Chart.js
        pdf_curve = self._compute_arrival_pdf(q10, q50, q90, last_date)
        
        return {
            "prediction": {
                "q10_cycle_length": q10,
                "q50_cycle_length": q50,
                "q90_cycle_length": q90,
                "date_q10_earliest": date_q10.strftime("%Y-%m-%d"),
                "date_q50_median": date_q50.strftime("%Y-%m-%d"),
                "date_q90_latest": date_q90.strftime("%Y-%m-%d"),
                "uncertainty_window_days": uncertainty_window_days,
                "confidence_score": confidence_score,
                "days_until_next_period": days_until_median
            },
            "phase_tracking": {
                "current_cycle_day": current_cycle_day,
                "phase_name": phase,
                "phase_description": phase_desc,
                "phase_color": phase_color,
                "estimated_ovulation_day": ovulation_day
            },
            "pdf_distribution": pdf_curve,
            "engineered_features": {
                "bmi": float(X["bmi"].iloc[0]),
                "bmi_disruption_flag": int(X["bmi_disruption_flag"].iloc[0]),
                "acute_stress_velocity": float(X["acute_stress_velocity"].iloc[0]),
                "cycle_variation_coeff": float(X["cycle_variation_coeff"].iloc[0]),
                "circadian_sleep_deficit": float(X["circadian_sleep_deficit"].iloc[0])
            }
        }

    def _compute_arrival_pdf(self, q10: float, q50: float, q90: float, last_date: datetime.date) -> Dict[str, Any]:
        """
        Constructs discretized Gaussian arrival density function centered around q50
        with scale sigma estimated from IQR / quantile spread (q90 - q10) / 2.56.
        """
        spread = max(q90 - q10, 2.0)
        sigma = spread / 2.56  # 80% coverage in normal distribution is 2 * 1.28 * sigma = 2.56 sigma
        
        start_day = int(math.floor(q10 - spread * 0.6))
        end_day = int(math.ceil(q90 + spread * 0.6))
        
        labels = []
        densities = []
        
        for d in range(start_day, end_day + 1):
            cur_d = last_date + datetime.timedelta(days=d)
            labels.append(cur_d.strftime("%b %d"))
            
            # Normal density
            p = (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((d - q50) / sigma) ** 2)
            densities.append(round(p, 5))
            
        return {
            "days": list(range(start_day, end_day + 1)),
            "labels": labels,
            "probabilities": densities,
            "q10_day": q10,
            "q50_day": q50,
            "q90_day": q90
        }

# Global singleton
ml_engine = MLEngine()
