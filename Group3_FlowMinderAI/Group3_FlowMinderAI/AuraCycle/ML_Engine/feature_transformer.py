"""
feature_transformer.py - Biological Feature Transformer
Transforms 43 raw non-leaking user clinical & lifestyle inputs into derived biological ratios.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

FLOW_MAP = {"Light": 1, "Medium": 2, "Heavy": 3}
EXERCISE_MAP = {"Low": 1, "Moderate": 2, "High": 3}

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw features into derived physiological indicators.
    Computes:
    - bmi_disruption_flag: 1 if BMI < 18.5 or BMI > 30.0, else 0
    - acute_stress_velocity: Delta Stress scaled over baseline
    - cycle_variation_coeff: cycle_std_dev / average_cycle_length_5
    - circadian_sleep_deficit: max(0, 7.5 - sleep_hours)
    - metabolic_risk_index: Composite of BMI disruption and weight change velocity
    """
    df = df.copy()

    # Categorical encodings
    if "flow_intensity" in df.columns:
        df["flow_intensity_num"] = df["flow_intensity"].map(FLOW_MAP).fillna(2).astype(int)
    if "exercise_intensity" in df.columns:
        df["exercise_intensity_num"] = df["exercise_intensity"].map(EXERCISE_MAP).fillna(2).astype(int)

    # 1. BMI Disruption Flag
    bmi = df["bmi"] if "bmi" in df.columns else (df["weight_kg"] / ((df["height_cm"] / 100.0) ** 2))
    df["bmi_disruption_flag"] = ((bmi < 18.5) | (bmi > 30.0)).astype(int)

    # 2. Cycle Variation Coefficient
    avg_cycle = df["average_cycle_length_5"].replace(0, 28.5)
    df["cycle_variation_coeff"] = (df["cycle_std_dev"] / avg_cycle).round(4)

    # 3. Circadian Sleep Deficit
    df["circadian_sleep_deficit"] = np.maximum(0.0, 7.5 - df["sleep_hours"]).round(2)

    # 4. Acute Stress Velocity
    base_stress = (df["anxiety_level"] + df["work_study_load"]) / 2.0
    stress_delta = df["stress_level"] - base_stress
    df["acute_stress_velocity"] = (stress_delta / (base_stress + 1e-4)).round(4)

    # 5. Composite Endocrine / Metabolic Disruption Score
    lifestyle_disruptions = (
        df.get("recent_travel", 0) +
        df.get("major_life_change", 0) +
        df.get("diet_change_recently", 0) +
        df.get("meal_skipping", 0) +
        df.get("recent_illness", 0)
    )
    df["lifestyle_disruption_score"] = lifestyle_disruptions

    # 6. Pathology Index
    pathology = (
        df.get("pcos_diagnosis", 0) * 3 +
        df.get("thyroid_condition", 0) * 2 +
        df.get("endometriosis", 0) * 2 +
        df.get("fibroids", 0) * 1.5 +
        df.get("emergency_contraception_recent", 0) * 2.5
    )
    df["endocrine_pathology_score"] = pathology

    # 7. Short-term vs long-term cycle delta
    df["cycle_momentum_delta"] = (df["previous_cycle_length"] - df["average_cycle_length_3"]).round(2)

    return df

def get_feature_columns() -> List[str]:
    """Returns the standardized feature column ordering for LightGBM model ingestion."""
    return [
        "age", "height_cm", "weight_kg", "bmi",
        "previous_cycle_length", "cycle_length_2", "cycle_length_3", "cycle_length_4", "cycle_length_5",
        "average_cycle_length_3", "average_cycle_length_5", "cycle_std_dev",
        "previous_period_duration", "usual_period_duration",
        "spotting_before_period", "spotting_between_periods", "cycle_irregularity_history",
        "stress_level", "anxiety_level", "sleep_hours", "sleep_quality",
        "exercise_frequency_days", "daily_steps", "work_study_load",
        "recent_travel", "major_life_change", "weight_change_30d_kg", "calorie_change_per_day",
        "meal_skipping", "diet_change_recently", "recent_illness",
        "pcos_diagnosis", "thyroid_condition", "endometriosis", "fibroids",
        "hormonal_contraception", "emergency_contraception_recent", "medication_change_recently",
        "flow_intensity_num", "exercise_intensity_num",
        "bmi_disruption_flag", "cycle_variation_coeff", "circadian_sleep_deficit",
        "acute_stress_velocity", "lifestyle_disruption_score", "endocrine_pathology_score",
        "cycle_momentum_delta"
    ]

class FeatureTransformer:
    """Class wrapper for feature transformation and validation."""
    def __init__(self):
        self.feature_columns = get_feature_columns()

    def transform(self, raw_data: Dict[str, Any], nlp_modifiers: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        d = dict(raw_data)
        
        height = float(d.get("height_cm", 162.0))
        weight = float(d.get("weight_kg", 58.0))
        bmi = float(d.get("bmi", round(weight / ((height / 100.0) ** 2), 1)))
        
        prev_len = float(d.get("previous_cycle_length", 28.0))
        c2 = float(d.get("cycle_length_2", prev_len))
        c3 = float(d.get("cycle_length_3", prev_len))
        c4 = float(d.get("cycle_length_4", prev_len))
        c5 = float(d.get("cycle_length_5", prev_len))
        
        avg_3 = float(d.get("average_cycle_length_3", (prev_len + c2 + c3) / 3.0))
        avg_5 = float(d.get("average_cycle_length_5", (prev_len + c2 + c3 + c4 + c5) / 5.0))
        std_dev = float(d.get("cycle_std_dev", np.std([prev_len, c2, c3, c4, c5])))
        
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
            "medication_change_recently": int(d.get("medication_change_recently", 0)),
        }
        
        df_single = pd.DataFrame([row])
        df_engineered = engineer_features(df_single)
        
        for col in self.feature_columns:
            if col not in df_engineered.columns:
                df_engineered[col] = 0.0
                
        return df_engineered[self.feature_columns]
