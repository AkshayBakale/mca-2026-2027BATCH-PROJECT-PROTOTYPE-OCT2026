"""
dataset_generator.py - Synthetic Cohort Dataset Generator for AuraCycle AI
Generates 60,000 realistic clinical records based on empirical gynecological priors.
Spans a cohort over the last 1 year with authentic physiological correlations.
"""

import os
import datetime
import numpy as np
import pandas as pd

def generate_cohort(num_records: int = 60000, seed: int = 42, output_path: str = None) -> pd.DataFrame:
    """
    Generate a synthetic cohort of menstrual health records with realistic biological priors.
    """
    np.random.seed(seed)
    
    # 1. User population priors
    num_users = int(num_records / 7.5) # ~8000 users with 6-10 cycles each
    user_ids = [f"U{i:05d}" for i in range(1, num_users + 1)]
    
    # User biological archetypes
    user_ages = np.clip(np.random.normal(loc=28.5, scale=6.2, size=num_users).round(), 18, 45).astype(int)
    user_heights = np.clip(np.random.normal(loc=161.5, scale=6.5, size=num_users), 145.0, 182.0).round(1)
    
    # BMI distribution: ~10% underweight (<18.5), ~60% normal (18.5-24.9), ~20% overweight (25-29.9), ~10% obese (>30)
    bmi_choices = np.random.choice(["under", "normal", "over", "obese"], p=[0.08, 0.62, 0.20, 0.10], size=num_users)
    user_bmis = np.zeros(num_users)
    for i, cat in enumerate(bmi_choices):
        if cat == "under":
            user_bmis[i] = np.random.uniform(16.0, 18.4)
        elif cat == "normal":
            user_bmis[i] = np.random.uniform(18.5, 24.9)
        elif cat == "over":
            user_bmis[i] = np.random.uniform(25.0, 29.9)
        else:
            user_bmis[i] = np.random.uniform(30.0, 35.0)
    user_bmis = user_bmis.round(1)
    user_weights = (user_bmis * ((user_heights / 100.0) ** 2)).round(1)
    
    # Clinical conditions priors
    user_pcos = (np.random.rand(num_users) < 0.12).astype(int)
    user_thyroid = (np.random.rand(num_users) < 0.08).astype(int)
    user_endo = (np.random.rand(num_users) < 0.10).astype(int)
    user_fibroids = (np.random.rand(num_users) < 0.07).astype(int)
    user_contraception = (np.random.rand(num_users) < 0.15).astype(int)
    
    # Baseline individual cycle length & period duration
    user_base_cycle = np.random.normal(loc=28.8, scale=2.1, size=num_users)
    # PCOS / Thyroid / Endometriosis widen and lengthen baseline
    user_base_cycle += user_pcos * np.random.uniform(5.0, 9.0, size=num_users)
    user_base_cycle += user_thyroid * np.random.uniform(2.0, 4.5, size=num_users)
    user_base_cycle = np.clip(user_base_cycle, 21.0, 40.0)
    
    user_usual_duration = np.clip(np.random.normal(loc=4.8, scale=1.1, size=num_users).round(), 3, 7).astype(int)
    user_flow_pref = np.random.choice(["Light", "Medium", "Heavy"], p=[0.25, 0.55, 0.20], size=num_users)
    
    # Generate sequential cycles for users
    records = []
    end_anchor_date = datetime.date(2026, 9, 1)
    
    for u_idx, u_id in enumerate(user_ids):
        # 6 to 10 cycles per user
        n_cycles = np.random.randint(6, 11)
        age = user_ages[u_idx]
        height = user_heights[u_idx]
        base_weight = user_weights[u_idx]
        base_bmi = user_bmis[u_idx]
        base_cycle = user_base_cycle[u_idx]
        usual_dur = user_usual_duration[u_idx]
        base_flow = user_flow_pref[u_idx]
        has_pcos = user_pcos[u_idx]
        has_thyroid = user_thyroid[u_idx]
        has_endo = user_endo[u_idx]
        has_fibroids = user_fibroids[u_idx]
        has_contra = user_contraception[u_idx]
        
        # Start date approximately 1 year ago
        curr_date = end_anchor_date - datetime.timedelta(days=int(n_cycles * base_cycle + np.random.randint(5, 30)))
        
        # History queue of past cycles
        hist_cycles = [
            int(np.clip(np.random.normal(base_cycle, 1.2), 20, 45))
            for _ in range(5)
        ]
        
        for c_num in range(1, n_cycles + 1):
            # Lifestyle factors for this cycle
            stress = int(np.random.choice(range(1, 11), p=[0.05, 0.08, 0.12, 0.15, 0.20, 0.15, 0.10, 0.08, 0.05, 0.02]))
            anxiety = int(np.clip(stress + np.random.choice([-1, 0, 1, 2], p=[0.2, 0.5, 0.2, 0.1]), 1, 10))
            sleep_hours = float(np.clip(round(np.random.normal(7.2 - (stress * 0.12), 1.1), 1), 4.0, 10.0))
            sleep_quality = int(np.clip(round(np.random.normal(7.0 - (stress * 0.35), 1.8)), 1, 10))
            
            ex_freq = int(np.clip(np.random.poisson(3.2), 0, 7))
            ex_intensity = np.random.choice(["Low", "Moderate", "High"], p=[0.35, 0.45, 0.20])
            daily_steps = int(np.clip(np.random.normal(7500 + (ex_freq * 800), 2200), 2000, 14000))
            work_load = int(np.clip(stress + np.random.randint(-1, 2), 1, 10))
            
            travel = int(np.random.rand() < 0.09)
            life_change = int(np.random.rand() < 0.06)
            recent_illness = int(np.random.rand() < 0.07)
            
            # Weight & diet variability
            weight_change = float(round(np.random.normal(0.0, 0.8), 1))
            cur_weight = round(base_weight + weight_change, 1)
            cur_bmi = round(cur_weight / ((height / 100.0) ** 2), 1)
            calorie_change = int(np.random.normal(0, 320))
            meal_skipping = int(np.random.rand() < (0.15 + 0.02 * stress))
            diet_change = int(np.random.rand() < 0.12)
            
            emergency_contra = int(np.random.rand() < 0.03 and not has_contra)
            med_change = int(np.random.rand() < 0.05)
            
            spotting_before = int(np.random.rand() < (0.10 + 0.12 * has_endo + 0.08 * emergency_contra))
            spotting_between = int(np.random.rand() < (0.06 + 0.10 * has_pcos))
            irregularity_hist = int(has_pcos or has_thyroid or (np.std(hist_cycles) > 2.8))
            
            # Flow intensity
            if has_endo or has_fibroids:
                flow = np.random.choice(["Medium", "Heavy"], p=[0.35, 0.65])
            elif has_contra:
                flow = np.random.choice(["Light", "Medium"], p=[0.60, 0.40])
            else:
                flow = base_flow
                
            prev_duration = int(np.clip(usual_dur + np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2]), 3, 7))
            
            # Compute actual next cycle length based on physiological formulas
            physiological_delta = 0.0
            # Acute Stress & Anxiety delay ovulation (follicular phase lengthening)
            if stress >= 7:
                physiological_delta += (stress - 6) * 0.75 + (anxiety - 5) * 0.35
            # Sleep deficit (< 6.5 hours)
            if sleep_hours < 6.5:
                physiological_delta += (6.5 - sleep_hours) * 0.5
            # Illness / Infection
            if recent_illness:
                physiological_delta += np.random.uniform(1.5, 3.5)
            # Travel & Circadian disruption
            if travel:
                physiological_delta += np.random.uniform(0.8, 2.2)
            # Emergency contraception disruption
            if emergency_contra:
                physiological_delta += np.random.uniform(2.5, 5.0)
            # BMI extremes
            if cur_bmi < 18.5:
                physiological_delta += np.random.uniform(1.0, 3.0)
            elif cur_bmi > 30.0:
                physiological_delta += np.random.uniform(1.2, 3.5)
            # Medication changes
            if med_change:
                physiological_delta += np.random.choice([-2.0, 2.0, 3.0], p=[0.3, 0.4, 0.3])
            # Contraception regulation stabilizes toward 28
            if has_contra:
                base_target = 28.0
            else:
                base_target = base_cycle
                
            noise = np.random.normal(0, 0.9)
            next_length_calc = int(np.clip(round(base_target + physiological_delta + noise), 20, 48))
            
            # Rolling statistics from history
            prev_1 = hist_cycles[-1]
            prev_2 = hist_cycles[-2]
            prev_3 = hist_cycles[-3]
            prev_4 = hist_cycles[-4]
            prev_5 = hist_cycles[-5]
            avg_3 = round(float(np.mean(hist_cycles[-3:])), 2)
            avg_5 = round(float(np.mean(hist_cycles[-5:])), 2)
            std_dev_5 = round(float(np.std(hist_cycles[-5:])), 2)
            
            # Dates
            last_start_str = curr_date.strftime("%Y-%m-%d")
            next_start_date = curr_date + datetime.timedelta(days=next_length_calc)
            next_start_str = next_start_date.strftime("%Y-%m-%d")
            
            records.append({
                "user_id": u_id,
                "cycle_number": c_num,
                "age": age,
                "height_cm": height,
                "weight_kg": cur_weight,
                "bmi": cur_bmi,
                "last_period_start": last_start_str,
                "previous_cycle_length": prev_1,
                "cycle_length_2": prev_2,
                "cycle_length_3": prev_3,
                "cycle_length_4": prev_4,
                "cycle_length_5": prev_5,
                "average_cycle_length_3": avg_3,
                "average_cycle_length_5": avg_5,
                "cycle_std_dev": std_dev_5,
                "previous_period_duration": prev_duration,
                "usual_period_duration": usual_dur,
                "flow_intensity": flow,
                "spotting_before_period": spotting_before,
                "spotting_between_periods": spotting_between,
                "cycle_irregularity_history": irregularity_hist,
                "stress_level": stress,
                "anxiety_level": anxiety,
                "sleep_hours": sleep_hours,
                "sleep_quality": sleep_quality,
                "exercise_frequency_days": ex_freq,
                "exercise_intensity": ex_intensity,
                "daily_steps": daily_steps,
                "work_study_load": work_load,
                "recent_travel": travel,
                "major_life_change": life_change,
                "weight_change_30d_kg": weight_change,
                "calorie_change_per_day": calorie_change,
                "meal_skipping": meal_skipping,
                "diet_change_recently": diet_change,
                "recent_illness": recent_illness,
                "pcos_diagnosis": has_pcos,
                "thyroid_condition": has_thyroid,
                "endometriosis": has_endo,
                "fibroids": has_fibroids,
                "hormonal_contraception": has_contra,
                "emergency_contraception_recent": emergency_contra,
                "medication_change_recently": med_change,
                "actual_next_period_start": next_start_str,
                "actual_next_cycle_length": next_length_calc,
                "days_until_next_period": next_length_calc
            })
            
            # Step forward for user
            curr_date = next_start_date
            hist_cycles.pop(0)
            hist_cycles.append(next_length_calc)
            
            if len(records) >= num_records:
                break
        if len(records) >= num_records:
            break

    df = pd.DataFrame(records[:num_records])
    
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[Dataset Generator] Successfully generated {len(df)} records at {output_path}")
        
    return df

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(out_dir, exist_ok=True)
    csv_file = os.path.join(out_dir, "menstrual_cohort_60k.csv")
    generate_cohort(num_records=60000, output_path=csv_file)
