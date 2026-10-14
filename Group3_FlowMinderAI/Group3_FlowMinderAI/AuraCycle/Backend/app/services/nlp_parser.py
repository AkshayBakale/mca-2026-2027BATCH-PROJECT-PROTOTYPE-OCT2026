"""
nlp_parser.py - NLP Free-Text Symptom Analyzer for AuraCycle AI
Extracts gynecological symptom entities, semantic severity ratings (1-5),
VADER sentiment/polarity, and computes physiological feature vector modifiers.
"""

import re
from typing import Dict, List, Any
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# Gynecological Entity Taxonomy with regular expression patterns
SYMPTOM_TAXONOMY = {
    "dysmenorrhea": [
        r"\bcramp(?:s|ing)?\b",
        r"\bpelvic\s+pain\b",
        r"\blower\s+abdominal\s+pain\b",
        r"\buterine\s+pain\b",
        r"\bperiod\s+pain\b",
        r"\bdysmenorrhea\b"
    ],
    "bloating": [
        r"\bbloat(?:ed|ing)?\b",
        r"\bwater\s+retention\b",
        r"\bpuffy\s+abdomen\b",
        r"\babdominal\s+distension\b"
    ],
    "nausea": [
        r"\bnause(?:a|ous|ated)?\b",
        r"\bqueasy\b",
        r"\bmorning\s+sickness\b",
        r"\bupset\s+stomach\b",
        r"\bvomi(?:t|ting)?\b"
    ],
    "spotting": [
        r"\bspot(?:ting)?\b",
        r"\blight\s+bleeding\b",
        r"\bbrown\s+discharge\b",
        r"\bmid-cycle\s+bleeding\b",
        r"\bbreakthrough\s+bleeding\b"
    ],
    "migraine": [
        r"\bmigraine(?:s)?\b",
        r"\bheadache(?:s)?\b",
        r"\bthrobbing\s+temple\b",
        r"\bhead\s+pressure\b"
    ],
    "fatigue": [
        r"\bfatigue(?:d)?\b",
        r"\bexhaust(?:ed|ion)?\b",
        r"\bletharg(?:ic|y)?\b",
        r"\btired(?:ness)?\b",
        r"\blow\s+energy\b",
        r"\bno\s+energy\b"
    ],
    "breast_tenderness": [
        r"\bbreast\s+tenderness\b",
        r"\bsore\s+breasts\b",
        r"\bbreast\s+pain\b",
        r"\btender\s+breasts\b"
    ],
    "mood_swings": [
        r"\bmood\s+swing(?:s)?\b",
        r"\birritab(?:le|ility)?\b",
        r"\banxi(?:ous|ety)?\b",
        r"\bcrying\s+spells?\b",
        r"\bweepy\b",
        r"\bdepressed\b"
    ],
    "insomnia": [
        r"\binsomnia\b",
        r"\bcan't\s+sleep\b",
        r"\bcannot\s+sleep\b",
        r"\bsleepless\b",
        r"\bwaking\s+up\s+at\s+night\b",
        r"\brestless\s+sleep\b"
    ],
    "ovulation_pain": [
        r"\bovulation\s+pain\b",
        r"\bmittelschmerz\b",
        r"\bone-sided\s+pelvic\s+pain\b",
        r"\bovary\s+twinge\b"
    ]
}

# Intensity modifiers for 1-5 severity grading
SEVERITY_MODIFIERS = {
    5: [r"\bunbearable\b", r"\bexcruciating\b", r"\bdebilitating\b", r"\bintense\b", r"\bsevere\b", r"\bagony\b", r"\bcan't\s+move\b"],
    4: [r"\bsharp\b", r"\bheavy\b", r"\bstrong\b", r"\bsignificant\b", r"\bconstant\b", r"\bthrobbing\b", r"\bbad\b"],
    3: [r"\bmoderate\b", r"\bnoticeable\b", r"\bpersistent\b", r"\bannoying\b"],
    2: [r"\bmild\b", r"\bslight\b", r"\blittle\b", r"\bdull\b", r"\boccasional\b", r"\bmanageable\b"],
    1: [r"\bvery\s+mild\b", r"\bminimal\b", r"\bfaint\b", r"\bbarely\b", r"\btiny\s+bit\b"]
}

def analyze_journal(text: str) -> Dict[str, Any]:
    """
    Parses unstructured free-text journal entry.
    Extracts entities, determines 1-5 severity, calculates sentiment,
    and returns numerical feature vector modifiers.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return {
            "entities": [],
            "severity_summary": {},
            "sentiment": {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0},
            "feature_modifiers": {
                "onset_shift_days": 0.0,
                "stress_boost": 0.0,
                "spotting_flag": 0,
                "cramp_severity": 0
            }
        }
        
    text_clean = text.lower()
    
    # 1. VADER Sentiment Scoring
    sentiment = analyzer.polarity_scores(text)
    
    # 2. Extract Entities and localize context windows
    detected_entities = []
    
    for symptom, patterns in SYMPTOM_TAXONOMY.items():
        for pat in patterns:
            match = re.search(pat, text_clean)
            if match:
                # Determine severity by inspecting nearby window (25 chars before and after match)
                start_idx = max(0, match.start() - 35)
                end_idx = min(len(text_clean), match.end() + 35)
                context_window = text_clean[start_idx:end_idx]
                
                # Check for severity modifiers
                assigned_severity = 3 # Default moderate
                for sev_level in [5, 4, 2, 1]:
                    for mod_pat in SEVERITY_MODIFIERS[sev_level]:
                        if re.search(mod_pat, context_window):
                            assigned_severity = sev_level
                            break
                    if assigned_severity != 3:
                        break
                        
                # Check for negation (e.g. "no cramps", "not bloated")
                is_negated = bool(re.search(r"\b(?:no|not|neither|nor|without|never|zero)\b\s+" + pat, text_clean[max(0, match.start()-20):match.end()]))
                if not is_negated:
                    detected_entities.append({
                        "entity": symptom,
                        "matched_term": match.group(0),
                        "severity": assigned_severity,
                        "context": context_window.strip()
                    })
                break  # Matched category once
                
    # 3. Aggregate severity by symptom category
    severity_summary = {item["entity"]: item["severity"] for item in detected_entities}
    
    # 4. Map into biological feature modifiers
    onset_shift = 0.0
    stress_boost = 0.0
    
    # If acute dysmenorrhea + spotting are recorded, onset is imminent (shortens countdown)
    if "dysmenorrhea" in severity_summary:
        cramp_sev = severity_summary["dysmenorrhea"]
        if cramp_sev >= 3:
            onset_shift -= (cramp_sev - 2) * 0.5  # pulls onset earlier
    else:
        cramp_sev = 0
        
    if "spotting" in severity_summary:
        onset_shift -= 0.5
        
    if "ovulation_pain" in severity_summary:
        # Ovulation is occurring now -> Luteal phase starts ~14 days until period
        onset_shift += 0.0
        
    # High sentiment negativity + fatigue + insomnia indicates acute stress
    if sentiment["compound"] <= -0.4:
        stress_boost += 1.5
    if "insomnia" in severity_summary:
        stress_boost += 1.0
    if "fatigue" in severity_summary and severity_summary["fatigue"] >= 4:
        stress_boost += 0.8
        
    feature_modifiers = {
        "onset_shift_days": round(onset_shift, 2),
        "stress_boost": round(stress_boost, 2),
        "spotting_flag": 1 if "spotting" in severity_summary else 0,
        "cramp_severity": cramp_sev
    }
    
    return {
        "entities": detected_entities,
        "severity_summary": severity_summary,
        "sentiment": sentiment,
        "feature_modifiers": feature_modifiers
    }
