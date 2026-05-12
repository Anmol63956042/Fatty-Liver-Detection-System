"""
predict.py
==========
Standalone inference module for the Fatty Liver Disease Detection System.

This module is entirely independent of train_model.py.  It loads only the
pre-serialised artefacts produced during training and performs single-patient
or batch predictions with risk stratification, abnormality flagging, and
clinical recommendations.

Required artefacts (models/):
    - scaler.pkl
    - label_encoder_gender.pkl
    - feature_cols.pkl
    - random_forest_model.pkl
    - ann_model.keras

Usage (interactive / script):
    python predict.py
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
MODELS_DIR = "models"

# Normal reference ranges for clinical flag detection
# Format: feature_name -> (lower_bound, upper_bound)
NORMAL_RANGES = {
    "Total_Bilirubin":            (0.2,  1.2),
    "Direct_Bilirubin":           (0.0,  0.3),
    "Alkaline_Phosphotase":       (44,   147),
    "Alamine_Aminotransferase":   (7,    56),
    "Aspartate_Aminotransferase": (10,   40),
    "Total_Protiens":             (6.0,  8.3),
    "Albumin":                    (3.5,  5.0),
    "Albumin_Globulin_Ratio":     (1.0,  2.5),
}

FEATURE_DISPLAY = {
    "Age":                       "Age",
    "Gender":                    "Gender",
    "Total_Bilirubin":           "Total Bilirubin",
    "Direct_Bilirubin":          "Direct Bilirubin",
    "Alkaline_Phosphotase":      "Alkaline Phosphotase",
    "Alamine_Aminotransferase":  "SGPT (Alamine AT)",
    "Aspartate_Aminotransferase":"SGOT (Aspartate AT)",
    "Total_Protiens":            "Total Proteins",
    "Albumin":                   "Albumin (ALB)",
    "Albumin_Globulin_Ratio":    "A/G Ratio",
}

RECOMMENDATIONS = {
    "High": [
        "Immediately consult a hepatologist for specialist evaluation.",
        "Request a complete Liver Function Test (LFT) panel.",
        "Schedule abdominal ultrasound or CT scan imaging.",
        "Discontinue alcohol consumption completely.",
        "Review all current medications for hepatotoxic potential.",
        "Monitor liver enzyme levels every two weeks.",
    ],
    "Moderate": [
        "Schedule a follow-up appointment with a physician within two weeks.",
        "Repeat LFT blood panel in 4 to 6 weeks.",
        "Adopt a low-fat, high-fibre dietary regimen.",
        "Increase physical activity to at least 30 minutes of moderate exercise daily.",
        "Eliminate alcohol and reduce intake of processed foods.",
        "Discuss Vitamin E supplementation with your doctor.",
    ],
    "Low": [
        "Maintain a balanced diet with adequate hydration.",
        "Sustain physical activity at 150 minutes per week of moderate exercise.",
        "Schedule a routine annual liver function check-up.",
        "Limit alcohol consumption to within recommended guidelines.",
        "Maintain a healthy body mass index (BMI).",
    ],
}


# ---------------------------------------------------------------------------
# Section 1: Load Pre-Trained Artefacts
# ---------------------------------------------------------------------------

def load_artefacts() -> dict:
    """
    Load all serialised model artefacts from the models directory.

    Returns
    -------
    dict containing: scaler, label_encoder, feature_cols, rf_model, ann_model
    """
    required = [
        "scaler.pkl",
        "label_encoder_gender.pkl",
        "feature_cols.pkl",
        "random_forest_model.pkl",
        "ann_model.keras",
    ]
    for fname in required:
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required artefact not found: {path}\n"
                "Run train_model.py first to generate model artefacts."
            )

    return {
        "scaler":         joblib.load(os.path.join(MODELS_DIR, "scaler.pkl")),
        "label_encoder":  joblib.load(os.path.join(MODELS_DIR, "label_encoder_gender.pkl")),
        "feature_cols":   joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl")),
        "rf_model":       joblib.load(os.path.join(MODELS_DIR, "random_forest_model.pkl")),
        "ann_model":      tf.keras.models.load_model(os.path.join(MODELS_DIR, "ann_model.keras")),
    }


# ---------------------------------------------------------------------------
# Section 2: Risk Stratification
# ---------------------------------------------------------------------------

def get_risk_level(probability: float) -> str:
    """
    Classify predicted probability into a clinical risk tier.

    Risk thresholds (consistent with project specification):
        >= 0.80  -> High
        >= 0.50  -> Moderate
        <  0.50  -> Low

    Parameters
    ----------
    probability : float in [0, 1], ensemble disease probability.

    Returns
    -------
    str : "High", "Moderate", or "Low"
    """
    if probability >= 0.80:
        return "High"
    elif probability >= 0.50:
        return "Moderate"
    else:
        return "Low"


# ---------------------------------------------------------------------------
# Section 3: Abnormality Detection
# ---------------------------------------------------------------------------

def detect_abnormal_parameters(input_dict: dict) -> list:
    """
    Compare input clinical values against established normal reference ranges
    and return a list of flags for parameters outside normal bounds.

    Parameters
    ----------
    input_dict : dict mapping internal feature names to their numeric values.

    Returns
    -------
    list of str : descriptive flag messages for each abnormal parameter.
    """
    flags = []
    for feat, (lo, hi) in NORMAL_RANGES.items():
        val = input_dict.get(feat)
        if val is None:
            continue
        val = float(val)
        display = FEATURE_DISPLAY.get(feat, feat)
        if val < lo:
            flags.append(f"{display}: LOW ({val}) - Normal range: {lo}-{hi}")
        elif val > hi:
            flags.append(f"{display}: HIGH ({val}) - Normal range: {lo}-{hi}")
    return flags


# ---------------------------------------------------------------------------
# Section 4: Core Prediction Function
# ---------------------------------------------------------------------------

def predict_single(
    age:                  float,
    gender:               str,
    total_bilirubin:      float,
    direct_bilirubin:     float,
    alkaline_phosphotase: float,
    alamine_at:           float,
    aspartate_at:         float,
    total_protiens:       float,
    albumin:              float,
    ag_ratio:             float,
    artefacts:            dict = None,
) -> dict:
    """
    Perform a single-patient prediction using the trained ensemble
    (Random Forest + ANN averaged probability).

    Parameters
    ----------
    age                  : Patient age in years.
    gender               : 'Male' or 'Female'.
    total_bilirubin      : Total Bilirubin in mg/dL.
    direct_bilirubin     : Direct Bilirubin in mg/dL.
    alkaline_phosphotase : Alkaline Phosphotase in U/L.
    alamine_at           : SGPT / Alamine Aminotransferase in U/L.
    aspartate_at         : SGOT / Aspartate Aminotransferase in U/L.
    total_protiens       : Total Proteins in g/dL.
    albumin              : Albumin in g/dL.
    ag_ratio             : Albumin/Globulin Ratio.
    artefacts            : Pre-loaded artefact dict (optional; loaded if None).

    Returns
    -------
    dict with keys:
        status, risk_level, confidence_pct, rf_confidence_pct,
        ann_confidence_pct, binary_label, abnormal_flags, recommendations
    """
    if artefacts is None:
        artefacts = load_artefacts()

    scaler        = artefacts["scaler"]
    label_encoder = artefacts["label_encoder"]
    feature_cols  = artefacts["feature_cols"]
    rf_model      = artefacts["rf_model"]
    ann_model     = artefacts["ann_model"]

    # Encode gender using the same LabelEncoder fitted during training
    gender_clean = gender.strip().capitalize()
    if gender_clean not in label_encoder.classes_:
        # Default to first available class if unseen label provided
        gender_enc = 0
    else:
        gender_enc = int(label_encoder.transform([gender_clean])[0])

    # Build internal named dictionary for abnormality detection
    raw_dict = {
        "Age":                       age,
        "Gender":                    gender_enc,
        "Total_Bilirubin":           total_bilirubin,
        "Direct_Bilirubin":          direct_bilirubin,
        "Alkaline_Phosphotase":      alkaline_phosphotase,
        "Alamine_Aminotransferase":  alamine_at,
        "Aspartate_Aminotransferase":aspartate_at,
        "Total_Protiens":            total_protiens,
        "Albumin":                   albumin,
        "Albumin_Globulin_Ratio":    ag_ratio,
    }

    # Align to feature_cols ordering saved during training
    row_values = [raw_dict[c] for c in feature_cols]
    row_array  = np.array(row_values, dtype=float).reshape(1, -1)
    row_scaled = scaler.transform(row_array)

    # Obtain probability from each model
    rf_prob  = float(rf_model.predict_proba(row_scaled)[0][1])
    ann_prob = float(ann_model.predict(row_scaled, verbose=0)[0][0])

    # Ensemble: simple average of both model probabilities
    ensemble_prob = (rf_prob + ann_prob) / 2.0
    binary_label  = int(ensemble_prob >= 0.5)
    risk_level    = get_risk_level(ensemble_prob)

    abnormal_flags   = detect_abnormal_parameters(raw_dict)
    recommendations  = RECOMMENDATIONS[risk_level]

    return {
        "status":             "Fatty Liver Disease Detected" if binary_label == 1
                              else "No Liver Disease Detected",
        "binary_label":       binary_label,
        "risk_level":         risk_level,
        "confidence_pct":     round(ensemble_prob * 100, 2),
        "rf_confidence_pct":  round(rf_prob * 100, 2),
        "ann_confidence_pct": round(ann_prob * 100, 2),
        "abnormal_flags":     abnormal_flags,
        "recommendations":    recommendations,
    }


# ---------------------------------------------------------------------------
# Section 5: Result Display
# ---------------------------------------------------------------------------

def display_result(result: dict) -> None:
    """
    Print a structured, human-readable prediction report to stdout.

    Parameters
    ----------
    result : dict returned by predict_single().
    """
    separator = "=" * 60

    print(separator)
    print("  FATTY LIVER DISEASE DETECTION - PREDICTION REPORT")
    print(separator)
    print(f"  Diagnosis   : {result['status']}")
    print(f"  Risk Level  : {result['risk_level']}")
    print(f"  Confidence  : {result['confidence_pct']}%  "
          f"(RF: {result['rf_confidence_pct']}%  |  ANN: {result['ann_confidence_pct']}%)")
    print(separator)

    if result["abnormal_flags"]:
        print("  ABNORMAL PARAMETERS DETECTED:")
        for flag in result["abnormal_flags"]:
            print(f"    - {flag}")
        print()

    print("  CLINICAL RECOMMENDATIONS:")
    for i, rec in enumerate(result["recommendations"], 1):
        print(f"    {i}. {rec}")

    print(separator + "\n")


# ---------------------------------------------------------------------------
# Section 6: Interactive Console Entry Point
# ---------------------------------------------------------------------------

def interactive_predict(artefacts: dict) -> None:
    """
    Prompt the user to enter patient clinical parameters via the console
    and display the resulting prediction report.

    Parameters
    ----------
    artefacts : Pre-loaded artefact dict to avoid repeated disk reads.
    """
    print("\n" + "=" * 60)
    print("  Enter Patient Clinical Parameters")
    print("=" * 60)

    def read_float(prompt, default=0.0):
        """Read a float from stdin with a fallback default on empty input."""
        raw = input(prompt).strip()
        return float(raw) if raw else default

    age          = read_float("  Age (years)                   : ")
    gender       = input("  Gender (Male/Female)           : ").strip() or "Male"
    total_bil    = read_float("  Total Bilirubin (mg/dL)       : ")
    direct_bil   = read_float("  Direct Bilirubin (mg/dL)      : ")
    alk_phos     = read_float("  Alkaline Phosphotase (U/L)    : ")
    sgpt         = read_float("  SGPT / Alamine AT (U/L)       : ")
    sgot         = read_float("  SGOT / Aspartate AT (U/L)     : ")
    total_prot   = read_float("  Total Proteins (g/dL)         : ")
    albumin      = read_float("  Albumin (g/dL)                : ")
    ag_ratio     = read_float("  A/G Ratio                     : ")

    result = predict_single(
        age=age,
        gender=gender,
        total_bilirubin=total_bil,
        direct_bilirubin=direct_bil,
        alkaline_phosphotase=alk_phos,
        alamine_at=sgpt,
        aspartate_at=sgot,
        total_protiens=total_prot,
        albumin=albumin,
        ag_ratio=ag_ratio,
        artefacts=artefacts,
    )
    display_result(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Default test patient records
# Modify these values to test different clinical scenarios.
# ---------------------------------------------------------------------------

TEST_PATIENTS = [
    {
        "label":                 "Test Patient 1 - Likely Disease (High Risk)",
        "age":                   45,
        "gender":                "Male",
        "total_bilirubin":       3.2,
        "direct_bilirubin":      1.5,
        "alkaline_phosphotase":  320,
        "alamine_at":            95,
        "aspartate_at":          82,
        "total_protiens":        5.8,
        "albumin":               3.1,
        "ag_ratio":              0.9,
    },
    {
        "label":                 "Test Patient 2 - Likely Healthy (Low Risk)",
        "age":                   32,
        "gender":                "Female",
        "total_bilirubin":       0.8,
        "direct_bilirubin":      0.2,
        "alkaline_phosphotase":  95,
        "alamine_at":            22,
        "aspartate_at":          25,
        "total_protiens":        7.1,
        "albumin":               4.2,
        "ag_ratio":              1.5,
    },
    {
        "label":                 "Test Patient 3 - Borderline (Moderate Risk)",
        "age":                   55,
        "gender":                "Male",
        "total_bilirubin":       1.8,
        "direct_bilirubin":      0.7,
        "alkaline_phosphotase":  180,
        "alamine_at":            60,
        "aspartate_at":          48,
        "total_protiens":        6.3,
        "albumin":               3.6,
        "ag_ratio":              1.1,
    },
]


if __name__ == "__main__":

    # Load artefacts once; reused across all test predictions
    print("Loading model artefacts ...")
    artefacts = load_artefacts()
    print("Artefacts loaded successfully.")

    # Run prediction for every default test patient
    for patient in TEST_PATIENTS:
        print(f"\n>>> {patient['label']}")
        result = predict_single(
            age                  = patient["age"],
            gender               = patient["gender"],
            total_bilirubin      = patient["total_bilirubin"],
            direct_bilirubin     = patient["direct_bilirubin"],
            alkaline_phosphotase = patient["alkaline_phosphotase"],
            alamine_at           = patient["alamine_at"],
            aspartate_at         = patient["aspartate_at"],
            total_protiens       = patient["total_protiens"],
            albumin              = patient["albumin"],
            ag_ratio             = patient["ag_ratio"],
            artefacts            = artefacts,
        )
        display_result(result)
