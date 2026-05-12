#```python
"""
app.py
Flask backend for Fatty Liver Disease Detection System
"""

import os
import warnings

import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

import shap  # optional (safe fallback added below)

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global variables (lazy loading)
# ---------------------------------------------------------------------------
MODELS_DIR = "models"

scaler = None
rf_model = None
feature_cols = None
le_gender = None
ann_model = None
shap_explainer = None


# ---------------------------------------------------------------------------
# Load models only when needed
# ---------------------------------------------------------------------------
def load_models():
    global scaler, rf_model, feature_cols, le_gender, ann_model, shap_explainer

    if scaler is None:
        print("Loading models...")

        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.pkl"))
        feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
        le_gender = joblib.load(os.path.join(MODELS_DIR, "label_encoder_gender.pkl"))

        ann_model = tf.keras.models.load_model(
            os.path.join(MODELS_DIR, "ann_model.keras"),
            compile=False
        )

        try:
            shap_explainer = shap.TreeExplainer(rf_model)
        except:
            shap_explainer = None

        print("Models loaded successfully!")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def get_risk_level(prob):
    if prob >= 0.80:
        return "High"
    elif prob >= 0.50:
        return "Moderate"
    return "Low"


def detect_abnormal_parameters(input_dict):
    NORMAL_RANGES = {
        "Total_Bilirubin": (0.2, 1.2),
        "Direct_Bilirubin": (0.0, 0.3),
        "Alkaline_Phosphotase": (44, 147),
        "Alamine_Aminotransferase": (7, 56),
        "Aspartate_Aminotransferase": (10, 40),
        "Total_Protiens": (6.0, 8.3),
        "Albumin": (3.5, 5.0),
        "Albumin_Globulin_Ratio": (1.0, 2.5),
    }

    flags = []
    for feat, (lo, hi) in NORMAL_RANGES.items():
        val = input_dict.get(feat)
        if val is None:
            continue

        val = float(val)
        if val < lo:
            flags.append(f"{feat} LOW ({val})")
        elif val > hi:
            flags.append(f"{feat} HIGH ({val})")

    return flags


def get_shap_values(row_scaled):
    if shap_explainer is None:
        return []

    try:
        shap_vals = shap_explainer.shap_values(row_scaled)

        if isinstance(shap_vals, list):
            return np.array(shap_vals[1]).flatten()

        arr = np.array(shap_vals)
        if arr.ndim == 3:
            return arr[0, :, 1]

        return arr.flatten()
    except:
        return []


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        load_models()

        data = request.get_json()

        # Gender encoding
        gender = data.get("Gender", "Male").capitalize()
        gender_enc = int(le_gender.transform([gender])[0]) if gender in le_gender.classes_ else 0

        input_dict = {
            "Age": float(data["Age"]),
            "Gender": gender_enc,
            "Total_Bilirubin": float(data["Total_Bilirubin"]),
            "Direct_Bilirubin": float(data["Direct_Bilirubin"]),
            "Alkaline_Phosphotase": float(data["Alkaline_Phosphotase"]),
            "Alamine_Aminotransferase": float(data["Alamine_Aminotransferase"]),
            "Aspartate_Aminotransferase": float(data["Aspartate_Aminotransferase"]),
            "Total_Protiens": float(data["Total_Protiens"]),
            "Albumin": float(data["Albumin"]),
            "Albumin_Globulin_Ratio": float(data["Albumin_Globulin_Ratio"]),
        }

        row = np.array([input_dict[c] for c in feature_cols]).reshape(1, -1)
        row_scaled = scaler.transform(row)

        # Predictions
        rf_prob = rf_model.predict_proba(row_scaled)[0][1]
        ann_prob = ann_model.predict(row_scaled, verbose=0)[0][0]
        final_prob = (rf_prob + ann_prob) / 2

        risk = get_risk_level(final_prob)

        # SHAP
        shap_vals = get_shap_values(row_scaled)

        shap_output = []
        for i in range(min(len(shap_vals), len(feature_cols))):
            shap_output.append({
                "feature": feature_cols[i],
                "value": float(round(shap_vals[i], 4))
            })

        # Final response (FIXED for frontend)
        return jsonify({
            "status": "Fatty Liver Disease Detected" if final_prob >= 0.5 else "No Liver Disease Detected",
            "binary_label": int(final_prob >= 0.5),
            "risk_level": risk,
            "confidence_pct": round(final_prob * 100, 2),
            "rf_confidence_pct": round(rf_prob * 100, 2),
            "ann_confidence_pct": round(ann_prob * 100, 2),
            "shap_factors": shap_output if shap_output else [],
            "abnormal_flags": detect_abnormal_parameters(input_dict),
            "recommendations": {
                "High": [
                    "Consult a doctor immediately",
                    "Get full liver tests",
                    "Avoid alcohol"
                ],
                "Moderate": [
                    "Improve diet",
                    "Exercise regularly",
                    "Follow-up check"
                ],
                "Low": [
                    "Maintain healthy lifestyle",
                    "Regular checkups"
                ]
            }[risk]
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()})


# ---------------------------------------------------------------------------
# Run server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
#```
