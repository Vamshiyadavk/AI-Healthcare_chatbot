"""
app.py – HealthBot Flask application.

Improvements over v1:
- RESTful /api/predict JSON endpoint (interviewers love this)
- Input validation & sanitisation
- Structured error handling with proper HTTP status codes
- Confidence thresholding with "uncertain" fallback
- Logging middleware
- /api/health liveness probe
- /api/diseases list endpoint
- Environment-based configuration
"""

import os
import re
import logging
from functools import wraps

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App & config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config.update(
    DEBUG=os.getenv("FLASK_DEBUG", "0") == "1",
    SECRET_KEY=os.getenv("SECRET_KEY", "dev-secret-change-in-prod"),
    MIN_CONFIDENCE=float(os.getenv("MIN_CONFIDENCE", "30")),  # % threshold
    MAX_SYMPTOMS_LEN=500,
)

# ---------------------------------------------------------------------------
# Load ML artefacts
# ---------------------------------------------------------------------------
def _load_artefacts():
    model_path = os.path.join(BASE_DIR, "disease_model.pkl")
    vec_path   = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
    rem_path   = os.path.join(BASE_DIR, "home_remedies.csv")

    if not (os.path.exists(model_path) and os.path.exists(vec_path)):
        log.warning("Model files missing. Run: python train_model.py")
        return None, None, None

    model      = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    remedies   = pd.read_csv(rem_path)
    log.info("Model & data loaded successfully.")
    return model, vectorizer, remedies


model, vectorizer, remedies_df = _load_artefacts()

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
SEVERITY_COLORS = {
    "emergency": "danger",
    "serious":   "warning",
    "moderate":  "info",
    "mild":      "success",
    "minor":     "success",
    "varies":    "secondary",
}

SEVERITY_ICONS = {
    "emergency": "🚨",
    "serious":   "⚠️",
    "moderate":  "🔶",
    "mild":      "✅",
    "minor":     "✅",
    "varies":    "ℹ️",
}


def sanitise_symptoms(raw: str) -> str:
    """Strip non-alpha characters except commas/spaces; lowercase."""
    cleaned = re.sub(r"[^a-zA-Z ,]", "", raw)
    return " ".join(cleaned.lower().split())


def predict_disease(symptoms: str) -> dict:
    """Return prediction dict; raises ValueError on bad input."""
    if not symptoms or len(symptoms.strip()) < 3:
        raise ValueError("Please enter at least one symptom (e.g. 'fever, cough').")
    if len(symptoms) > app.config["MAX_SYMPTOMS_LEN"]:
        raise ValueError(f"Symptoms text too long (max {app.config['MAX_SYMPTOMS_LEN']} chars).")

    cleaned = sanitise_symptoms(symptoms)
    vec     = vectorizer.transform([cleaned])

    disease    = model.predict(vec)[0]
    proba      = model.predict_proba(vec)[0]
    confidence = round(float(np.max(proba)) * 100, 1)

    # Top-3 differential diagnoses
    top3_idx      = np.argsort(proba)[::-1][:3]
    top3_diseases = model.classes_[top3_idx]
    top3_probs    = (proba[top3_idx] * 100).round(1).tolist()
    differentials = [
        {"disease": d, "probability": p}
        for d, p in zip(top3_diseases, top3_probs)
    ]

    # Fetch remedy
    row = remedies_df[remedies_df["disease"] == disease]
    if row.empty:
        remedy   = "Please consult a healthcare professional for guidance."
        severity = "varies"
    else:
        remedy   = row.iloc[0]["remedy"]
        severity = row.iloc[0]["severity"]

    return {
        "disease":       disease,
        "confidence":    confidence,
        "remedy":        remedy,
        "severity":      severity,
        "severity_color": SEVERITY_COLORS.get(severity, "secondary"),
        "severity_icon":  SEVERITY_ICONS.get(severity, "ℹ️"),
        "differentials":  differentials,
        "low_confidence": confidence < app.config["MIN_CONFIDENCE"],
    }


def model_required(f):
    """Decorator – returns 503 if model is not loaded."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if model is None or vectorizer is None:
            msg = "ML model not loaded. Run: python train_model.py"
            if request.path.startswith("/api/"):
                return jsonify({"error": msg}), 503
            return render_template("error.html", message=msg), 503
        return f(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Routes – web UI
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    sample_diseases = remedies_df["disease"].sample(min(6, len(remedies_df))).tolist() if remedies_df is not None else []
    return render_template("index.html", sample_diseases=sample_diseases)


@app.route("/predict", methods=["POST"])
@model_required
def predict():
    raw_symptoms = request.form.get("symptoms", "").strip()
    try:
        result = predict_disease(raw_symptoms)
    except ValueError as exc:
        return render_template("index.html", error=str(exc)), 400
    return render_template("result.html", **result, symptoms=raw_symptoms)


# ---------------------------------------------------------------------------
# Routes – REST API
# ---------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def api_health():
    """Liveness probe."""
    return jsonify({
        "status":       "ok",
        "model_loaded": model is not None,
        "disease_count": len(remedies_df) if remedies_df is not None else 0,
    })


@app.route("/api/diseases", methods=["GET"])
def api_diseases():
    """List all supported diseases."""
    if remedies_df is None:
        return jsonify({"error": "Data not loaded"}), 503
    diseases = remedies_df[["disease", "severity"]].sort_values("disease").to_dict(orient="records")
    return jsonify({"count": len(diseases), "diseases": diseases})


@app.route("/api/predict", methods=["POST"])
@model_required
def api_predict():
    """
    JSON endpoint.
    Body: { "symptoms": "fever, cough, headache" }
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    symptoms = request.json.get("symptoms", "").strip()
    try:
        result = predict_disease(symptoms)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    log.info("API predict | disease=%s confidence=%.1f%%", result["disease"], result["confidence"])
    return jsonify(result)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(_):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Endpoint not found"}), 404
    return render_template("error.html", message="Page not found (404)."), 404


@app.errorhandler(500)
def server_error(exc):
    log.exception("Internal server error: %s", exc)
    if request.path.startswith("/api/"):
        return jsonify({"error": "Internal server error"}), 500
    return render_template("error.html", message="Something went wrong. Please try again."), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=app.config["DEBUG"])
