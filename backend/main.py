"""
main.py
-------
FastAPI entry point for the PhishGuard backend.

Exposes six endpoints:
  POST /predict          — scan a URL, returns prediction + SHAP + WHOIS
  GET  /whois            — standalone WHOIS domain analysis
  POST /feedback         — submit a user correction
  GET  /feedback/summary — view accumulated feedback stats
  POST /retrain          — manually trigger model retraining
  GET  /health           — server + model status check

Run locally:
    cd backend
    venv\\Scripts\\activate          # Windows
    source venv/bin/activate        # macOS / Linux
    uvicorn main:app --reload

Environment variables (set in .env, loaded via python-dotenv):
    GOOGLE_SAFE_BROWSING_API_KEY
    PHISHTANK_API_KEY
    OPEN_PAGE_RANK_API_KEY
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

# Load .env before importing any module that reads os.getenv()
load_dotenv()

from feature_extractor import extract_features
from predictor import get_registry, PredictionResult
from explainer import explain_local, explain_global
from whois_analyzer import analyze_whois
from feedback import (
    save_feedback, get_feedback_summary,
    retrain_models, FeedbackEntry,
)


# ── startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once on server startup:
      1. Load all three models + select best by F1
      2. Pre-compute and cache global SHAP explanation
    This ensures the first /predict request is fast.
    """
    print("[startup] Loading models...")
    registry = get_registry()
    print(f"[startup] Best model: {registry.best_model_name}")

    print("[startup] Pre-computing global SHAP explanation...")
    explain_global(registry)
    print("[startup] Global SHAP cache ready.")

    print("[startup] PhishGuard API is ready 🛡️")
    yield
    print("[shutdown] PhishGuard API shutting down.")


# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "PhishGuard API",
    description = "Adaptive Explainable Phishing Detection — NGIT CSE 2025-26",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# Open CORS for development — restrict origins before deploying to production
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── request / response schemas ────────────────────────────────────────────────

class ScanRequest(BaseModel):
    url: str

    model_config = {"json_schema_extra": {"example": {"url": "http://paypal-secure-login.xyz/verify"}}}


class FeedbackRequest(BaseModel):
    url             : str
    predicted_label : str    # "phishing" or "legitimate"
    correct_label   : str    # what the user says it actually is

    model_config = {"json_schema_extra": {"example": {
        "url"             : "http://paypal-secure-login.xyz/verify",
        "predicted_label" : "phishing",
        "correct_label"   : "legitimate",
    }}}


class WHOISRequest(BaseModel):
    url: str

    model_config = {"json_schema_extra": {"example": {"url": "http://example.com"}}}


# ── helper ────────────────────────────────────────────────────────────────────

def _extract_domain(url: str) -> str:
    """Extract bare domain from a full URL."""
    from urllib.parse import urlparse
    return urlparse(url).netloc.replace("www.", "").split(":")[0]


def _serialise_result(
    result     : PredictionResult,
    explanation,
    whois_result,
) -> dict:
    """
    Build the full JSON response for /predict.
    Converts dataclasses to dicts for FastAPI serialisation.
    """
    return {
        # ── Primary prediction ─────────────────────────────────────────────
        "prediction": {
            "label"       : result.label,
            "is_phishing" : result.is_phishing,
            "confidence"  : result.confidence,
            "trust_score" : result.trust_score,
            "model_used"  : result.model_used,
            "warning"     : result.warning,
        },

        # ── Model comparison (all three models) ────────────────────────────
        "model_comparison": [
            {
                "name"        : s.name,
                "label"       : s.label,
                "confidence"  : s.confidence,
                "trust_score" : s.trust_score,
            }
            for s in result.all_scores
        ],

        # ── SHAP explanation ───────────────────────────────────────────────
        "explanation": {
            "model_used"       : explanation.model_used,
            "explanation_type" : explanation.explanation_type,
            "base_value"       : explanation.base_value,
            "prediction_delta" : explanation.prediction_delta,

            # Local — top 5 features for THIS prediction
            "local_features": [
                {
                    "name"       : f.name,
                    "label"      : f.label,
                    "value"      : f.value,
                    "shap_value" : f.shap_value,
                    "direction"  : f.direction,
                }
                for f in explanation.local_features
            ],

            # Global — top 5 most important features overall
            "global_features": [
                {
                    "name"       : f.name,
                    "label"      : f.label,
                    "shap_value" : f.shap_value,
                    "direction"  : f.direction,
                }
                for f in explanation.global_features
            ],
        },

        # ── WHOIS analysis ─────────────────────────────────────────────────
        "whois": {
            "domain_name"       : whois_result.domain_name,
            "registrar"         : whois_result.registrar,
            "creation_date"     : whois_result.creation_date,
            "expiry_date"       : whois_result.expiry_date,
            "last_updated"      : whois_result.last_updated,
            "country"           : whois_result.country,
            "name_servers"      : whois_result.name_servers,
            "privacy_protected" : whois_result.privacy_protected,
            "domain_age_days"   : whois_result.domain_age_days,
            "days_until_expiry" : whois_result.days_until_expiry,
            "days_since_update" : whois_result.days_since_update,
            "risk_level"        : whois_result.risk_level,
            "lookup_success"    : whois_result.lookup_success,
            "queried_at"        : whois_result.queried_at,
            "risk_flags": [
                {
                    "severity" : f.severity,
                    "title"    : f.title,
                    "detail"   : f.detail,
                    "icon"     : f.icon,
                }
                for f in whois_result.risk_flags
            ],
        },

        # ── Raw features (for debugging / advanced frontend use) ───────────
        "features": result.features,

        # ── Meta ───────────────────────────────────────────────────────────
        "scanned_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.post("/predict", summary="Scan a URL for phishing")
async def predict(request: ScanRequest):
    """
    Main endpoint — scans a URL end-to-end.

    Pipeline:
      1. Extract 28 features from the URL (parsing + DNS + HTML + APIs)
      2. Run all three models, select best by F1
      3. Generate local + global SHAP explanation
      4. Perform WHOIS domain analysis
      5. Return everything in one unified response

    The frontend calls this single endpoint and distributes the response
    across TrustScore, SHAPExplanation, WHOISPanel, and ModelComparison.
    """
    url = request.url.strip()

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    try:
        # Step 1 — Feature extraction
        features = extract_features(url)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid URL: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}")

    try:
        # Step 2 — Prediction
        from predictor import predict as run_predict
        result = run_predict(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    try:
        # Step 3 — SHAP explanation
        registry    = get_registry()
        explanation = explain_local(features, registry)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {e}")

    try:
        # Step 4 — WHOIS analysis
        domain       = _extract_domain(url)
        whois_result = analyze_whois(domain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"WHOIS analysis failed: {e}")

    # Step 5 — Serialise and return
    return _serialise_result(result, explanation, whois_result)


@app.get("/whois", summary="Standalone WHOIS domain analysis")
async def whois_lookup(url: str):
    """
    Standalone WHOIS endpoint — useful when the frontend wants to refresh
    domain info without re-running the full prediction pipeline.

    Query param: ?url=http://example.com
    """
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    domain = _extract_domain(url)
    if not domain:
        raise HTTPException(status_code=422, detail="Could not extract domain from URL.")

    try:
        result = analyze_whois(domain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"WHOIS lookup failed: {e}")

    return {
        "domain_name"       : result.domain_name,
        "registrar"         : result.registrar,
        "creation_date"     : result.creation_date,
        "expiry_date"       : result.expiry_date,
        "last_updated"      : result.last_updated,
        "country"           : result.country,
        "name_servers"      : result.name_servers,
        "privacy_protected" : result.privacy_protected,
        "domain_age_days"   : result.domain_age_days,
        "days_until_expiry" : result.days_until_expiry,
        "risk_level"        : result.risk_level,
        "lookup_success"    : result.lookup_success,
        "queried_at"        : result.queried_at,
        "risk_flags": [
            {
                "severity" : f.severity,
                "title"    : f.title,
                "detail"   : f.detail,
                "icon"     : f.icon,
            }
            for f in result.risk_flags
        ],
    }


@app.post("/feedback", summary="Submit a correction for a prediction")
async def submit_feedback(request: FeedbackRequest):
    """
    Receives a user correction and saves it to feedback.csv.

    The frontend shows a thumbs-up / thumbs-down button after each scan.
    If the user disagrees with the prediction, they submit this endpoint
    with the URL, what the model said, and what the correct label is.
    """
    # Validate labels
    valid_labels = {"phishing", "legitimate"}
    if request.predicted_label not in valid_labels:
        raise HTTPException(
            status_code=422,
            detail=f"predicted_label must be one of: {valid_labels}",
        )
    if request.correct_label not in valid_labels:
        raise HTTPException(
            status_code=422,
            detail=f"correct_label must be one of: {valid_labels}",
        )
    if request.predicted_label == request.correct_label:
        raise HTTPException(
            status_code=422,
            detail="predicted_label and correct_label are the same — no correction needed.",
        )

    # Re-extract features for the URL to attach to feedback row
    url = request.url.strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    try:
        features = extract_features(url)
    except Exception:
        # If feature extraction fails, save with empty features rather than rejecting
        features = {}

    entry = FeedbackEntry(
        url             = url,
        predicted_label = request.predicted_label,
        correct_label   = request.correct_label,
        features        = features,
    )

    success = save_feedback(entry)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to save feedback.")

    summary = get_feedback_summary()

    return {
        "message"       : "Feedback saved. Thank you for helping improve PhishGuard!",
        "total_feedback": summary.get("total", 0),
    }


@app.get("/feedback/summary", summary="View accumulated feedback statistics")
async def feedback_summary():
    """
    Returns a summary of all stored user corrections.
    Useful for the team to decide when to trigger retraining.
    """
    return get_feedback_summary()


@app.post("/retrain", summary="Manually trigger model retraining")
async def retrain():
    """
    Retrains all three models on original data + user feedback.

    Steps performed:
      1. Load phishing.csv + feedback.csv
      2. Train Logistic Regression, Random Forest, XGBoost
      3. Evaluate each by F1 score
      4. Save new .pkl files (overwrites existing models)
      5. Clear SHAP cache + reset ModelRegistry

    ⚠️  This is a blocking call — the server will be unresponsive
    during training (typically 10–30 seconds for this dataset size).
    For production, move to a background task.
    """
    result = retrain_models()

    if not result.success:
        raise HTTPException(
            status_code = 500,
            detail      = f"Retraining failed: {result.error}",
        )

    return {
        "message"          : result.message,
        "new_best_model"   : result.new_best_model,
        "f1_scores"        : result.f1_scores,
        "feedback_used"    : result.feedback_count,
        "total_trained_on" : result.training_size,
        "duration_seconds" : result.duration_seconds,
    }


@app.get("/health", summary="Server and model health check")
async def health():
    """
    Returns current server status, loaded model info, and feedback count.
    Useful for confirming the backend is live before frontend development.
    """
    try:
        registry = get_registry()
        models_loaded = list(registry.models.keys())
        best_model    = registry.best_model_name
        models_ok     = True
    except Exception as e:
        models_loaded = []
        best_model    = None
        models_ok     = False

    feedback_count = get_feedback_summary().get("total", 0)

    return {
        "status"         : "ok" if models_ok else "degraded",
        "models_loaded"  : models_loaded,
        "best_model"     : best_model,
        "feedback_count" : feedback_count,
        "api_version"    : "1.0.0",
        "checked_at"     : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ── dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)