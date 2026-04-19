"""
predictor.py
------------
Loads trained models from ../backend/models/, selects the best model
dynamically by F1 score, and runs predictions on extracted features.

Returns a structured PredictionResult containing:
  - label         : "phishing" or "legitimate"
  - confidence    : float (0.0 – 1.0), probability of the predicted class
  - trust_score   : int (0 – 100), human-readable confidence percentage
  - model_used    : name of the best-performing model
  - all_scores    : predictions from all three models (for ModelComparison panel)
  - features      : raw feature dict (passed to explainer.py for SHAP)

Model selection
---------------
On startup, all three models are loaded and evaluated against a small
held-out validation slice (recreated with the same random_state=42 used
during training). The model with the highest F1 score is selected as the
primary predictor for the session. This avoids re-selecting on every
request (expensive) while still being data-driven rather than hardcoded.
"""

import os
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# ── paths ─────────────────────────────────────────────────────────────────────

MODELS_DIR   = os.path.join(os.path.dirname(__file__), "models")
DATA_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "phishing.csv")

MODEL_PATHS  = {
    "Logistic Regression" : os.path.join(MODELS_DIR, "model_lr.pkl"),
    "Random Forest"       : os.path.join(MODELS_DIR, "model_rf.pkl"),
    "XGBoost"             : os.path.join(MODELS_DIR, "model_xgb.pkl"),
}
SCALER_PATH  = os.path.join(MODELS_DIR, "scaler.pkl")

# Features dropped from training (must match train_models.ipynb)
DROPPED_FEATURES = ["web_traffic", "Links_pointing_to_page"]


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class ModelScore:
    """Prediction result from a single model."""
    name       : str
    label      : str          # "phishing" or "legitimate"
    confidence : float        # probability of predicted class
    trust_score: int          # 0–100


@dataclass
class PredictionResult:
    """Full prediction result returned to the FastAPI endpoint."""
    label       : str                        # primary model's verdict
    confidence  : float                      # primary model's confidence
    trust_score : int                        # primary model's trust score (0–100)
    model_used  : str                        # name of best model
    all_scores  : list[ModelScore]           # all three models (for comparison panel)
    features    : dict                       # raw feature dict (for SHAP in explainer.py)
    is_phishing : bool                       # convenience bool for frontend logic
    warning     : Optional[str] = None       # real-time warning message if phishing


# ── model registry (loaded once at import time) ───────────────────────────────

class ModelRegistry:
    """
    Loads all models on startup and selects the best by F1 score.
    Singleton — instantiated once as MODULE_REGISTRY at the bottom of this file.
    """

    def __init__(self):
        self.models   : dict = {}   # name → model object
        self.scaler              = None
        self.best_model_name : str = ""
        self._load_models()
        self._select_best_model()

    def _load_models(self):
        """Load all .pkl files from the models directory."""
        missing = []
        for name, path in MODEL_PATHS.items():
            if not os.path.exists(path):
                missing.append(path)
                continue
            self.models[name] = joblib.load(path)

        if missing:
            raise FileNotFoundError(
                f"Missing model files — have you run train_models.ipynb?\n"
                f"  {chr(10).join(missing)}"
            )

        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(
                f"Missing scaler: {SCALER_PATH}\n"
                "Run train_models.ipynb to generate scaler.pkl."
            )
        self.scaler = joblib.load(SCALER_PATH)

    def _select_best_model(self):
        """
        Recreate the same validation split used during training and
        evaluate all three models by F1 score. Selects the best.
        """
        if not os.path.exists(DATA_PATH):
            # Fallback: default to XGBoost if dataset isn't available at runtime
            self.best_model_name = "XGBoost"
            return

        df = pd.read_csv(DATA_PATH)
        df["Result"] = df["Result"].replace(-1, 0)
        df = df.drop(columns=DROPPED_FEATURES, errors="ignore")

        X = df.drop("Result", axis=1)
        y = df["Result"]

        # Same split as training — random_state=42, stratify=y, test_size=0.3
        _, X_val, _, y_val = train_test_split(
            X, y,
            test_size=0.3,
            random_state=42,
            stratify=y,
        )

        X_val_scaled = self.scaler.transform(X_val)

        best_f1   = -1.0
        best_name = "XGBoost"

        for name, model in self.models.items():
            X_input = X_val_scaled if name == "Logistic Regression" else X_val
            y_pred  = model.predict(X_input)
            score   = f1_score(y_val, y_pred)
            if score > best_f1:
                best_f1   = score
                best_name = name

        self.best_model_name = best_name
        print(f"[predictor] Best model: {best_name} (F1 = {best_f1:.4f})")

    @property
    def best_model(self):
        return self.models[self.best_model_name]


# Instantiate once — imported by main.py and explainer.py
MODULE_REGISTRY: Optional[ModelRegistry] = None

def get_registry() -> ModelRegistry:
    """
    Lazy singleton — initialised on first call.
    FastAPI's startup event calls this so errors surface early.
    """
    global MODULE_REGISTRY
    if MODULE_REGISTRY is None:
        MODULE_REGISTRY = ModelRegistry()
    return MODULE_REGISTRY


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_input_df(features: dict) -> pd.DataFrame:
    """
    Convert the raw feature dict from feature_extractor.py into a
    single-row DataFrame in the exact column order the models expect.
    """
    return pd.DataFrame([features])


def _confidence_to_trust(confidence: float, label: str) -> int:
    """
    Convert model probability to a 0–100 trust score.

    Trust score represents how trustworthy the URL is:
      - Legitimate prediction with high confidence → high trust (close to 100)
      - Phishing prediction with high confidence   → low trust (close to 0)
    """
    if label == "legitimate":
        return round(confidence * 100)
    else:
        # Invert: high phishing confidence = low trust
        return round((1 - confidence) * 100)


def _make_warning(features: dict) -> Optional[str]:
    """
    Generate a real-time security warning string based on
    the most critical phishing signals found in the features.
    Returns None if no strong signals found.
    """
    warnings = []

    if features.get("having_IP_Address") == -1:
        warnings.append("URL uses a raw IP address instead of a domain name")
    if features.get("SSLfinal_State") == -1:
        warnings.append("SSL certificate is invalid, expired, or missing")
    if features.get("age_of_domain") == -1:
        warnings.append("Domain was registered less than 6 months ago")
    if features.get("DNSRecord") == -1:
        warnings.append("No DNS record found for this domain")
    if features.get("Prefix_Suffix") == -1:
        warnings.append("Domain contains a hyphen — common in spoofed URLs")
    if features.get("Shortining_Service") == -1:
        warnings.append("URL uses a shortening service to hide the real destination")
    if features.get("Statistical_report") == -1:
        warnings.append("Domain is flagged in PhishTank's phishing database")
    if features.get("Google_Index") == -1:
        warnings.append("URL is flagged by Google Safe Browsing")

    if not warnings:
        return None

    # Return the most critical warning (first match) as a single string
    return warnings[0]


# ── main prediction function ──────────────────────────────────────────────────

def predict(features: dict) -> PredictionResult:
    """
    Main entry point called by the FastAPI endpoint.

    Parameters
    ----------
    features : dict
        Output of feature_extractor.extract_features(url) —
        a dict of 28 feature name → value pairs (all in {-1, 0, 1}).

    Returns
    -------
    PredictionResult with full prediction data for the frontend.
    """
    registry = get_registry()
    df_input = _build_input_df(features)
    scaled_values = registry.scaler.transform(df_input)
    df_scaled = pd.DataFrame(scaled_values, columns=df_input.columns)

    all_scores = []

    for name, model in registry.models.items():
        X_input = df_scaled if name == "Logistic Regression" else df_input

        raw_pred   = model.predict(X_input)[0]
        proba      = model.predict_proba(X_input)[0]

        # proba[0] = P(phishing), proba[1] = P(legitimate)
        label      = "legitimate" if raw_pred == 1 else "phishing"
        confidence = float(proba[raw_pred])  # confidence in the predicted class
        trust      = _confidence_to_trust(confidence, label)

        all_scores.append(ModelScore(
            name        = name,
            label       = label,
            confidence  = round(confidence, 4),
            trust_score = trust,
        ))

    # Primary prediction — from best model
    best_score  = next(s for s in all_scores if s.name == registry.best_model_name)
    is_phishing = best_score.label == "phishing"

    return PredictionResult(
        label       = best_score.label,
        confidence  = best_score.confidence,
        trust_score = best_score.trust_score,
        model_used  = registry.best_model_name,
        all_scores  = all_scores,
        features    = features,
        is_phishing = is_phishing,
        warning     = _make_warning(features) if is_phishing else None,
    )