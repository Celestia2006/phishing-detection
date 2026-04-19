"""
explainer.py
------------
Generates SHAP explanations for phishing detection predictions.

Two explanation types:
  - Local  : per-prediction waterfall — which features drove THIS result
  - Global : summary bar chart        — which features matter most overall

Both return only the top 5 features for a clean frontend display.

Usage (called from main.py)
---------------------------
    from explainer import explain_local, explain_global, ExplanationResult

    # Local — call after every prediction
    local  = explain_local(features, registry)

    # Global — call once at startup, cache the result
    global_ = explain_global(registry)

Dependencies
------------
    predictor.py  → provides ModelRegistry (model + scaler)
    feature_extractor.py → provides the features dict passed in here
"""

import os
import numpy as np
import pandas as pd
import shap

from dataclasses import dataclass
from typing import Optional
from sklearn.model_selection import train_test_split

# ── paths ─────────────────────────────────────────────────────────────────────

DATA_PATH        = os.path.join(os.path.dirname(__file__), "..", "data", "phishing.csv")
DROPPED_FEATURES = ["web_traffic", "Links_pointing_to_page"]
TOP_N            = 5   # number of features to return in all explanations


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class SHAPFeature:
    """A single feature's SHAP contribution."""
    name       : str
    value      : int          # raw feature value (-1, 0, or 1)
    shap_value : float        # SHAP contribution (positive = pushes toward phishing)
    direction  : str          # "phishing" | "legitimate" | "neutral"
    label      : str          # human-readable feature label for the frontend


@dataclass
class ExplanationResult:
    """
    Returned to the FastAPI endpoint and passed to the frontend
    SHAPExplanation.jsx component.
    """
    # Local explanation (per-prediction)
    local_features  : list[SHAPFeature]   # top-5 features for this URL
    base_value      : float               # SHAP expected value (baseline log-odds)
    prediction_delta: float               # sum of all SHAP values for this sample

    # Global explanation (model-level, cached at startup)
    global_features : list[SHAPFeature]  # top-5 most important features overall

    # Metadata
    model_used      : str
    explanation_type: str = "tree"        # "tree" for RF/XGB, "linear" for LR


# ── human-readable feature labels ─────────────────────────────────────────────
# Maps raw feature names (UCI format) to clean display labels for the UI

FEATURE_LABELS = {
    "having_IP_Address"          : "IP Address in URL",
    "URL_Length"                 : "URL Length",
    "Shortining_Service"         : "URL Shortener Used",
    "having_At_Symbol"           : "@ Symbol in URL",
    "double_slash_redirecting"   : "Double Slash Redirect",
    "Prefix_Suffix"              : "Hyphen in Domain",
    "having_Sub_Domain"          : "Subdomain Depth",
    "SSLfinal_State"             : "SSL Certificate",
    "HTTPS_token"                : "HTTPS in Domain Name",
    "Domain_registeration_length": "Domain Registration Length",
    "port"                       : "Non-Standard Port",
    "Request_URL"                : "External Resource Ratio",
    "URL_of_Anchor"              : "Suspicious Anchor Links",
    "Links_in_tags"              : "External Links in Tags",
    "SFH"                        : "Form Action Handler",
    "Submitting_to_email"        : "Form Submits to Email",
    "Abnormal_URL"               : "URL Matches WHOIS Domain",
    "Redirect"                   : "Double Slash Redirects",
    "on_mouseover"               : "Mouseover Status Spoofing",
    "RightClick"                 : "Right-Click Disabled",
    "popUpWidnow"                : "Popup Window Present",
    "Iframe"                     : "Invisible iFrame",
    "age_of_domain"              : "Domain Age",
    "DNSRecord"                  : "DNS Record Exists",
    "Favicon"                    : "External Favicon",
    "Google_Index"               : "Google Safe Browsing",
    "Statistical_report"         : "PhishTank Blacklist",
    "Page_Rank"                  : "Page Rank Score",
}


# ── SHAP explainer cache ───────────────────────────────────────────────────────

# Global explanation is expensive to compute — cache it after first call
_global_cache: Optional[ExplanationResult] = None


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_explainer(model, model_name: str):
    """
    Build the appropriate SHAP explainer for the model type.
    - TreeExplainer  for Random Forest and XGBoost (fast, exact)
    - LinearExplainer for Logistic Regression
    """
    if model_name == "Logistic Regression":
        return shap.LinearExplainer(model, shap.maskers.Independent(
            np.zeros((1, 28))
        )), "linear"
    else:
        return shap.TreeExplainer(model), "tree"


def _direction(shap_val: float) -> str:
    """Convert a SHAP value to a direction label for frontend colouring."""
    if shap_val > 0.01:
        return "phishing"
    if shap_val < -0.01:
        return "legitimate"
    return "neutral"


def _build_shap_features(
    feature_names : list[str],
    feature_values: list,
    shap_values   : np.ndarray,
    top_n         : int = TOP_N,
) -> list[SHAPFeature]:
    """
    Sort features by absolute SHAP value and return the top_n as
    a list of SHAPFeature objects.
    """
    indexed = sorted(
        enumerate(shap_values),
        key=lambda x: float(np.abs(np.asarray(x[1])).flatten()[0]),
        reverse=True,
    )[:top_n]

    return [
        SHAPFeature(
            name       = feature_names[i],
            value      = int(np.asarray(feature_values[i]).flatten()[0]),
            shap_value = round(float(np.asarray(shap_values[i]).flatten()[0]), 4),
            direction  = _direction(float(np.asarray(shap_values[i]).flatten()[0])),
            label      = FEATURE_LABELS.get(feature_names[i], feature_names[i]),
        )
        for i, _ in indexed
    ]


def _load_validation_data():
    """
    Load and recreate the same validation split used in training.
    Used for global SHAP summary computation.
    Returns (X_val, feature_names) or (None, None) if data unavailable.
    """
    if not os.path.exists(DATA_PATH):
        return None, None

    df = pd.read_csv(DATA_PATH)
    df["Result"] = df["Result"].replace(-1, 0)
    df = df.drop(columns=DROPPED_FEATURES, errors="ignore")

    X = df.drop("Result", axis=1)
    y = df["Result"]

    _, X_val, _, _ = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    return X_val, X_val.columns.tolist()


# ── public API ────────────────────────────────────────────────────────────────

def explain_local(features: dict, registry) -> ExplanationResult:
    """
    Generate a local (per-prediction) SHAP explanation.

    Parameters
    ----------
    features : dict
        Raw feature dict from feature_extractor.extract_features().
        Must contain all 28 feature keys.
    registry : ModelRegistry
        The loaded ModelRegistry instance from predictor.py.
        Provides the best model and scaler.

    Returns
    -------
    ExplanationResult with local_features (top 5) and cached global_features.
    """
    model      = registry.best_model
    model_name = registry.best_model_name

    # Build input — single row DataFrame
    df_input       = pd.DataFrame([features])
    feature_names  = df_input.columns.tolist()
    feature_values = df_input.values[0]

    # Scale if Logistic Regression
    if model_name == "Logistic Regression":
        X_input = registry.scaler.transform(df_input)
    else:
        X_input = df_input.values

    # Compute SHAP values
    explainer, expl_type = _get_explainer(model, model_name)
    shap_vals = explainer.shap_values(X_input)

    # For binary classifiers, shap_values may return a list [class_0, class_1]
    # We want class_1 (legitimate = 1) contributions
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    # Single-sample: shap_vals is shape (1, n_features) → flatten
    shap_vals_flat = shap_vals[0] if shap_vals.ndim == 2 else shap_vals

    local_feats = _build_shap_features(
        feature_names  = feature_names,
        feature_values = feature_values,
        shap_values    = shap_vals_flat,
        top_n          = TOP_N,
    )

    # Base value — scalar for tree models, array for linear
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[1]) if len(base_val) > 1 else float(base_val[0])
    else:
        base_val = float(base_val)

    # Ensure global cache is populated
    global_feats = _get_global_features(registry)

    return ExplanationResult(
        local_features   = local_feats,
        base_value       = round(base_val, 4),
        prediction_delta = round(float(shap_vals_flat.sum()), 4),
        global_features  = global_feats,
        model_used       = model_name,
        explanation_type = expl_type,
    )


def explain_global(registry) -> list[SHAPFeature]:
    """
    Generate global SHAP feature importance by running TreeExplainer
    on a sample of the validation set.

    Called once at FastAPI startup and cached — subsequent calls return
    the cached result immediately.

    Parameters
    ----------
    registry : ModelRegistry
        Loaded ModelRegistry from predictor.py.

    Returns
    -------
    List of top-5 SHAPFeature objects representing global feature importance.
    """
    return _get_global_features(registry)


def _get_global_features(registry) -> list[SHAPFeature]:
    """Internal — compute or return cached global SHAP features."""
    global _global_cache

    if _global_cache is not None:
        return _global_cache.global_features

    model      = registry.best_model
    model_name = registry.best_model_name

    X_val, feature_names = _load_validation_data()

    if X_val is None:
        # Dataset unavailable at runtime — return empty list gracefully
        return []

    # Use a sample of 200 rows for speed (sufficient for global importance)
    sample_size = min(200, len(X_val))
    X_sample    = X_val.sample(n=sample_size, random_state=42)

    if model_name == "Logistic Regression":
        X_input = registry.scaler.transform(X_sample)
    else:
        X_input = X_sample.values

    explainer, _ = _get_explainer(model, model_name)
    shap_vals    = explainer.shap_values(X_input)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    # Global importance = mean absolute SHAP value across all samples
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)

    global_feats = _build_shap_features(
        feature_names  = feature_names,
        feature_values = mean_abs_shap,   # use mean |SHAP| as the "value" for display
        shap_values    = mean_abs_shap,
        top_n          = TOP_N,
    )

    # Patch direction — global importance has no direction, mark all neutral
    for f in global_feats:
        f.direction = "neutral"
        f.value     = 0         # not meaningful for global; reset to avoid confusion

    # Store in cache — wrap in a minimal ExplanationResult shell
    _global_cache = ExplanationResult(
        local_features   = [],
        base_value       = 0.0,
        prediction_delta = 0.0,
        global_features  = global_feats,
        model_used       = model_name,
    )

    return global_feats


def clear_global_cache():
    """
    Clear the global SHAP cache — call this if models are retrained
    via the feedback/retraining mechanism in feedback.py.
    """
    global _global_cache
    _global_cache = None
    print("[explainer] Global SHAP cache cleared.")