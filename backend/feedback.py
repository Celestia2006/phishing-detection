"""
feedback.py
-----------
Handles user feedback collection and manual model retraining.

Two responsibilities:
  1. Collect — save user corrections (URL + features + correct label)
               to a growing feedback CSV file
  2. Retrain  — when manually triggered via a FastAPI endpoint,
                retrain all three models on original data + feedback data
                and replace the saved .pkl files

Feedback file
-------------
Corrections are appended to:
    ../data/feedback.csv

Each row contains:
  - timestamp       : when the feedback was submitted
  - url             : the URL that was scanned
  - predicted_label : what the model said ("phishing" / "legitimate")
  - correct_label   : what the user says it actually is
  - all 28 feature columns (same as training data)

Retraining
----------
Triggered manually via POST /retrain in main.py.
Retrains Logistic Regression, Random Forest, and XGBoost on:
    original phishing.csv  +  all accepted feedback rows
Overwrites the existing .pkl files in ../backend/models/.
Clears the SHAP global cache so explanations reflect the new model.

Usage (called from main.py)
---------------------------
    from feedback import save_feedback, retrain_models, FeedbackEntry

    # On POST /feedback
    save_feedback(entry)

    # On POST /retrain
    result = retrain_models()
"""

import os
import time
import joblib
import numpy as np
import pandas as pd

from datetime import datetime
from dataclasses import dataclass
from typing import Optional

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

# ── paths ─────────────────────────────────────────────────────────────────────

BASE_DIR         = os.path.dirname(__file__)
DATA_PATH        = os.path.join(BASE_DIR, "..", "data", "phishing.csv")
FEEDBACK_PATH    = os.path.join(BASE_DIR, "..", "data", "feedback.csv")
MODELS_DIR       = os.path.join(BASE_DIR, "models")

DROPPED_FEATURES = ["web_traffic", "Links_pointing_to_page"]

# ── feature columns — must match FEATURE_ORDER in feature_extractor.py ────────

FEATURE_COLUMNS = [
    "having_IP_Address", "URL_Length", "Shortining_Service",
    "having_At_Symbol", "double_slash_redirecting", "Prefix_Suffix",
    "having_Sub_Domain", "SSLfinal_State", "Domain_registeration_length",
    "Favicon", "port", "HTTPS_token", "Request_URL",
    "URL_of_Anchor", "Links_in_tags", "SFH", "Submitting_to_email",
    "Abnormal_URL", "Redirect", "on_mouseover", "RightClick",
    "popUpWidnow", "Iframe", "age_of_domain", "DNSRecord",
    "Page_Rank", "Google_Index", "Statistical_report",
]


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class FeedbackEntry:
    """
    A single user correction submitted via the frontend.
    Received as a JSON body on POST /feedback in main.py.
    """
    url             : str
    predicted_label : str          # "phishing" or "legitimate" — what model said
    correct_label   : str          # "phishing" or "legitimate" — what user says
    features        : dict         # raw feature dict from PredictionResult.features


@dataclass
class RetrainResult:
    """
    Summary of a completed retraining run.
    Returned to the frontend after POST /retrain.
    """
    success           : bool
    message           : str
    feedback_count    : int                  # how many feedback rows were used
    training_size     : int                  # total rows trained on
    new_best_model    : Optional[str]        # name of newly selected best model
    f1_scores         : dict                 # {model_name: f1_score} after retraining
    duration_seconds  : float               # how long retraining took
    error             : Optional[str] = None


# ── feedback storage ──────────────────────────────────────────────────────────

def save_feedback(entry: FeedbackEntry) -> bool:
    """
    Append a user correction to feedback.csv.

    Creates the file with a header row on first call.
    Each subsequent call appends a single row — no full file rewrite.

    Parameters
    ----------
    entry : FeedbackEntry
        The user's correction with URL, labels, and extracted features.

    Returns
    -------
    True on success, False on failure.
    """
    try:
        # Build the row — metadata + all feature values
        row = {
            "timestamp"       : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "url"             : entry.url,
            "predicted_label" : entry.predicted_label,
            "correct_label"   : entry.correct_label,
        }

        # Attach all feature values in the canonical column order
        for col in FEATURE_COLUMNS:
            row[col] = entry.features.get(col, 0)

        df_row = pd.DataFrame([row])

        # Append mode — write header only if file doesn't exist yet
        write_header = not os.path.exists(FEEDBACK_PATH)
        df_row.to_csv(
            FEEDBACK_PATH,
            mode   = "a",
            header = write_header,
            index  = False,
        )

        print(f"[feedback] Saved correction for: {entry.url}")
        return True

    except Exception as e:
        print(f"[feedback] Failed to save feedback: {e}")
        return False


def get_feedback_count() -> int:
    """
    Return the number of feedback entries currently stored.
    Returns 0 if the feedback file does not exist yet.
    """
    if not os.path.exists(FEEDBACK_PATH):
        return 0
    try:
        df = pd.read_csv(FEEDBACK_PATH)
        return len(df)
    except Exception:
        return 0


def get_feedback_summary() -> dict:
    """
    Return a summary of stored feedback for the /feedback/summary endpoint.
    Useful for the team to monitor how much feedback has accumulated.
    """
    if not os.path.exists(FEEDBACK_PATH):
        return {
            "total"           : 0,
            "phishing_corrections"    : 0,
            "legitimate_corrections"  : 0,
            "message"         : "No feedback collected yet.",
        }
    try:
        df = pd.read_csv(FEEDBACK_PATH)
        phishing_corrections   = len(df[df["correct_label"] == "phishing"])
        legitimate_corrections = len(df[df["correct_label"] == "legitimate"])
        return {
            "total"                   : len(df),
            "phishing_corrections"    : phishing_corrections,
            "legitimate_corrections"  : legitimate_corrections,
            "message"                 : f"{len(df)} correction(s) stored and ready for retraining.",
        }
    except Exception as e:
        return {"total": 0, "error": str(e)}


# ── retraining ────────────────────────────────────────────────────────────────

def _load_combined_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load and combine original training data with feedback corrections.

    Feedback rows use the correct_label column as the ground truth target,
    replacing whatever the model originally predicted.

    Returns
    -------
    (X, y) — features DataFrame and target Series, ready for training.
    """
    # Load original dataset
    df_original = pd.read_csv(DATA_PATH)
    df_original["Result"] = df_original["Result"].replace(-1, 0)
    df_original = df_original.drop(columns=DROPPED_FEATURES, errors="ignore")

    if not os.path.exists(FEEDBACK_PATH):
        # No feedback yet — train on original data only
        X = df_original.drop("Result", axis=1)
        y = df_original["Result"]
        return X, y

    # Load feedback
    df_feedback = pd.read_csv(FEEDBACK_PATH)

    if df_feedback.empty:
        X = df_original.drop("Result", axis=1)
        y = df_original["Result"]
        return X, y

    # Convert correct_label string → binary int to match training format
    df_feedback["Result"] = df_feedback["correct_label"].map({
        "legitimate" : 1,
        "phishing"   : 0,
    })

    # Keep only feature columns + Result — drop metadata columns
    feedback_cols   = FEATURE_COLUMNS + ["Result"]
    df_feedback_clean = df_feedback[
        [c for c in feedback_cols if c in df_feedback.columns]
    ].dropna(subset=["Result"])

    # Combine original + feedback
    df_combined = pd.concat([df_original, df_feedback_clean], ignore_index=True)

    X = df_combined.drop("Result", axis=1)
    y = df_combined["Result"]

    return X, y


def retrain_models() -> RetrainResult:
    """
    Retrain all three models on original data + user feedback.

    Steps:
      1. Load and combine datasets
      2. Train/test split (same params as original training)
      3. Fit Logistic Regression, Random Forest, XGBoost
      4. Evaluate each on test split by F1 score
      5. Overwrite .pkl files with newly trained models
      6. Clear SHAP global cache so explanations stay accurate
      7. Reset ModelRegistry so predictor.py picks up new models

    Returns
    -------
    RetrainResult with full summary of what happened.

    Note
    ----
    This function is intentionally synchronous — for a student project
    the training time is short enough. For production, wrap in a
    background task (FastAPI BackgroundTasks or Celery).
    """
    start_time = time.time()

    try:
        # ── Step 1 — Load data ────────────────────────────────────────────────
        X, y = _load_combined_data()
        feedback_count = get_feedback_count()
        training_size  = len(X)

        print(f"[retrain] Starting retraining on {training_size} samples "
              f"({feedback_count} from feedback)...")

        # ── Step 2 — Train/test split ─────────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size    = 0.3,
            random_state = 42,
            stratify     = y,
        )

        # ── Step 3 — Fit scaler ───────────────────────────────────────────────
        scaler        = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # ── Step 4 — Train all three models ───────────────────────────────────
        models_to_train = {
            "Logistic Regression": LogisticRegression(
                max_iter     = 1000,
                random_state = 42,
                C            = 1.0,
                solver       = "lbfgs",
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators = 100,
                random_state = 42,
                n_jobs       = -1,
            ),
            "XGBoost": XGBClassifier(
                n_estimators     = 100,
                max_depth        = 6,
                learning_rate    = 0.1,
                subsample        = 0.8,
                colsample_bytree = 0.8,
                eval_metric      = "logloss",
                random_state     = 42,
                n_jobs           = -1,
            ),
        }

        trained_models = {}
        f1_scores      = {}

        for name, model in models_to_train.items():
            print(f"[retrain] Training {name}...")
            X_tr = X_train_scaled if name == "Logistic Regression" else X_train
            X_te = X_test_scaled  if name == "Logistic Regression" else X_test

            model.fit(X_tr, y_train)
            y_pred         = model.predict(X_te)
            score          = f1_score(y_test, y_pred)
            trained_models[name] = model
            f1_scores[name]      = round(score, 4)
            print(f"[retrain]   {name} F1 = {score:.4f}")

        # ── Step 5 — Select best model ────────────────────────────────────────
        best_name = max(f1_scores, key=f1_scores.get)
        print(f"[retrain] Best model after retraining: {best_name} "
              f"(F1 = {f1_scores[best_name]:.4f})")

        # ── Step 6 — Save new .pkl files ──────────────────────────────────────
        os.makedirs(MODELS_DIR, exist_ok=True)

        joblib.dump(trained_models["Logistic Regression"],
                    os.path.join(MODELS_DIR, "model_lr.pkl"))
        joblib.dump(trained_models["Random Forest"],
                    os.path.join(MODELS_DIR, "model_rf.pkl"))
        joblib.dump(trained_models["XGBoost"],
                    os.path.join(MODELS_DIR, "model_xgb.pkl"))
        joblib.dump(trained_models[best_name],
                    os.path.join(MODELS_DIR, "model_best.pkl"))
        joblib.dump(scaler,
                    os.path.join(MODELS_DIR, "scaler.pkl"))

        print("[retrain] All models saved to ../backend/models/")

        # ── Step 7 — Clear caches so new models are picked up ─────────────────
        # Clear SHAP global cache
        try:
            from explainer import clear_global_cache
            clear_global_cache()
        except ImportError:
            pass

        # Reset ModelRegistry singleton so predictor reloads fresh models
        try:
            import predictor
            predictor.MODULE_REGISTRY = None
            print("[retrain] ModelRegistry reset — will reload on next request.")
        except ImportError:
            pass

        duration = round(time.time() - start_time, 2)
        print(f"[retrain] Retraining complete in {duration}s")

        return RetrainResult(
            success          = True,
            message          = (
                f"Retraining complete. Best model: {best_name} "
                f"(F1 = {f1_scores[best_name]:.4f}). "
                f"Trained on {training_size} samples "
                f"({feedback_count} from user feedback)."
            ),
            feedback_count   = feedback_count,
            training_size    = training_size,
            new_best_model   = best_name,
            f1_scores        = f1_scores,
            duration_seconds = duration,
        )

    except Exception as e:
        duration = round(time.time() - start_time, 2)
        print(f"[retrain] Retraining failed: {e}")
        return RetrainResult(
            success          = False,
            message          = "Retraining failed. See error for details.",
            feedback_count   = get_feedback_count(),
            training_size    = 0,
            new_best_model   = None,
            f1_scores        = {},
            duration_seconds = duration,
            error            = str(e),
        )