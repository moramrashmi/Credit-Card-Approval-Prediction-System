"""
predict.py — Standalone prediction module
==========================================
This module is imported by app.py to serve predictions. It:
  1. Loads model artifacts once at module level (no repeated disk I/O).
  2. Accepts a raw dict of user inputs.
  3. Applies the single pipeline (FeatureEngineering + Preprocessing + Classifier).
  4. Returns probability, binary label, and human-readable risk level.
"""

import joblib
import pandas as pd
import traceback

from config import BEST_MODEL_PATH, FEATURE_NAMES_PATH
from utils import get_logger, risk_label

logger = get_logger(__name__)


# ─── One-time artifact loading ────────────────────────────────────────────────

def _load_artifacts():
    """Load model pipeline and feature names from disk. Called once at import."""
    try:
        pipeline      = joblib.load(BEST_MODEL_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        logger.info("Model pipeline loaded from %s", BEST_MODEL_PATH)
        return pipeline, feature_names
    except FileNotFoundError as exc:
        logger.error(
            "Model artifacts not found. Run `python train.py` first.\n%s", exc
        )
        # Return None to allow app to start even if models aren't trained yet
        return None, None


_pipeline, _feature_names = _load_artifacts()


# ─── Public API ───────────────────────────────────────────────────────────────

def predict(raw_input: dict) -> dict:
    """
    Generate a credit approval prediction for a single applicant.
    """
    if _pipeline is None:
        raise RuntimeError("Model pipeline not loaded. Run `python train.py` first.")

    # 1. Build a single-row DataFrame
    df_input = pd.DataFrame([raw_input])

    # 2. Ensure columns are in the same order as training
    # Add any missing columns with 0/empty string
    for col in _feature_names:
        if col not in df_input.columns:
            df_input[col] = ""

    df_input = df_input[_feature_names]

    # 3. Run prediction through the full pipeline
    try:
        prob_approved = float(_pipeline.predict_proba(df_input)[0][1])
        approved      = bool(_pipeline.predict(df_input)[0])
    except Exception as e:
        logger.error("Error during prediction: %s", traceback.format_exc())
        raise ValueError(f"Prediction failed: {e}")

    # 4. Derive derived fields
    prob_pct   = round(prob_approved * 100, 1)
    risk       = risk_label(prob_approved)
    confidence = _confidence_description(prob_approved, approved)

    result = {
        "approved":       approved,
        "approval_label": "Approved" if approved else "Rejected",
        "probability":    prob_pct,
        "risk_level":     risk,
        "confidence":     confidence,
    }

    logger.info(
        "Prediction → %s | P(approve)=%.3f | Risk=%s",
        result["approval_label"], prob_approved, risk,
    )
    return result


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _confidence_description(prob: float, approved: bool) -> str:
    """Map probability to a plain-English confidence description."""
    if approved:
        if prob >= 0.90:
            return "Very High Confidence — Excellent credit profile"
        if prob >= 0.75:
            return "High Confidence — Strong financial indicators"
        if prob >= 0.60:
            return "Moderate Confidence — Acceptable risk profile"
        return "Low Confidence — Borderline case, manual review recommended"
    else:
        if prob <= 0.10:
            return "Very High Confidence — Poor credit profile"
        if prob <= 0.25:
            return "High Confidence — Weak financial indicators"
        if prob <= 0.40:
            return "Moderate Confidence — High-risk profile"
        return "Low Confidence — Borderline case, manual review recommended"
