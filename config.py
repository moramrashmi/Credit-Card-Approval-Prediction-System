"""
config.py — Centralised project configuration
==============================================
All paths, hyperparameters, feature lists, and constants live here.
Importing this module in any other file keeps the project DRY and ensures
a single source of truth for configuration changes.
"""

import os
from pathlib import Path

# ─── Project Root ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# ─── Directory Paths ──────────────────────────────────────────────────────────
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR  = BASE_DIR / "models"
STATIC_DIR  = BASE_DIR / "static"
PLOTS_DIR   = STATIC_DIR / "images" / "plots"

# ─── Dataset ──────────────────────────────────────────────────────────────────
DATASET_URL = "rikdifos/credit-card-approval-prediction"
APP_DATA_PATH = DATASET_DIR / "application_record.csv"
CREDIT_DATA_PATH = DATASET_DIR / "credit_record.csv"

TARGET_COLUMN = "ApprovalStatus"

# ─── Model Artifact Paths ─────────────────────────────────────────────────────
BEST_MODEL_PATH   = MODELS_DIR / "best_model.pkl"
PIPELINE_PATH     = MODELS_DIR / "pipeline.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"

# ─── Feature Groups ───────────────────────────────────────────────────────────
# Raw categorical columns (before encoding)
CATEGORICAL_FEATURES = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE"
]

# Raw numeric columns (before scaling)
NUMERIC_FEATURES = [
    "CNT_CHILDREN", "AMT_INCOME_TOTAL", "DAYS_BIRTH", "DAYS_EMPLOYED",
    "FLAG_MOBIL", "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL", "CNT_FAM_MEMBERS"
]

# Engineered feature names added in preprocessing
ENGINEERED_FEATURES = [
    "IS_EMPLOYED",
    "YEARS_EMPLOYED",
    "AGE_YEARS",
    "EMPLOYMENT_STABILITY",
    "INCOME_GROUP",
    "INCOME_PER_FAM_MEMBER",
    "FINANCIAL_STABILITY_SCORE"
]

# ─── Training Hyperparameters ─────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

# XGBoost grid-search parameter grid (kept small for speed)
XGB_PARAM_GRID = {
    "n_estimators":     [100, 200],
    "max_depth":        [3, 5],
    "learning_rate":    [0.05, 0.1],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# ─── Flask ────────────────────────────────────────────────────────────────────
FLASK_SECRET_KEY = os.environ.get("SECRET_KEY", "credit-approval-secret-key-2024")
FLASK_DEBUG      = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
PORT             = int(os.environ.get("PORT", 5000))

# ─── Risk Thresholds ──────────────────────────────────────────────────────────
# These probability thresholds map approval probability → risk label
RISK_HIGH_THRESHOLD   = 0.40   # below → High Risk
RISK_MEDIUM_THRESHOLD = 0.70   # below → Medium Risk, above → Low Risk

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE   = "%Y-%m-%d %H:%M:%S"
