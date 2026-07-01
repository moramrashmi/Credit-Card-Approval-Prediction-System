"""
preprocessing.py — Data loading, cleaning, feature engineering, and pipeline
=============================================================================
Design decisions
----------------
* Merges Kaggle application_record.csv and credit_record.csv based on ID.
* Transforms multiple payment statuses into a binary Approved/Rejected target.
* All transformations are wrapped in sklearn Pipelines / ColumnTransformers
  so the exact same object can be applied during training AND inference — no
  risk of train/test leakage or inconsistent feature order.
* Feature engineering is a custom sklearn BaseEstimator. It is stateful, learning
  percentile thresholds and max limits from the training data.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from config import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, ENGINEERED_FEATURES,
    APP_DATA_PATH, CREDIT_DATA_PATH, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE,
)
from utils import get_logger, summarise_dataframe

logger = get_logger(__name__)


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_raw_data() -> pd.DataFrame:
    """
    Load Kaggle Credit Card Approval dataset.
    Merges application and credit records and binarises the target.
    """
    logger.info("Loading dataset files...")
    app_df = pd.read_csv(APP_DATA_PATH)
    credit_df = pd.read_csv(CREDIT_DATA_PATH)

    # Convert status to binary: 
    # High-risk (0): '1', '2', '3', '4', '5'
    # Good-credit (1): 'C', 'X', '0'
    credit_df['IsBad'] = credit_df['STATUS'].isin(['1', '2', '3', '4', '5']).astype(int)
    
    # Group by ID, if sum > 0, then Bad (0), else Good (1)
    target_df = credit_df.groupby('ID')['IsBad'].sum().reset_index()
    target_df[TARGET_COLUMN] = (target_df['IsBad'] == 0).astype(int)
    target_df = target_df.drop(columns=['IsBad'])

    # Merge records
    df = pd.merge(app_df, target_df, on='ID', how='inner')

    # Remove duplicates
    duplicates = df.duplicated(subset=['ID']).sum()
    if duplicates > 0:
        logger.info(f"Removing {duplicates} duplicate IDs.")
        df = df.drop_duplicates(subset=['ID'], keep='first')
        
    df = df.drop(columns=['ID'])
    
    logger.info("Raw data loaded and merged.")
    summarise_dataframe(df)
    return df


# ─── Feature Engineering Transformer ─────────────────────────────────────────

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom sklearn transformer that creates domain-specific features.
    It is stateful and learns parameters from the training set to prevent leakage.
    """

    def __init__(self):
        self.income_bins_ = None
        self.max_financial_score_ = 1.0

    def fit(self, X, y=None):
        X = X.copy()
        
        # Learn thresholds for income buckets (33, 66 percentiles)
        income = pd.to_numeric(X["AMT_INCOME_TOTAL"], errors="coerce").fillna(0)
        self.income_bins_ = [-np.inf, np.percentile(income, 33), np.percentile(income, 66), np.inf]
        
        # Dummy transform to find the max financial score for scaling
        X_tmp = self._transform_features(X)
        self.max_financial_score_ = X_tmp["FINANCIAL_STABILITY_SCORE"].max()
        if pd.isna(self.max_financial_score_) or self.max_financial_score_ <= 0:
            self.max_financial_score_ = 1.0
            
        return self

    def transform(self, X):
        X_out = self._transform_features(X)
        
        # Risk score (inverse of stability; clipped to [0, 1])
        X_out["RiskScore"] = 1 - (X_out["FINANCIAL_STABILITY_SCORE"] / self.max_financial_score_)
        X_out["RiskScore"] = X_out["RiskScore"].clip(0, 1)
        
        return X_out

    def _transform_features(self, X):
        X = X.copy()

        # Guard: coerce numeric features
        for col in ["DAYS_BIRTH", "DAYS_EMPLOYED", "AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"]:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

        # 1. Age in years
        X["AGE_YEARS"] = np.abs(X["DAYS_BIRTH"]) / 365.25
        
        # 2. Employment
        # DAYS_EMPLOYED is positive (365243) if unemployed, negative days if employed
        X["IS_EMPLOYED"] = (X["DAYS_EMPLOYED"] < 0).astype(int)
        X["YEARS_EMPLOYED"] = np.where(X["DAYS_EMPLOYED"] < 0, np.abs(X["DAYS_EMPLOYED"]) / 365.25, 0)
        
        # 3. Employment Stability
        X["EMPLOYMENT_STABILITY"] = np.log1p(X["YEARS_EMPLOYED"]) * X["IS_EMPLOYED"]
        
        # 4. Income Category
        if self.income_bins_ is not None:
            X["INCOME_GROUP"] = pd.cut(
                X["AMT_INCOME_TOTAL"],
                bins=self.income_bins_,
                labels=[0, 1, 2]
            ).astype(float)
        else:
            X["INCOME_GROUP"] = 0.0

        # 5. Income per family member
        X["INCOME_PER_FAM_MEMBER"] = X["AMT_INCOME_TOTAL"] / X["CNT_FAM_MEMBERS"].clip(lower=1)
        
        # 6. Financial stability score
        X["FINANCIAL_STABILITY_SCORE"] = (
            X["INCOME_PER_FAM_MEMBER"] * 0.4 + 
            X["EMPLOYMENT_STABILITY"] * 1000 * 0.6
        )
        
        return X


# ─── Preprocessing Pipeline Builder ──────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - imputes + scales numeric features
      - imputes + ordinal-encodes categorical features
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",  numeric_pipeline,      NUMERIC_FEATURES + ENGINEERED_FEATURES + ["RiskScore"]),
            ("cat",  categorical_pipeline,  CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor

def build_full_pipeline(classifier) -> Pipeline:
    """
    Builds the complete sklearn pipeline from raw data to classifier.
    """
    return Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier)
    ])


# ─── Public API ───────────────────────────────────────────────────────────────

def prepare_data():
    """
    Load data and split.
    Feature engineering is now handled inside the Pipeline during training.
    """
    df = load_raw_data()
    df.dropna(subset=[TARGET_COLUMN], inplace=True)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info(
        "Split — Train: %d rows | Test: %d rows | Positive rate (train): %.2f%%",
        len(X_train), len(X_test), y_train.mean() * 100,
    )
    return X_train, X_test, y_train, y_test
