"""
train.py — Model training, evaluation, and artifact saving
===========================================================
Trains four classifiers inside identical sklearn Pipelines so that the
preprocessor is always coupled with the model. The best model (selected
by cross-validation ROC-AUC) is saved to disk.

Run:
    python train.py
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config import (
    BEST_MODEL_PATH, PIPELINE_PATH, CV_FOLDS, FEATURE_NAMES_PATH, MODELS_DIR,
    RANDOM_STATE, XGB_PARAM_GRID,
)
from preprocessing import build_full_pipeline, prepare_data
from utils import (
    evaluate_model, get_logger,
    plot_confusion_matrix, plot_roc_curve, plot_feature_importance,
)

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


# ─── Model Definitions ────────────────────────────────────────────────────────

def get_candidate_models(y_train) -> dict:
    """
    Return dict of {model_name: unfitted estimator}.
    """
    scale_pos_weight = 1
    if y_train is not None:
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        if pos_count > 0:
            scale_pos_weight = neg_count / pos_count

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
        ),
    }


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_all_models(X_train, X_test, y_train, y_test) -> tuple:
    """
    Fit every candidate model inside a full Pipeline.
    Evaluate each using Cross-Validation on training set, then evaluate on test set.
    """
    candidates = get_candidate_models(y_train)
    results = []
    trained_pipelines = {}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, clf in candidates.items():
        logger.info("─── Training: %s ───", name)

        pipe = build_full_pipeline(clf)

        # Cross-validation on training set
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                    scoring="roc_auc", n_jobs=-1)
        logger.info("%s CV ROC-AUC: %.4f ± %.4f", name, cv_scores.mean(), cv_scores.std())

        # Full fit on training data
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

        # Test-set evaluation
        metrics = evaluate_model(pipe, X_test, y_test, model_name=name)
        metrics["cv_roc_auc_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_roc_auc_std"]  = round(cv_scores.std(), 4)
        results.append(metrics)

        # Visualisations
        y_pred = pipe.predict(X_test)
        plot_confusion_matrix(
            y_test, y_pred,
            labels=["Rejected", "Approved"],
            filename=f"cm_{name.replace(' ', '_').lower()}.png",
        )
        plot_roc_curve(
            pipe, X_test, y_test,
            model_name=name,
            filename=f"roc_{name.replace(' ', '_').lower()}.png",
        )

    # Sort by cross-validation performance to avoid test-set leakage
    results_df = pd.DataFrame(results).set_index("model").sort_values("cv_roc_auc_mean", ascending=False)
    logger.info("\n\n=== Model Comparison ===\n%s\n", results_df.to_string())
    return results_df, trained_pipelines


# ─── Save Artifacts ───────────────────────────────────────────────────────────

def save_artifacts(best_pipeline: Pipeline, feature_names: list, best_name: str) -> None:
    """
    Save the complete fitted Pipeline.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    logger.info("Best pipeline saved → %s", BEST_MODEL_PATH)

    joblib.dump(feature_names, FEATURE_NAMES_PATH)
    logger.info("Feature names saved → %s", FEATURE_NAMES_PATH)
    
    # Save best model name for UI
    with open(MODELS_DIR / "best_model_name.txt", "w") as f:
        f.write(best_name)


# ─── Feature Importance Plot ──────────────────────────────────────────────────

def save_feature_importance_plot(pipeline: Pipeline) -> None:
    """Extract feature importances from the best model and save a bar chart."""
    clf = pipeline.named_steps["classifier"]
    
    # Get feature names from the preprocessor output if possible
    try:
        # Sklearn 1.2+ supports get_feature_names_out out of the box for most transformers
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    except Exception:
        # Fallback if get_feature_names_out fails
        logger.warning("Could not extract feature names for importance plot. Using default names.")
        # Just use generic names
        num_features = getattr(clf, "feature_importances_", getattr(clf, "coef_", [[]])[0])
        feature_names = [f"Feature_{i}" for i in range(len(num_features))]

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        plot_feature_importance(feature_names, importances,
                                 filename="feature_importance.png")
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
        plot_feature_importance(feature_names, importances,
                                 filename="feature_importance.png")


# ─── Results Persistence ──────────────────────────────────────────────────────

def save_results(results_df: pd.DataFrame) -> None:
    """Save the model comparison table as JSON for the Flask UI."""
    results_path = MODELS_DIR / "results.json"
    results_df.reset_index().to_json(results_path, orient="records", indent=2)
    logger.info("Results saved → %s", results_path)


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("══════════════════════════════════════════")
    logger.info("  Credit Card Approval — Model Training   ")
    logger.info("══════════════════════════════════════════")

    # 1. Load and split data
    X_train, X_test, y_train, y_test = prepare_data()

    # Feature names
    feature_names = list(X_train.columns)

    # 2. Train all candidate models
    results_df, trained_pipelines = train_all_models(X_train, X_test, y_train, y_test)

    # 3. Select best by CV ROC-AUC
    best_name = results_df.index[0]
    logger.info("Best model: %s  (CV ROC-AUC = %.4f)", best_name, results_df.loc[best_name, "cv_roc_auc_mean"])
    best_pipeline = trained_pipelines[best_name]

    # 5. Final evaluation of best model
    logger.info("─── Final evaluation of best model ───")
    evaluate_model(best_pipeline, X_test, y_test, model_name=f"{best_name} (final)")

    # 6. Save artifacts
    save_artifacts(best_pipeline, feature_names, best_name)
    save_feature_importance_plot(best_pipeline)
    save_results(results_df)

    logger.info("══════════════════════════════════════════")
    logger.info("  Training complete. Artifacts saved.     ")
    logger.info("══════════════════════════════════════════")


if __name__ == "__main__":
    main()
