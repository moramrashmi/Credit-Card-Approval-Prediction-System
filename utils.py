"""
utils.py — Shared utility functions
====================================
Provides logging setup, metric helpers, plot saving, and data-download
utilities. Every other module imports from here to avoid duplication.
"""

import logging
import sys
from pathlib import Path

from config import LOG_FORMAT, LOG_DATE, LOG_LEVEL, PLOTS_DIR


# ─── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a consistently formatted logger for any module.

    On Windows the default stdout stream uses cp1252 which cannot encode
    Unicode box-drawing characters. We wrap stdout in a UTF-8 TextIOWrapper
    so that all log messages render correctly on any platform.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        import io
        utf8_stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        ) if hasattr(sys.stdout, "buffer") else sys.stdout
        handler = logging.StreamHandler(utf8_stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    return logger


# ─── Dataset Download ─────────────────────────────────────────────────────────

def download_dataset(url: str, destination: Path) -> Path:
    """
    Download a CSV from *url* to *destination* if not already present.

    Returns the destination path so callers can chain operations.
    """
    import requests

    logger = get_logger(__name__)
    if destination.exists():
        logger.info("Dataset already exists at %s — skipping download.", destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading dataset from %s …", url)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    destination.write_bytes(response.content)
    logger.info("Saved dataset to %s  (%d bytes).", destination, len(response.content))
    return destination


# ─── Metrics ──────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Compute classification metrics for a fitted *model*.

    Returns a dict with accuracy, precision, recall, f1, roc_auc
    and also prints the full classification report.
    """
    from sklearn.metrics import (
        accuracy_score, classification_report,
        f1_score, precision_score, recall_score, roc_auc_score,
    )

    logger = get_logger(__name__)
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    metrics = {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else "N/A",
    }

    logger.info(
        "\n%s Results\n"
        "  Accuracy : %.4f\n"
        "  Precision: %.4f\n"
        "  Recall   : %.4f\n"
        "  F1       : %.4f\n"
        "  ROC-AUC  : %s",
        model_name,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["roc_auc"],
    )
    logger.info("\nClassification Report:\n%s", classification_report(y_test, y_pred))
    return metrics


# ─── Plot Helpers ─────────────────────────────────────────────────────────────

def _save_fig(filename: str) -> None:
    """Save current matplotlib figure to PLOTS_DIR and close it."""
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_confusion_matrix(y_test, y_pred, labels: list, filename: str) -> None:
    """Render and save a labelled confusion matrix heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12)
    _save_fig(filename)


def plot_roc_curve(model, X_test, y_test, model_name: str, filename: str) -> None:
    """Render and save a ROC curve for *model*."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay

    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=model_name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_title(f"ROC Curve — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    _save_fig(filename)


def plot_feature_importance(feature_names: list, importances,
                             top_n: int = 15, filename: str = "feature_importance.png") -> None:
    """Bar chart of top-N feature importances."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    indices = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in reversed(indices)],
        importances[indices[::-1]],
        color="steelblue",
    )
    ax.set_title("Top Feature Importances", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    _save_fig(filename)


# ─── DataFrame Helpers ────────────────────────────────────────────────────────

def summarise_dataframe(df) -> None:
    """Print a rich summary of a DataFrame (shape, dtypes, nulls, stats)."""
    logger = get_logger(__name__)
    logger.info("Shape: %s", df.shape)
    logger.info("Columns:\n%s", df.dtypes.to_string())
    logger.info("Null counts:\n%s", df.isnull().sum().to_string())
    logger.info("Duplicates: %d", df.duplicated().sum())
    logger.info("Target distribution:\n%s", df.iloc[:, -1].value_counts().to_string())


def risk_label(probability: float) -> str:
    """
    Map an approval probability to a human-readable risk category.

    probability < 0.40  → High Risk
    probability < 0.70  → Medium Risk
    probability >= 0.70 → Low Risk
    """
    from config import RISK_HIGH_THRESHOLD, RISK_MEDIUM_THRESHOLD
    if probability < RISK_HIGH_THRESHOLD:
        return "High Risk"
    if probability < RISK_MEDIUM_THRESHOLD:
        return "Medium Risk"
    return "Low Risk"
