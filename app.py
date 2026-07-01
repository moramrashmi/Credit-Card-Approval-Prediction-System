"""
app.py — Flask web application entry point
==========================================
Architecture decisions
----------------------
* The model is loaded ONCE at module level (via predict.py) — not on every
  request. This keeps response times fast and avoids repeated disk I/O.
* All form validation happens server-side before calling predict().
* Results are passed to result.html as a context dict.
"""

import logging
from pathlib import Path
import os

from flask import Flask, flash, redirect, render_template, request, url_for, send_from_directory

from config import FLASK_DEBUG, FLASK_SECRET_KEY, PORT, MODELS_DIR

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

# Configure Flask's built-in logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Lazy model loader ────────────────────────────────────────────────────────

_predict_fn = None

def _get_predict():
    """Import and cache the predict function on first use."""
    global _predict_fn
    if _predict_fn is None:
        try:
            from predict import predict as _fn
            _predict_fn = _fn
        except Exception:
            _predict_fn = None
    return _predict_fn


# ─── Input validation ─────────────────────────────────────────────────────────

VALID_GENDER         = {"M", "F"}
VALID_Y_N            = {"Y", "N"}
VALID_INCOME_TYPE    = {"Working", "Commercial associate", "Pensioner", "State servant", "Student"}
VALID_EDUCATION      = {"Higher education", "Secondary / secondary special", "Incomplete higher", "Lower secondary", "Academic degree"}
VALID_FAMILY_STATUS  = {"Civil marriage", "Married", "Single / not married", "Separated", "Widow"}
VALID_HOUSING_TYPE   = {"Rented apartment", "House / apartment", "Municipal apartment", "With parents", "Co-op apartment", "Office apartment"}
VALID_OCCUPATION     = {"", "Security staff", "Sales staff", "Accountants", "Laborers", "Managers", "Drivers", "Core staff", "High skill tech staff", "Cleaning staff", "Private service staff", "Cooking staff", "Low-skill Laborers", "Medicine staff", "Secretaries", "Waiters/barmen staff", "HR staff", "Realty agents", "IT staff"}

def validate_form(form: dict) -> list:
    errors = []

    def _require_float(key, label, min_val=0.0, max_val=1e9):
        try:
            val = float(form.get(key, ""))
            if val < min_val or val > max_val:
                errors.append(f"{label} must be between {min_val} and {max_val}.")
        except (ValueError, TypeError):
            errors.append(f"{label} must be a valid number.")

    def _require_choice(key, label, valid_set):
        val = str(form.get(key, "")).strip()
        # Case insensitive check for convenience, though exact match preferred
        if val not in valid_set and val.capitalize() not in valid_set and val.upper() not in valid_set:
             # Just doing an exact match with the sets provided which have exact case
             if val not in valid_set:
                 errors.append(f"{label} has an invalid value.")

    _require_float("Age",               "Age",              min_val=18, max_val=100)
    _require_float("YearsEmployed",     "Years Employed",   min_val=0,  max_val=60)
    _require_float("CNT_CHILDREN",      "Children Count",   min_val=0,  max_val=20)
    _require_float("AMT_INCOME_TOTAL",  "Annual Income",    min_val=0,  max_val=1e9)
    _require_float("CNT_FAM_MEMBERS",   "Family Members",   min_val=1,  max_val=25)

    _require_choice("CODE_GENDER",         "Gender",           VALID_GENDER)
    _require_choice("FLAG_OWN_CAR",        "Own Car",          VALID_Y_N)
    _require_choice("FLAG_OWN_REALTY",     "Own Property",     VALID_Y_N)
    _require_choice("NAME_INCOME_TYPE",    "Income Type",      VALID_INCOME_TYPE)
    _require_choice("NAME_EDUCATION_TYPE", "Education Level",  VALID_EDUCATION)
    _require_choice("NAME_FAMILY_STATUS",  "Family Status",    VALID_FAMILY_STATUS)
    _require_choice("NAME_HOUSING_TYPE",   "Housing Type",     VALID_HOUSING_TYPE)
    
    # Optional occupation
    occ = form.get("OCCUPATION_TYPE", "").strip()
    if occ and occ not in VALID_OCCUPATION:
         errors.append("Occupation Type has an invalid value.")

    return errors


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico', mimetype='image/vnd.microsoft.icon'
    )

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/predict", methods=["GET", "POST"])
def predict_view():
    if request.method == "GET":
        return render_template("predict.html")

    form_data = request.form.to_dict()
    errors    = validate_form(form_data)

    if errors:
        return render_template("predict.html", errors=errors, form_data=form_data)

    predict_fn = _get_predict()
    if predict_fn is None:
        return render_template(
            "predict.html",
            errors=["Model not found. Please train the model first by running `python train.py`."],
            form_data=form_data,
        )

    try:
        age_years = float(form_data["Age"])
        years_employed = float(form_data["YearsEmployed"])
        is_employed = str(form_data.get("IsEmployed", "Y")) == "Y"

        raw_input = {
            "CODE_GENDER":         form_data["CODE_GENDER"],
            "FLAG_OWN_CAR":        form_data["FLAG_OWN_CAR"],
            "FLAG_OWN_REALTY":     form_data["FLAG_OWN_REALTY"],
            "CNT_CHILDREN":        float(form_data["CNT_CHILDREN"]),
            "AMT_INCOME_TOTAL":    float(form_data["AMT_INCOME_TOTAL"]),
            "NAME_INCOME_TYPE":    form_data["NAME_INCOME_TYPE"],
            "NAME_EDUCATION_TYPE": form_data["NAME_EDUCATION_TYPE"],
            "NAME_FAMILY_STATUS":  form_data["NAME_FAMILY_STATUS"],
            "NAME_HOUSING_TYPE":   form_data["NAME_HOUSING_TYPE"],
            "DAYS_BIRTH":          -int(age_years * 365.25),
            "DAYS_EMPLOYED":       -int(years_employed * 365.25) if is_employed else 365243,
            "FLAG_MOBIL":          int(form_data.get("FLAG_MOBIL", 1)),
            "FLAG_WORK_PHONE":     int(form_data.get("FLAG_WORK_PHONE", 0)),
            "FLAG_PHONE":          int(form_data.get("FLAG_PHONE", 0)),
            "FLAG_EMAIL":          int(form_data.get("FLAG_EMAIL", 0)),
            "OCCUPATION_TYPE":     form_data.get("OCCUPATION_TYPE", ""),
            "CNT_FAM_MEMBERS":     float(form_data["CNT_FAM_MEMBERS"]),
        }

        result = predict_fn(raw_input)
        
        # Determine model name used for prediction
        model_name = "Best Model"
        model_name_path = MODELS_DIR / "best_model_name.txt"
        if model_name_path.exists():
            model_name = model_name_path.read_text().strip()

        return render_template("result.html", result=result, applicant=form_data, model_name=model_name)

    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        return render_template(
            "predict.html",
            errors=[f"An unexpected error occurred: {str(exc)}"],
            form_data=form_data,
        )


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def server_error(e):
    # Could render a dedicated 500 page, but index is safe fallback
    return render_template("index.html"), 500


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=FLASK_DEBUG)
