"""
tests/test_app.py — Flask application test suite
==================================================
Tests cover:
  - All page routes (GET)
  - Valid prediction POST
  - Invalid input validation errors
  - Boundary values
  - Missing fields
  - Non-numeric inputs

Run with:
    python -m pytest tests/ -v
"""

import sys
from pathlib import Path

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from app import app, validate_form


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """Flask test client with test config."""
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as client:
        yield client


VALID_FORM = {
    "CODE_GENDER":         "M",
    "Age":                 "35",
    "CNT_CHILDREN":        "0",
    "CNT_FAM_MEMBERS":     "1",
    "AMT_INCOME_TOTAL":    "50000",
    "NAME_INCOME_TYPE":    "Working",
    "OCCUPATION_TYPE":     "Laborers",
    "IsEmployed":          "Y",
    "YearsEmployed":       "5",
    "NAME_EDUCATION_TYPE": "Higher education",
    "NAME_FAMILY_STATUS":  "Single / not married",
    "FLAG_OWN_CAR":        "Y",
    "FLAG_OWN_REALTY":     "Y",
    "NAME_HOUSING_TYPE":   "House / apartment",
    "FLAG_MOBIL":          "1",
    "FLAG_WORK_PHONE":     "0",
    "FLAG_PHONE":          "0",
    "FLAG_EMAIL":          "0"
}


# ─── Page Route Tests ─────────────────────────────────────────────────────────

class TestPages:
    def test_index_get(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"CreditAI" in resp.data or b"Credit" in resp.data

    def test_about_get(self, client):
        resp = client.get("/about")
        assert resp.status_code == 200

    def test_contact_get(self, client):
        resp = client.get("/contact")
        assert resp.status_code == 200

    def test_predict_get(self, client):
        resp = client.get("/predict")
        assert resp.status_code == 200
        assert b"prediction-form" in resp.data or b"Predict" in resp.data

    def test_404(self, client):
        resp = client.get("/nonexistent-route-xyz")
        # Our 404 handler returns 404 page
        assert resp.status_code == 404


# ─── Validation Logic Tests ───────────────────────────────────────────────────

class TestValidation:
    def test_valid_form_passes(self):
        errors = validate_form(VALID_FORM)
        assert errors == [], f"Expected no errors but got: {errors}"

    def test_invalid_age_too_young(self):
        data = {**VALID_FORM, "Age": "15"}
        errors = validate_form(data)
        assert any("Age" in e for e in errors)

    def test_invalid_age_too_old(self):
        data = {**VALID_FORM, "Age": "120"}
        errors = validate_form(data)
        assert any("Age" in e for e in errors)

    def test_non_numeric_age(self):
        data = {**VALID_FORM, "Age": "abc"}
        errors = validate_form(data)
        assert any("Age" in e for e in errors)

    def test_negative_income(self):
        data = {**VALID_FORM, "AMT_INCOME_TOTAL": "-500"}
        errors = validate_form(data)
        assert any("Income" in e for e in errors)

    def test_invalid_gender_value(self):
        data = {**VALID_FORM, "CODE_GENDER": "X"}
        errors = validate_form(data)
        assert any("Gender" in e for e in errors)

    def test_invalid_family_status(self):
        data = {**VALID_FORM, "NAME_FAMILY_STATUS": "UnknownStatus"}
        errors = validate_form(data)
        assert any("Family" in e for e in errors)

    def test_years_employed_boundary_zero(self):
        """YearsEmployed = 0 should be valid (fresh graduate)."""
        data = {**VALID_FORM, "YearsEmployed": "0"}
        errors = validate_form(data)
        assert errors == []

    def test_years_employed_boundary_sixty(self):
        """YearsEmployed = 60 is the upper boundary — should be valid."""
        data = {**VALID_FORM, "YearsEmployed": "60"}
        errors = validate_form(data)
        assert errors == []

    def test_years_employed_exceeds_max(self):
        data = {**VALID_FORM, "YearsEmployed": "65"}
        errors = validate_form(data)
        assert any("Years Employed" in e for e in errors)

    def test_missing_field_empty_gender(self):
        data = {**VALID_FORM, "CODE_GENDER": ""}
        errors = validate_form(data)
        assert any("Gender" in e for e in errors)

    def test_income_zero(self):
        """Income = 0 is valid."""
        data = {**VALID_FORM, "AMT_INCOME_TOTAL": "0"}
        errors = validate_form(data)
        assert errors == []

    def test_multiple_errors_returned(self):
        """Multiple invalid fields → multiple error messages."""
        data = {**VALID_FORM, "Age": "abc", "AMT_INCOME_TOTAL": "-1", "CODE_GENDER": "z"}
        errors = validate_form(data)
        assert len(errors) >= 3


# ─── Prediction POST Tests ────────────────────────────────────────────────────

class TestPredictionRoute:
    def test_invalid_post_returns_errors(self, client):
        """Posting invalid data → re-renders predict.html with error list."""
        bad_data = {**VALID_FORM, "Age": "abc"}
        resp = client.post("/predict", data=bad_data)
        assert resp.status_code == 200
        # Should show error message in page
        assert b"error" in resp.data.lower() or b"fix" in resp.data.lower() or b"invalid" in resp.data.lower()

    def test_valid_post_returns_200(self, client):
        """
        Valid form post should return 200 (either result page or
        predict page with model-not-found error if model not trained yet).
        """
        resp = client.post("/predict", data=VALID_FORM)
        assert resp.status_code == 200
