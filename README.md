# 💳 CreditAI — Credit Card Approval Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask)
![Random Forest](https://img.shields.io/badge/Random_Forest-scikit--learn-orange?style=for-the-badge)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple?style=for-the-badge&logo=bootstrap)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Vercel](https://img.shields.io/badge/Deploy-Vercel-black?style=for-the-badge&logo=vercel)

**An AI-powered credit card approval prediction system built with Flask and scikit-learn.**  
*B.Tech Final Year Project · Portfolio Showcase · Production-Ready*

[🚀 Live Demo](#) · [📊 Model Results](#model-results) · [🛠 Installation](#installation) · [📖 API Docs](#backend-api)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Results](#model-results)
- [Deployment (Vercel)](#deployment-vercel)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

Banks process thousands of credit card applications daily. Manual review is slow, expensive, and inconsistent. **CreditAI** automates this with a machine learning pipeline trained on the **Kaggle Credit Card Approval Dataset**.

Given 17 applicant features (financial, demographic, and employment data), the system predicts:
- ✅ **Approved** or ❌ **Rejected**
- 📊 **Approval probability** (0–100%)
- 🛡️ **Risk level** (Low / Medium / High)
- 💬 **Confidence explanation** in plain English

---

## Features

| Feature | Description |
|---|---|
| 🤖 **ML Pipeline** | Logistic Regression, Decision Tree, Random Forest, XGBoost trained and compared |
| 🌲 **Random Forest** | Best-performing model selected automatically via cross-validation |
| 🔧 **Feature Engineering** | Employment stability, income groups, financial stability score, risk score |
| 📊 **Cross-Validation** | 5-fold stratified CV to prevent overfitting |
| 🌐 **Flask Web App** | 5 pages: Home, About, Predict, Result, Contact |
| 🎨 **Premium UI** | Banking-grade Bootstrap 5 design with smooth animations |
| ✅ **Validation** | Client-side (JS) + server-side (Python) input validation |
| 📈 **Confidence Score** | Dynamic probability thresholding mapping to risk levels |
| ☁️ **Deployment** | Serverless-ready for Vercel with `vercel.json` |

---

## Architecture

```
User Input (Flask Form)
        │
        ▼
 Input Validation (app.py)
        │
        ▼
 Feature Engineering (preprocessing.py)
   ├── EmploymentStability
   ├── IncomeGroup
   ├── FinancialStabilityScore
   └── RiskScore
        │
        ▼
 sklearn ColumnTransformer
   ├── Numeric: Impute → StandardScaler
   └── Categorical: Impute → OrdinalEncoder
        │
        ▼
 Random Forest Classifier
        │
        ▼
 Result (Probability + Decision + Risk)
```

---

## Folder Structure

```
CreditCardApprovalPrediction/
├── dataset/
│   ├── application_record.csv     # Applicant demographics and income
│   └── credit_record.csv          # Monthly balance and loan statuses
├── models/
│   ├── best_model.pkl             # Trained pipeline (preprocessor + model)
│   ├── feature_names.pkl          # Feature name list for inference
│   └── results.json               # Model comparison metrics
├── static/
│   ├── css/style.css              # Custom UI styles
│   ├── js/main.js                 # Form validation, animations
│   └── images/plots/              # Evaluation plots
├── templates/
│   ├── index.html                 # Landing page
│   ├── predict.html               # Prediction form
│   ├── result.html                # Result display
│   ├── about.html                 # Project about page
│   └── contact.html               # Contact page
├── tests/
│   └── test_app.py                # pytest test suite
├── app.py                         # Flask application
├── train.py                       # Model training script
├── preprocessing.py               # Feature engineering & pipeline
├── predict.py                     # Prediction module
├── utils.py                       # Shared utilities
├── config.py                      # Centralised configuration
├── requirements.txt
├── vercel.json                    # Vercel deployment configuration
├── wsgi.py                        # Serverless entry point
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.12+
- pip
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/CreditCardApprovalPrediction.git
cd CreditCardApprovalPrediction
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Train the Model
```bash
python train.py
```

This will:
- Process the datasets from `dataset/`
- Train 4 ML models (Logistic Regression, Decision Tree, Random Forest, XGBoost)
- Run cross-validation
- Save the best model to `models/best_model.pkl`

### Step 5: Run the Application
```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## Usage

### Web Interface
1. Navigate to `http://localhost:5000`
2. Click **"Predict Now"**
3. Fill in all 17 applicant fields
4. Click **"Predict Approval"**
5. View the decision, probability score, and risk level

### Run Tests
```bash
pytest tests/ -v
```

---

## Model Results

Based on 5-fold stratified cross-validation on the Kaggle dataset:

| Model | Accuracy | Precision | Recall | F1 | CV ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~55.6% | ~0.89 | ~0.56 | ~0.69 | ~0.536 |
| Decision Tree | ~60.0% | ~0.90 | ~0.60 | ~0.72 | ~0.592 |
| XGBoost | ~74.1% | ~0.92 | ~0.76 | ~0.83 | ~0.686 |
| **Random Forest (Best)** | **~82.6%** | **~0.91** | **~0.88** | **~0.89** | **~0.697** |

> **Why Random Forest?**  
> Random Forest outperformed the other models on cross-validated ROC-AUC on this specific dataset, offering the best balance between precision and recall while preventing test-set leakage.

---

## Dataset

- **Source:** [Kaggle Credit Card Approval Prediction Dataset](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)
- **Instances:** Over 36,000 unique applicants (after merging).
- **Features:** 17 (financial, demographics, employment history).
- **Target Extraction:** Target extracted by analyzing `credit_record.csv`. Any applicant with an overdue status (1-5) is flagged as high-risk/rejected. All others are approved.
- **Handling Missing Values:** Median imputation for numerical data, mode imputation for categorical data.

---

## Deployment (Vercel)

Deploy this application for free on Vercel's serverless platform.

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/credit-approval-system.git
git push -u origin main
```

### Step 2: Deploy on Vercel
1. Go to [Vercel.com](https://vercel.com) and log in with GitHub.
2. Click **Add New...** -> **Project**.
3. Import your `credit-approval-system` repository.
4. Vercel will automatically detect the `vercel.json` and `wsgi.py` files.
5. Click **Deploy**.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <strong>Built with ❤️ as a B.Tech Final Year Project</strong><br/>
  <em>Demonstrating end-to-end ML engineering: Data → Model → API → UI → Deployment</em>
</div>
