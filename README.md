# 💳 Credit Card Approval Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-EC6B23?style=for-the-badge)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5-7952B3?style=for-the-badge&logo=bootstrap)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

### AI-Powered Credit Card Approval Prediction using Machine Learning & Flask

Predict credit card approval decisions using advanced machine learning models trained on real-world banking datasets.

</div>

---

# 📌 Overview

Banks receive thousands of credit card applications every day. Manual review of each application is time-consuming and prone to human error.

This project automates the approval process using Machine Learning by analyzing an applicant's financial profile and predicting whether a credit card application is likely to be **Approved** or **Rejected**.

The system provides

- ✅ Instant approval prediction
- 📈 Approval probability
- ⚠️ Risk classification
- 🤖 Best-performing ML model inference
- 🌐 Interactive Flask web application

---

# ✨ Features

- Professional Banking Dashboard UI
- Machine Learning Pipeline
- Real-Time Prediction
- Feature Engineering
- Multiple Classification Algorithms
- Approval Probability
- Risk Level Detection
- Applicant Summary
- Responsive Design
- Deployment Ready
- Clean Modular Architecture
- Automated Testing

---

# 🧠 Machine Learning Workflow

```text
Application Record
        +
Credit Record
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Encoding
        │
        ▼
Scaling
        │
        ▼
Model Training
        │
        ▼
Cross Validation
        │
        ▼
Best Model Selection
        │
        ▼
Prediction
```

---

# 📊 Dataset

This project uses two datasets:

## application_record.csv

Contains applicant information such as

- Gender
- Income Type
- Annual Income
- Education
- Occupation
- Family Status
- Housing Type
- Employment Duration
- Family Members

---

## credit_record.csv

Contains customer credit history

- ID
- MONTHS_BALANCE
- STATUS

The datasets are merged using

```
ID
```

to build a complete applicant profile.

---

# ⚙️ Data Preprocessing

The preprocessing pipeline performs

- Missing Value Imputation
- Duplicate Removal
- Dataset Merge
- Binary Target Generation
- Feature Encoding
- Feature Scaling
- Feature Engineering
- Stratified Train-Test Split

---

# 🔧 Feature Engineering

The following engineered features are automatically created.

| Feature | Description |
|----------|-------------|
| Age Years | Converts DAYS_BIRTH into years |
| Years Employed | Converts employment days into years |
| Employment Stability | Measures employment consistency |
| Income Group | Categorizes applicants into Low / Medium / High income |
| Income Per Family Member | Income normalized by family size |
| Financial Stability Score | Composite financial indicator |
| Risk Score | Normalized financial risk metric |

---

# 🤖 Machine Learning Models

The project trains multiple classification algorithms.

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

The best model is selected using

- Cross Validation
- ROC-AUC
- Accuracy
- Precision
- Recall
- F1 Score

---

# 📈 Model Performance

| Model | Accuracy | ROC-AUC |
|--------|----------|----------|
| Random Forest | 82.6% | 0.726 |
| XGBoost | 74.1% | 0.712 |
| Decision Tree | 60.0% | 0.596 |
| Logistic Regression | 55.6% | 0.552 |

🏆 **Best Model:** Random Forest

---

# 🌐 Web Application

The Flask application provides

### 🏠 Home

Professional landing page.

---

### 📝 Prediction Form

Applicants enter

- Gender
- Income
- Education
- Employment
- Housing
- Family Information
- Occupation

---

### 📊 Result Dashboard

Displays

- Approval Decision
- Approval Probability
- Risk Category
- Model Used
- Applicant Details
- Recommendation

---

# 📂 Project Structure

```
CreditCardApprovalPrediction/

│
├── dataset/
│   ├── application_record.csv
│   └── credit_record.csv
│
├── models/
│
├── notebooks/
│
├── static/
│
├── templates/
│
├── tests/
│
├── app.py
├── preprocessing.py
├── train.py
├── predict.py
├── config.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/CreditCardApprovalPrediction.git

cd CreditCardApprovalPrediction
```

Create virtual environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Train the Model

```bash
python train.py
```

---

# ▶️ Run Flask Application

```bash
python app.py
```

Open

```
http://127.0.0.1:5000
```

---

# 🧪 Testing

Run

```bash
pytest
```

Current Status

✅ 20 Tests Passed

---

# 📸 Screenshots

## Home Page

> *(Add screenshot here)*

---

## Prediction Form

> *(Add screenshot here)*

---

## Prediction Result

> *(Add screenshot here)*

---

# 💻 Technologies Used

### Backend

- Python
- Flask

### Machine Learning

- Scikit-Learn
- XGBoost
- NumPy
- Pandas
- Joblib

### Visualization

- Matplotlib
- Seaborn

### Frontend

- HTML5
- CSS3
- Bootstrap 5
- JavaScript

---

# 🌍 Deployment

The project is deployment-ready.

Supports

- Vercel
- Render
- Gunicorn
- WSGI

---

# 📌 Future Improvements

- Explainable AI using SHAP
- Probability Calibration
- REST API
- User Authentication
- Admin Dashboard
- Batch Prediction
- Docker Support

---

# ⚠️ Disclaimer

This project is developed **for educational and research purposes only**.

It demonstrates how machine learning can assist in credit approval decisions.

It **must not** be used as a real-world banking approval system without:

- Regulatory compliance
- Bias evaluation
- Fairness analysis
- Human review



# ⭐ Support

If you found this project useful,

⭐ Star this repository

🍴 Fork it

🤝 Contribute

---
