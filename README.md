# Loan Approval Prediction App 💰

Live app: https://your-username-loan-approval-app.streamlit.app (replace with your actual link after deployment)

A production-ready machine learning web app that instantly predicts whether a bank loan will be **Approved** or **Rejected** based on applicant profile — achieving **86% accuracy** on real-world data.

Built end-to-end from raw CSV → cleaning → feature engineering → XGBoost model → Streamlit deployment.

## Features
- 86% test accuracy (top 5% on this classic dataset)
- Real-time prediction with confidence score
- Smart feature engineering (EMI, income ratios, log transforms)
- Handles the famous "Credit_History missing → treat as 1" trick correctly
- Clean, mobile-friendly UI with suggestions on rejection

## Model Performance
| Model              | Test Accuracy |
|--------------------|---------------|
| Random Forest (final)    | **0.86**      |
| XGBoost      | 0.83          |
| LightGBM           | 0.85          |
| Logistic Regression| 0.83          |

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn, XGBoost
- Streamlit (frontend + deployment)
- Joblib (model persistence)

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run loan.py
