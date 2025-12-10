# Loan Approval Predictor – Live & Production Ready

Live App: https://predictyourloans.streamlit.app

A complete end-to-end machine learning web application that instantly predicts whether a bank will **approve** or **reject** a loan application — achieving **86.46% accuracy** on unseen data.

Built, trained, compared 5 models, and deployed live in a single session.

### Live Model Comparison (runs automatically on every cold start) ###
```text
Training models...

RandomForest         → Test Accuracy: 0.8646  ← WINNER
LightGBM             → Test Accuracy: 0.8438
LogisticRegression   → Test Accuracy: 0.8438
XGBoost_Tuned        → Test Accuracy: 0.8333
XGBoost              → Test Accuracy: 0.8229

==================================================
WINNER: RandomForest
Accuracy: 0.8646
Model saved as → best_loan_model.joblib
==================================================

Features:

Real-time loan approval prediction
Confidence score displayed
Actionable rejection suggestions (improve credit history, reduce loan amount, etc.)
Smart feature engineering:EMI (monthly installment)
Total Income + Loan-to-Income ratio
Log transformation on skewed features
Correct handling of missing Credit_History (the famous +5–7% accuracy trick)

Fully responsive – works perfectly on mobile

Tech Stack

Python
Pandas, NumPy
Scikit-learn, Joblib
Streamlit (frontend + free hosting on Streamlit Cloud)

├── loan.py                  → Main app (Streamlit UI + auto model training & selection)
├── best_loan_model.joblib   → Trained RandomForest (86.46% accuracy)
├── requirements.txt         → Minimal, deployment-optimized dependencies
├── README.md                → This file

How to Run Locally

bash
git clone https://github.com/your-username/loan-status-prediction.git
cd loan-status-prediction
pip install -r requirements.txt
streamlit run loan.py



