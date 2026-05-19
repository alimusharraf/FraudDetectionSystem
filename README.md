# Real-Time Fraud Detection System

## Description

This project is an end-to-end machine learning based fraud detection system developed using the IEEE-CIS Fraud Detection dataset. The system analyzes transaction behavior and predicts whether a transaction is fraudulent or legitimate using advanced machine learning models and explainable AI techniques.

The project includes data preprocessing, feature engineering, imbalance handling using SMOTE, multiple model training, SHAP explainability, risk segmentation, and an interactive Streamlit dashboard for fraud analysis and monitoring.


## Features

- Fraud transaction prediction using machine learning
- Risk segmentation into Critical Risk, Suspicious, and Clear categories
- Explainable AI using SHAP values
- Interactive Streamlit dashboard
- Plotly based visualizations
- Fraud probability analysis
- Transaction explorer with SHAP explanation
- Threshold optimization and model tuning


## Machine Learning Models Used

- XGBoost
- LightGBM
- Isolation Forest


## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- SHAP
- Streamlit
- Plotly
- Matplotlib
- Seaborn
- Joblib


## Project Structure

```text
FraudDetection/

│
├── charts/
├── dashboard/
│   ├── app.py
│   ├── model.pkl
│   ├── scaler.pkl
│
├── data/
│   ├── final_data.csv
│   ├── train_identity.csv
│   ├── train_transaction.csv
│
├── analysis.ipynb
├── README.md
├── requirements.txt
```

## Run Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```


## Dashboard Features

- Fraud overview dashboard
- Transaction explorer
- SHAP explainability page
- Fraud probability visualization
- Interactive charts and analytics


## Visualizations Included

- SHAP Global Summary Plot
- Fraud Rate by Hour of Day
- Transaction Amount Distribution
- Risk Tier Donut Chart
- Precision-Recall Curve
- Interactive Plotly Scatter Plot


## Future Improvements

- Real-time fraud monitoring
- Deep learning based fraud detection
- Cloud deployment
- Device fingerprint analysis
- Geolocation based fraud detection


Machine Learning & Explainable AI Internship Project