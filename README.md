# Loan Default Risk Prediction using Machine Learning

## Overview
This project predicts whether a borrower is likely to default on a loan using machine learning techniques. Financial institutions need to evaluate the risk before approving loans. By analyzing borrower financial information such as income, credit score, debt-to-income ratio, and credit history, machine learning models can estimate the probability of loan default.

This project also includes an interactive Streamlit web application that allows users to enter borrower details and receive a real-time prediction of loan default risk.

## Problem Statement
Loan default prediction is an important problem in the banking and financial sector. If a borrower fails to repay a loan, the lender faces financial loss. Machine learning models can help financial institutions analyze borrower data and identify high-risk borrowers before loan approval.

## Dataset
The dataset used in this project contains borrower financial information including:

Credit Policy  
Interest Rate  
Installment Amount  
Annual Income  
Debt-to-Income Ratio  
FICO Credit Score  
Revolving Balance  
Revolving Utilization  
Inquiries in Last 6 Months  
Delinquencies in Last 2 Years  
Public Records  

These features help the model understand borrower financial behavior.

## Machine Learning Model
This project uses Logistic Regression for classification.

Workflow used in the project:

1. Data cleaning and preprocessing  
2. Exploratory Data Analysis (EDA)  
3. Feature selection  
4. Train-test split  
5. Model training  
6. Model evaluation  
7. Save trained model  
8. Build Streamlit prediction application  

## Model Performance
Model Accuracy: approximately 84%

The model predicts whether a loan is:

Low Risk Loan  
High Risk of Default  

## Streamlit Web Application
The Streamlit application allows users to enter borrower financial information and predict loan default risk. It provides an interactive interface for testing the model with different borrower inputs.

The application also visualizes feature importance to show which financial factors influence the prediction.

## Technologies Used
Python  
Pandas  
NumPy  
Scikit-learn  
Matplotlib  
Streamlit  

## Project Structure
loan-default-prediction

app.py  
loan_default_model_training.ipynb  
loan_data.csv  
loan_default_model.pkl  
README.md  

## How to Run the Project

Install dependencies:

pip install streamlit pandas numpy scikit-learn matplotlib

Run the Streamlit application:

streamlit run app.py

## Future Improvements
Use advanced models such as Random Forest and XGBoost  
Perform hyperparameter tuning  
Deploy the application online  
Add explainable AI techniques such as SHAP  

## Author
Manneti Yeswanth Reddy  
B.Tech Artificial Intelligence and Data Science  
Saveetha School of Engineering
