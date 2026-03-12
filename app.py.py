import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page Title
st.title("Loan Default Prediction App")
st.write("Enter borrower details to predict loan default risk.")

# Load trained model
model = pickle.load(open("loan_default_model.pkl","rb"))

# -----------------------------
# USER INPUTS
# -----------------------------

credit_policy = st.number_input("Credit Policy (0 or 1)", min_value=0.0, max_value=1.0)
int_rate = st.number_input("Interest Rate")
installment = st.number_input("Installment")
log_annual_inc = st.number_input("Log Annual Income")
dti = st.number_input("Debt-to-Income Ratio")

# Extra important financial features
fico = st.number_input("FICO Credit Score", value=700)
days_with_cr_line = st.number_input("Days With Credit Line", value=3000)
revol_bal = st.number_input("Revolving Balance", value=5000)
revol_util = st.number_input("Revolving Utilization (%)", value=50)

inq_last_6mths = st.number_input("Inquiries in Last 6 Months", value=1)
delinq_2yrs = st.number_input("Delinquencies in Last 2 Years", value=0)
pub_rec = st.number_input("Public Records", value=0)

# -----------------------------
# PREDICTION BUTTON
# -----------------------------

if st.button("Predict Loan Status"):

    # Model expects 18 features
    input_data = np.array([[

        credit_policy,
        int_rate,
        installment,
        log_annual_inc,
        dti,

        fico,
        days_with_cr_line,
        revol_bal,
        revol_util,

        inq_last_6mths,
        delinq_2yrs,
        pub_rec,

        0,0,0,0,0,0
    ]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.error(f"⚠ High Risk of Default ({probability[0][1]*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk Loan ({probability[0][0]*100:.2f}%)")


# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------

st.subheader("Model Feature Importance")

importance = model.feature_importances_

features = [
"credit_policy","int_rate","installment","log_annual_inc","dti",
"fico","days_with_cr_line","revol_bal","revol_util",
"inq_last_6mths","delinq_2yrs","pub_rec",
"purpose1","purpose2","purpose3","purpose4","purpose5","purpose6"
]

importance_df = pd.DataFrame({
"Feature":features,
"Importance":importance
}).sort_values(by="Importance",ascending=False)

st.bar_chart(importance_df.set_index("Feature"))