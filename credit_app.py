import streamlit as st
import numpy as np
import joblib

# Load the scaler and model
scaler = joblib.load("scaler.joblib")
model = joblib.load("credit_model.pkl")

st.title("Credit Risk Scoring Model By BAHRI Salma")

# Input your financial ratios
debt_ratio = st.number_input("Debt Ratio", min_value=0.0, format="%.2f")
equity_ratio = st.number_input("Equity Ratio", min_value=0.0, format="%.2f")
liquidity_ratio = st.number_input("Liquidity Ratio", min_value=0.0, format="%.2f")
solvency_ratio = st.number_input("Solvency Ratio", min_value=0.0, format="%.2f")
net_profit_margin = st.number_input("Net Profit Margin", format="%.2f")
rorwa = st.number_input("RORWA", format="%.2f")
net_fin_debt_ebitda = st.number_input("Net Financial Debt/EBITDA", format="%.2f")
debt_service_capability = st.number_input("Debt Service Capability", format="%.2f")

if st.button("Get Credit Score"):
    # Prepare input features
    features = np.array([[debt_ratio, equity_ratio, liquidity_ratio, solvency_ratio,
                          net_profit_margin, rorwa, net_fin_debt_ebitda, debt_service_capability]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict risk probability
    y_proba = model.predict_proba(features_scaled)[:, 1][0]
    
    # Calculate credit score
    credit_score = 100 - 100 * y_proba
    
    # Define credit category
    def credit_category(score):
        if score >= 70:
            return 'Creditworthy'
        elif score >= 40:
            return 'Borderline'
        else:
            return 'Deny Credit'
    
    category = credit_category(credit_score)
    
    # Show results
    st.write(f"Predicted Probability of Risk: {y_proba:.4f}")
    st.write(f"Credit Score: {credit_score:.2f}")
    st.write(f"Credit Category: {category}")

