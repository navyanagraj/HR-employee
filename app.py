import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

@st.cache_resource
def load_artifacts():
    model, scaler = None, None
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
    if os.path.exists("scaler.pkl"):
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace; }
.stApp { background: #0f0f13; color: #e8e8f0; }
.block-container { padding: 2rem 2rem; }
.churn    { background: #2d0a0a; border: 2px solid #ff4444; border-radius: 12px;
            padding: 1.5rem; text-align: center; color: #ff6b6b;
            font-family: 'Space Mono', monospace; font-size: 1.4rem; font-weight: 700; }
.no-churn { background: #0a2d0a; border: 2px solid #44ff88; border-radius: 12px;
            padding: 1.5rem; text-align: center; color: #4dff91;
            font-family: 'Space Mono', monospace; font-size: 1.4rem; font-weight: 700; }
div[data-testid="stSidebar"] { background: #16161e; }
</style>
""", unsafe_allow_html=True)

st.title("📉 Customer Churn Predictor")
st.markdown("*Predict whether a mall customer is at risk of churning.*")
st.divider()

st.sidebar.header("🛍️ Customer Details")

# ── Exact features from your notebook ─────────────────────────────────────────
# Columns after preprocessing: CustomerID, Gender (LabelEncoded), Age,
# Annual Income (k$), Spending Score (1-100), then all scaled with StandardScaler
# Churn was derived: (Spending Score < 0 after scaling) → 1 else 0

customer_id    = st.sidebar.number_input("Customer ID",              1,    1000, 101,  step=1)
gender         = st.sidebar.selectbox(   "Gender",                   ["Male", "Female"])
age            = st.sidebar.slider(      "Age",                      18,   70,   35)
annual_income  = st.sidebar.number_input("Annual Income (k$)",       15,   140,  60,   step=1)
spending_score = st.sidebar.slider(      "Spending Score (1–100)",   1,    100,  50,
    help="Score assigned by the mall (1=low spender, 100=high spender)")

# ── Encode Gender: LabelEncoder on ['Female','Male'] → Female=0, Male=1 ───────
gender_encoded = 1 if gender == "Male" else 0

# ── Column order matches your notebook's X (after LabelEncoding + StandardScaler) ─
# notebook: X = df.drop('Churn', axis=1)
# df columns before drop: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)
input_data = pd.DataFrame([{
    "CustomerID":             int(customer_id),
    "Gender":                 int(gender_encoded),
    "Age":                    int(age),
    "Annual Income (k$)":     int(annual_income),
    "Spending Score (1-100)": int(spending_score),
}])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Age",       age)
c2.metric("Income",    f"${annual_income}k")
c3.metric("Spend Score", spending_score)
c4.metric("Gender",    gender)
st.divider()

if st.button("🔮 Predict Churn", use_container_width=True, type="primary"):
    if model is None:
        st.error("⚠️ model.pkl not found. Add your trained model file.")
    else:
        try:
            data = input_data.copy()
            # Your notebook scaled ALL columns including CustomerID
            if scaler is not None:
                data = pd.DataFrame(scaler.transform(data), columns=input_data.columns)

            prediction = model.predict(data)[0]
            proba      = model.predict_proba(data)[0] if hasattr(model, "predict_proba") else None

            if prediction == 1:
                st.markdown('<div class="churn">⚠️ HIGH CHURN RISK</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="no-churn">✅ LIKELY TO STAY</div>', unsafe_allow_html=True)

            if proba is not None:
                st.subheader("Confidence")
                col1, col2 = st.columns(2)
                col1.metric("Stay",  f"{proba[0]*100:.1f}%")
                col2.metric("Churn", f"{proba[1]*100:.1f}%")
                st.progress(float(proba[1]))

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("💡 Ensure model.pkl (and optionally scaler.pkl) are in the same folder as app.py")

with st.expander("📋 Feature vector sent to model"):
    st.dataframe(input_data, use_container_width=True)

with st.expander("ℹ️ Model Info"):
    st.markdown("""
    **Dataset:** Mall Customer dataset (200 rows, 5 features)  
    **Target:** Churn = 1 if Spending Score < 0 after StandardScaler (i.e. below-average spender)  
    **Algorithms trained:** Logistic Regression, KNN, Decision Tree, Random Forest, SVC, Gradient Boosting  
    **Preprocessing:** LabelEncoder (Gender) → StandardScaler → SMOTE (on train set)  
    **Features (exact order):** CustomerID · Gender · Age · Annual Income (k$) · Spending Score (1-100)
    """)
