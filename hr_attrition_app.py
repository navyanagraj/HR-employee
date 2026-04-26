import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="HR Attrition Predictor", page_icon="👥", layout="centered")

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
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
.stApp { background: #0d1117; color: #c9d1d9; }
.stay-box  { background: #0d2818; border: 1px solid #238636; border-radius: 10px;
             padding: 1.5rem; text-align: center; color: #3fb950;
             font-family: 'IBM Plex Mono', monospace; font-size: 1.3rem; font-weight: 600; }
.leave-box { background: #2d1117; border: 1px solid #da3633; border-radius: 10px;
             padding: 1.5rem; text-align: center; color: #f85149;
             font-family: 'IBM Plex Mono', monospace; font-size: 1.3rem; font-weight: 600; }
div[data-testid="stSidebar"] { background: #161b22; }
</style>
""", unsafe_allow_html=True)

st.title("👥 HR Employee Attrition Predictor")
st.markdown("*Predict whether an employee is at risk of leaving (IBM HR Analytics dataset).*")
st.divider()

st.sidebar.header("👤 Employee Details")

# ── Raw inputs (before get_dummies) ──────────────────────────────────────────
st.sidebar.subheader("Personal")
age              = st.sidebar.slider("Age", 18, 60, 35)
gender           = st.sidebar.selectbox("Gender", ["Male", "Female"])
marital_status   = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
distance_home    = st.sidebar.slider("Distance from Home (km)", 1, 30, 5)
education        = st.sidebar.selectbox("Education Level",
    ["Below College (1)", "College (2)", "Bachelor (3)", "Master (4)", "Doctor (5)"])
education_field  = st.sidebar.selectbox("Education Field",
    ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])

st.sidebar.subheader("Job")
department       = st.sidebar.selectbox("Department",
    ["Sales", "Research & Development", "Human Resources"])
job_role         = st.sidebar.selectbox("Job Role", [
    "Sales Executive", "Research Scientist", "Laboratory Technician",
    "Manufacturing Director", "Healthcare Representative", "Manager",
    "Sales Representative", "Research Director", "Human Resources"])
job_level        = st.sidebar.slider("Job Level", 1, 5, 2)
job_involvement  = st.sidebar.slider("Job Involvement (1–4)", 1, 4, 3)
job_satisfaction = st.sidebar.slider("Job Satisfaction (1–4)", 1, 4, 3)
env_satisfaction = st.sidebar.slider("Environment Satisfaction (1–4)", 1, 4, 3)
rel_satisfaction = st.sidebar.slider("Relationship Satisfaction (1–4)", 1, 4, 3)
work_life        = st.sidebar.slider("Work-Life Balance (1–4)", 1, 4, 3)
business_travel  = st.sidebar.selectbox("Business Travel",
    ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
overtime         = st.sidebar.selectbox("Overtime", ["No", "Yes"])

st.sidebar.subheader("Experience & Pay")
total_years       = st.sidebar.slider("Total Working Years",          0, 40, 10)
years_at_company  = st.sidebar.slider("Years at Company",             0, 40, 5)
years_in_role     = st.sidebar.slider("Years in Current Role",        0, 20, 3)
years_since_promo = st.sidebar.slider("Years Since Last Promotion",   0, 15, 1)
years_with_mgr    = st.sidebar.slider("Years With Current Manager",   0, 20, 3)
num_companies     = st.sidebar.slider("Num Companies Worked",         0, 10, 2)
training_last_yr  = st.sidebar.slider("Training Times Last Year",     0, 6,  2)
monthly_income    = st.sidebar.number_input("Monthly Income ($)",     1000, 20000, 5000, step=100)
hourly_rate       = st.sidebar.number_input("Hourly Rate",            30,   100,   65,   step=1)
daily_rate        = st.sidebar.number_input("Daily Rate",             100,  1500,  800,  step=10)
monthly_rate      = st.sidebar.number_input("Monthly Rate",           2000, 27000, 14000,step=100)
percent_hike      = st.sidebar.slider("Percent Salary Hike",         11, 25, 14)
perf_rating       = st.sidebar.selectbox("Performance Rating",        ["Excellent (3)", "Outstanding (4)"])
stock_option      = st.sidebar.slider("Stock Option Level (0–3)",     0, 3, 1)

# ── Build raw DataFrame matching original CSV columns (before get_dummies) ────
edu_num  = int(education.split("(")[1].replace(")", ""))
perf_num = int(perf_rating.split("(")[1].replace(")", ""))

raw = pd.DataFrame([{
    "Age":                     int(age),
    "BusinessTravel":          business_travel,
    "DailyRate":               int(daily_rate),
    "Department":              department,
    "DistanceFromHome":        int(distance_home),
    "Education":               edu_num,
    "EducationField":          education_field,
    "EmployeeCount":           1,         # constant in dataset
    "EmployeeNumber":          1,         # not predictive but was in X
    "EnvironmentSatisfaction": int(env_satisfaction),
    "Gender":                  gender,
    "HourlyRate":              int(hourly_rate),
    "JobInvolvement":          int(job_involvement),
    "JobLevel":                int(job_level),
    "JobRole":                 job_role,
    "JobSatisfaction":         int(job_satisfaction),
    "MaritalStatus":           marital_status,
    "MonthlyIncome":           int(monthly_income),
    "MonthlyRate":             int(monthly_rate),
    "NumCompaniesWorked":      int(num_companies),
    "Over18":                  "Y",       # constant in dataset
    "OverTime":                overtime,
    "PercentSalaryHike":       int(percent_hike),
    "PerformanceRating":       perf_num,
    "RelationshipSatisfaction":int(rel_satisfaction),
    "StandardHours":           80,        # constant in dataset
    "StockOptionLevel":        int(stock_option),
    "TotalWorkingYears":       int(total_years),
    "TrainingTimesLastYear":   int(training_last_yr),
    "WorkLifeBalance":         int(work_life),
    "YearsAtCompany":          int(years_at_company),
    "YearsInCurrentRole":      int(years_in_role),
    "YearsSinceLastPromotion": int(years_since_promo),
    "YearsWithCurrManager":    int(years_with_mgr),
}])

# ── Apply get_dummies(drop_first=True) — same as your notebook ────────────────
raw_dummies = pd.get_dummies(raw, drop_first=True)

# ── These are the expected dummy columns after get_dummies(drop_first=True) ───
# Generated from IBM HR dataset categorical columns:
# BusinessTravel: Non-Travel(base) → Travel_Frequently, Travel_Rarely
# Department: Human Resources(base) → Research & Development, Sales
# EducationField: Human Resources(base) → Life Sciences, Marketing, Medical, Other, Technical Degree
# Gender: Female(base) → Male
# JobRole: Healthcare Representative(base) → Human Resources, Laboratory Technician,
#          Manager, Manufacturing Director, Research Director, Research Scientist,
#          Sales Executive, Sales Representative
# MaritalStatus: Divorced(base) → Married, Single
# Over18: N(base) → Y
# OverTime: No(base) → Yes

expected_dummies = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount',
    'EmployeeNumber', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
    'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
    'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
    'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
    'YearsWithCurrManager',
    'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
    'Department_Research & Development', 'Department_Sales',
    'EducationField_Life Sciences', 'EducationField_Marketing',
    'EducationField_Medical', 'EducationField_Other',
    'EducationField_Technical Degree',
    'Gender_Male',
    'JobRole_Human Resources', 'JobRole_Laboratory Technician',
    'JobRole_Manager', 'JobRole_Manufacturing Director',
    'JobRole_Research Director', 'JobRole_Research Scientist',
    'JobRole_Sales Executive', 'JobRole_Sales Representative',
    'MaritalStatus_Married', 'MaritalStatus_Single',
    'Over18_Y',
    'OverTime_Yes',
]

# Align columns to match training data (add missing cols as 0, drop extras)
for col in expected_dummies:
    if col not in raw_dummies.columns:
        raw_dummies[col] = 0

input_data = raw_dummies[expected_dummies]

# ── Summary ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Age",       age)
c2.metric("Income",    f"${monthly_income:,}")
c3.metric("Yrs @ Co.", years_at_company)
c4.metric("Overtime",  overtime)
st.divider()

if st.button("🔍 Predict Attrition", use_container_width=True, type="primary"):
    if model is None:
        st.error("⚠️ model.pkl not found. Add your trained model file.")
    else:
        try:
            data = input_data.copy()
            if scaler is not None:
                data = pd.DataFrame(scaler.transform(data), columns=input_data.columns)

            pred  = model.predict(data)[0]
            proba = model.predict_proba(data)[0] if hasattr(model, "predict_proba") else None

            if pred == 1:
                st.markdown('<div class="leave-box">🚨 HIGH ATTRITION RISK — Employee likely to leave</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="stay-box">✅ LOW RISK — Employee likely to stay</div>',
                            unsafe_allow_html=True)

            if proba is not None:
                st.subheader("Confidence")
                col1, col2 = st.columns(2)
                col1.metric("Stay",  f"{proba[0]*100:.1f}%")
                col2.metric("Leave", f"{proba[1]*100:.1f}%")
                st.progress(float(proba[1]))

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("💡 Ensure model.pkl (and optionally scaler.pkl) are in the same folder as app.py")

with st.expander("📋 Feature vector sent to model (after get_dummies)"):
    st.dataframe(input_data, use_container_width=True)

with st.expander("ℹ️ Model Info"):
    st.markdown("""
    **Dataset:** IBM HR Employee Attrition (1470 rows, 35 original features)  
    **Best Model:** Logistic Regression (F1 best score)  
    **Preprocessing:** get_dummies(drop_first=True) → StandardScaler → SMOTE (on train set)  
    **Total features after encoding:** 48 dummy columns  
    """)
