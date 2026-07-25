import os
import pandas as pd
import streamlit as st
import xgboost as xgb

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="centered"
)

# Ensure working directory is the current file location
os.chdir(os.path.dirname(__file__))

# ---------------- LOAD MODEL ---------------- #
model = xgb.XGBRegressor()
model.load_model("xg_model.json")

# ---------------- APP ---------------- #
st.title("💼 Employee Salary Prediction")

st.write("""
This application predicts the estimated annual salary of an employee using an
XGBoost Machine Learning model.

Fill in the employee details below and click **Predict Salary**.
""")

st.sidebar.header("About")
st.sidebar.info("""
**Model:** XGBoost Regressor

**Features Used:** 14

**Output:** Estimated Employee Salary
""")

# ---------------- INPUTS ---------------- #

age = st.number_input(
    "Age (Years)",
    min_value=18,
    max_value=70,
    step=1
)

experience = st.number_input(
    "Years of Experience",
    min_value=0,
    max_value=50,
    step=1
)

education_options = {
    "High School": 0,
    "Bachelor's Degree": 1,
    "Master's Degree": 2,
    "PhD": 3
}

education = st.selectbox(
    "Education Level",
    list(education_options.keys())
)

department_options = {
    "Operations": 0,
    "Marketing": 1,
    "Sales": 2,
    "IT": 3,
    "HR": 4
}

department = st.selectbox(
    "Department",
    list(department_options.keys())
)

city_options = {
    "Chennai": 0,
    "Delhi": 1,
    "Mumbai": 2,
    "Hyderabad": 3
}

city = st.selectbox(
    "City",
    list(city_options.keys())
)

job_level_options = {
    "Senior": 0,
    "Mid": 1,
    "Junior": 2
}

job_level = st.selectbox(
    "Job Level",
    list(job_level_options.keys())
)

gender_options = {
    "Male": 0,
    "Female": 1
}

gender = st.selectbox(
    "Gender",
    list(gender_options.keys())
)

performance = st.number_input(
    "Performance Rating",
    min_value=1,
    max_value=5,
    step=1
)

certifications = st.number_input(
    "Number of Certifications",
    min_value=0,
    max_value=10,
    step=1
)

overtime = st.number_input(
    "Overtime Hours",
    min_value=0,
    max_value=100,
    step=1
)

remote_options = {
    "Yes": 0,
    "No": 1
}

remote_work = st.selectbox(
    "Remote Work",
    list(remote_options.keys())
)

tenure = st.number_input(
    "Company Tenure (Years)",
    min_value=0,
    max_value=20,
    step=1
)

projects = st.number_input(
    "Projects Completed",
    min_value=0,
    max_value=50,
    step=1
)

skill_score = st.number_input(
    "Skill Score",
    min_value=0,
    max_value=100,
    step=1
)

# ---------------- CREATE DATAFRAME ---------------- #

employee_data = pd.DataFrame({
    "Age": age,
    "Education": education_options[education],
    "Gender": gender_options[gender],
    "Department": department_options[department],
    "Job_Level": job_level_options[job_level],
    "Experience_Years": experience,
    "Performance_Rating": performance,
    "Certifications": certifications,
    "Overtime_Hours": overtime,
    "Remote_Work": remote_options[remote_work],
    "City": city_options[city],
    "Company_Tenure": tenure,
    "Projects_Completed": projects,
    "Skill_Score": skill_score
}, index=[0])

st.subheader("Entered Employee Details")
st.dataframe(employee_data, use_container_width=True)

# ---------------- PREDICTION ---------------- #

if st.button("Predict Salary"):

    with st.spinner("Predicting salary..."):

        prediction = model.predict(employee_data)

    salary = round(float(prediction[0]), 2)

    st.success(f"💰 Estimated Annual Salary: ₹{salary:,.2f} Lakhs")
