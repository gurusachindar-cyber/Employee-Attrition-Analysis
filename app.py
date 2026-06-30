import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------------
# 1. GENERATE DUMMY HR DATA (If you don't have a CSV file)
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Age': np.random.randint(22, 60, n_samples),
        'Department': np.random.choice(['Sales', 'Engineering', 'HR', 'Marketing'], n_samples),
        'MonthlySalary': np.random.randint(3000, 15000, n_samples),
        'YearsAtCompany': np.random.randint(1, 15, n_samples),
        'WorkLifeBalance': np.random.randint(1, 5, n_samples), # 1-Bad, 4-Best
        'JobSatisfaction': np.random.randint(1, 5, n_samples),
        'Attrition': np.random.choice([0, 1], n_samples, p=[0.84, 0.16]) # 16% turnover rate
    }
    
    # Adjusting data artificially so patterns exist for the AI to learn
    df = pd.DataFrame(data)
    df.loc[df['MonthlySalary'] < 4500, 'Attrition'] = np.random.choice([0, 1], len(df[df['MonthlySalary'] < 4500]), p=[0.4, 0.6])
    df.loc[df['JobSatisfaction'] == 1, 'Attrition'] = np.random.choice([0, 1], len(df[df['JobSatisfaction'] == 1]), p=[0.5, 0.5])
    return df

df = load_data()

# ------------------------------------------------------------------
# 2. DATA PREPROCESSING & MODEL TRAINING
# ------------------------------------------------------------------
# Encode categorical data (Department)
le = LabelEncoder()
df_model = df.copy()
df_model['Department'] = le.fit_transform(df_model['Department'])

# Features and Target
X = df_model.drop('Attrition', axis=1)
y = df_model['Attrition']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Classifier (Random Forest)
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ------------------------------------------------------------------
# 3. STREAMLIT HR DASHBOARD UI
# ------------------------------------------------------------------
st.set_page_config(page_title="HR Attrition Dashboard", layout="wide")

st.title("Employee Attrition Analysis & Prediction Dashboard")
st.markdown("This dashboard helps HR teams understand key turnover drivers and predict employee flight risks.")

st.divider()

# --- KPI METRICS ---
col1, col2, col3, col4 = st.columns(4)
total_employees = len(df)
attrition_count = df['Attrition'].sum()
attrition_rate = (attrition_count / total_employees) * 100
avg_salary = df['MonthlySalary'].mean()

with col1:
    st.metric(label="Total Employees", value=total_employees)
with col2:
    st.metric(label="Total Attrition (Resigned)", value=int(attrition_count), delta_color="inverse")
with col3:
    st.metric(label="Attrition Rate", value=f"{attrition_rate:.2f}%")
with col4:
    st.metric(label="Avg Monthly Salary", value=f"${avg_salary:,.2f}")

st.divider()

# --- VISUALIZATIONS (EDA) ---
st.subheader("Factors Influencing Resignation")

col_left, col_right = st.columns(2)

with col_left:
    st.write("### Attrition by Department")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='Department', hue='Attrition', palette='Set2', ax=ax)
    ax.set_ylabel("Employee Count")
    st.pyplot(fig)

with col_right:
    st.write("### Salary Distribution vs Attrition")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='Attrition', y='MonthlySalary', palette='pastel', ax=ax)
    ax.set_xticklabels(['Stayed (0)', 'Left (1)'])
    st.pyplot(fig)

st.divider()

# --- PREDICTION SECTION ---
st.subheader(" Predict Employee Attrition Risk")
st.markdown("Enter employee details below to calculate the likelihood of them leaving the company.")

p_col1, p_col2, p_col3 = st.columns(3)

with p_col1:
    age = st.slider("Age", 18, 65, 30)
    dept = st.selectbox("Department", ['Sales', 'Engineering', 'HR', 'Marketing'])

with p_col2:
    salary = st.number_input("Monthly Salary ($)", min_value=1000, max_value=20000, value=5000)
    years = st.slider("Years at Company", 0, 20, 3)

with p_col3:
    satisfaction = st.slider("Job Satisfaction (1 = Low, 4 = High)", 1, 4, 3)
    balance = st.slider("Work-Life Balance (1 = Low, 4 = High)", 1, 4, 3)

# Predict button logic
if st.button("Predict Attrition Risk", type="primary"):
    # Prepare input vector
    dept_encoded = le.transform([dept])[0]
    input_data = pd.DataFrame([[age, dept_encoded, salary, years, balance, satisfaction]], 
                              columns=['Age', 'Department', 'MonthlySalary', 'YearsAtCompany', 'WorkLifeBalance', 'JobSatisfaction'])
    
    # Make Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100
    
    st.subheader("Result:")
    if prediction == 1:
        st.error(f"**High Risk Alert!** This employee is likely to leave. (Turnover Probability: {probability:.1f}%)")
    else:
        st.success(f"**Safe!** This employee is likely to stay. (Turnover Probability: {probability:.1f}%)")

st.sidebar.write(f"**Model Accuracy:** {accuracy*100:.2f}%")
st.sidebar.markdown("""
### Quick Guide:
1. Review KPIs at the top.
2. Analyze charts to spot trends (e.g., low-salary turnover).
3. Use the interactive form to evaluate real-time flight risk.
""")