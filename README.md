# Employee Attrition Analysis & Prediction Dashboard

An advanced HR Analytics project built to understand, visualize, and predict employee turnover using machine learning classification algorithms. This project features an interactive KPI dashboard designed for HR teams to make data-driven decisions.

---

##  Developed By
* **Name:** s.gurusachindar
* **Role:** Machine Learning & Data Analytics Engineer

---

##  Project Goal
Employee attrition is a critical challenge for modern organizations. Replacing a talented employee incurs significant recruitment, onboarding, and training costs. 
The goal of this project is to:
1. Identify the key driving factors behind employee resignations (e.g., age, monthly salary, department, work-life balance).
2. Build an automated predictive model to identify high-risk flight profiles before they submit resignations.
3. Design a user-friendly executive dashboard with key metrics and actionable visualization insights.

---

## Key Features

### 1. HR Executive Dashboard (KPIs)
* **Total Employees:** Tracks the absolute workforce count being analyzed.
* **Attrition Rate (%):** Calculates active turnover percentage to highlight overall workforce stability.
* **Average Salary Mapping:** Provides benchmark tracking to evaluate internal compensation trends.

### 2. Exploratory Data Analysis (EDA)
* **Attrition by Department:** Visualizes which specific company departments face the highest turnover.
* **Salary vs Attrition Box Plot:** Dynamically captures the impact of lower income limits on employee retention.

### 3. Interactive Prediction Engine
* Input forms (sliders, drop-downs) allowing HR users to test individual employee retention probability.
* Real-time calculation using trained machine learning architectures.

---

##  Tech Stack & Architecture

* **Language:** Python 3.x
* **Frontend/Dashboard:** Streamlit Framework
* **Data Processing:** Pandas, NumPy
* **Data Visualization:** Seaborn, Matplotlib
* **Machine Learning:** Scikit-Learn
* **Algorithm Implemented:** Random Forest Classifier

---

##  Machine Learning Workflow

1. **Dataset Structuring:** Generated or processed standard HR operational fields.
2. **Feature Engineering:** Categorical features (Departments) were preprocessed using `LabelEncoder`.
3. **Data Splitting:** Divided data into Train and Test datasets (80/20 ratio) to prevent overfitting.
4. **Model Training:** Fit a **Random Forest Classifier** utilizing an ensemble of 100 decision trees.
5. **Validation:** Achieved reliable prediction benchmarks on cross-validation testing.

---

##  Installation & Usage Guide

Follow these simple steps to host this dashboard locally on your computer:

### Step 1: Clone the Project
```bash
git clone [https://github.com/YOUR_USERNAME/Employee-Attrition-Analysis.git](https://github.com/YOUR_USERNAME/Employee-Attrition-Analysis.git)
cd Employee-Attrition-Analysis
