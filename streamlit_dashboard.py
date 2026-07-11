"""
app.py
------
Streamlit dashboard for the Credit Card Approval Prediction project.

Run with:
    streamlit run app.py

Features:
  - Overview of the synthetic dataset
  - Model comparison (Logistic Regression vs Random Forest vs XGBoost)
  - Feature importance explorer
  - Live prediction interface (fill in an applicant's details, get a
    real-time approval prediction + probability from any trained model)
"""

import os
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_generator import generate_data

# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Credit Card Approval Predictor",
    page_icon="💳",
    layout="wide",
)

MODEL_DIR = "models"
DATA_PATH = os.path.join("data", "credit_card_data.csv")


# ------------------------------------------------------------------ #
# Cached loaders
# ------------------------------------------------------------------ #
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        df = generate_data(5000)
        os.makedirs("data", exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)
    return df


@st.cache_resource
def load_models():
    models = {}
    for name, fname in [
        ("Logistic Regression", "logistic_regression.joblib"),
        ("Random Forest", "random_forest.joblib"),
        ("XGBoost", "xgboost.joblib"),
    ]:
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


@st.cache_data
def load_metrics():
    path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


df = load_data()
models = load_models()
metadata = load_metrics()

if not models or metadata is None:
    st.error(
        "No trained models found. Please run `python train_models.py` first "
        "to generate data and train the models."
    )
    st.stop()

# ------------------------------------------------------------------ #
# Sidebar navigation
# ------------------------------------------------------------------ #
st.sidebar.title("💳 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Model Comparison", "Feature Importance", "Live Prediction"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Credit Card Approval Prediction\n\n"
    "Synthetic data · Logistic Regression, Random Forest & XGBoost"
)

# ------------------------------------------------------------------ #
# PAGE: Overview
# ------------------------------------------------------------------ #
if page == "Overview":
    st.title("💳 Credit Card Approval Prediction Dashboard")
    st.markdown(
        "This dashboard explores a **synthetic** credit card application "
        "dataset and compares machine learning models trained to predict "
        "approval outcomes."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Applications", f"{len(df):,}")
    c2.metric("Approval Rate", f"{df['approved'].mean() * 100:.1f}%")
    c3.metric("Avg. Credit Score", f"{df['credit_score'].mean():.0f}")
    c4.metric("Avg. Annual Income", f"${df['annual_income'].mean():,.0f}")

    st.markdown("### Dataset Sample")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("### Distributions")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df, x="credit_score", color="approved",
            barmode="overlay", nbins=40,
            title="Credit Score Distribution by Approval",
            color_discrete_map={0: "#d62728", 1: "#2ca02c"},
        )
        fig.update_layout(legend_title_text="Approved")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            df, x="annual_income", color="approved",
            barmode="overlay", nbins=40,
            title="Annual Income Distribution by Approval",
            color_discrete_map={0: "#d62728", 1: "#2ca02c"},
        )
        fig.update_layout(legend_title_text="Approved")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        emp_approval = (
            df.groupby("employment_status")["approved"].mean().reset_index()
        )
        fig = px.bar(
            emp_approval, x="employment_status", y="approved",
            title="Approval Rate by Employment Status",
            labels={"approved": "Approval Rate"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        edu_approval = df.groupby("education")["approved"].mean().reset_index()
        fig = px.bar(
            edu_approval, x="education", y="approved",
            title="Approval Rate by Education Level",
            labels={"approved": "Approval Rate"},
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ #
# PAGE: Model Comparison
# ------------------------------------------------------------------ #
elif page == "Model Comparison":
    st.title("📊 Model Comparison")

    results = metadata["results"]
    metrics_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    metrics_df = metrics_df[["Model", "accuracy", "precision", "recall", "f1", "roc_auc"]]

    st.markdown(f"**Best model by ROC-AUC:** 🏆 `{metadata['best_model']}`")
    st.dataframe(metrics_df.set_index("Model"), use_container_width=True)

    melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(
        melted, x="Metric", y="Score", color="Model", barmode="group",
        title="Model Performance Comparison",
        range_y=[0, 1],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Confusion Matrices")
    cols = st.columns(len(results))
    for col, (name, res) in zip(cols, results.items()):
        cm = np.array(res["confusion_matrix"])
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Predicted: Denied", "Predicted: Approved"],
                y=["Actual: Denied", "Actual: Approved"],
                text=cm,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            )
        )
        fig.update_layout(title=name, height=350, margin=dict(t=40, b=10))
        col.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ #
# PAGE: Feature Importance
# ------------------------------------------------------------------ #
elif page == "Feature Importance":
    st.title("🔍 Feature Importance (Random Forest)")

    fi_df = pd.DataFrame(metadata["feature_importance"])
    fi_df = fi_df.sort_values("importance", ascending=True).tail(20)

    fig = px.bar(
        fi_df, x="importance", y="feature", orientation="h",
        title="Top Features Driving Approval Decisions",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Importance values come from the Random Forest model and reflect how "
        "much each feature (after encoding) reduces prediction uncertainty. "
        "Encoded categorical features appear as `cat__<column>_<category>`."
    )

# ------------------------------------------------------------------ #
# PAGE: Live Prediction
# ------------------------------------------------------------------ #
elif page == "Live Prediction":
    st.title("🎯 Live Approval Prediction")
    st.markdown("Fill in applicant details to get a real-time prediction.")

    model_choice = st.selectbox("Choose a model", list(models.keys()))

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Age", 18, 80, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            education = st.selectbox(
                "Education", ["High School", "Bachelors", "Masters", "PhD"]
            )
            marital_status = st.selectbox(
                "Marital Status", ["Single", "Married", "Divorced"]
            )

        with col2:
            employment_status = st.selectbox(
                "Employment Status",
                ["Employed", "Self-Employed", "Unemployed", "Retired"],
            )
            years_employed = st.slider("Years Employed", 0.0, 40.0, 5.0)
            annual_income = st.number_input(
                "Annual Income ($)", min_value=0, value=50000, step=1000
            )
            credit_score = st.slider("Credit Score", 300, 850, 650)

        with col3:
            existing_debt = st.number_input(
                "Existing Debt ($)", min_value=0, value=5000, step=500
            )
            num_credit_cards = st.slider("Number of Existing Credit Cards", 0, 10, 2)
            num_dependents = st.slider("Number of Dependents", 0, 6, 0)
            has_property = st.selectbox("Owns Property?", ["Yes", "No"]) == "Yes"

        submitted = st.form_submit_button("Predict Approval", use_container_width=True)

    if submitted:
        debt_to_income = existing_debt / (annual_income + 1)

        input_df = pd.DataFrame(
            [
                {
                    "age": age,
                    "years_employed": years_employed,
                    "annual_income": annual_income,
                    "credit_score": credit_score,
                    "existing_debt": existing_debt,
                    "debt_to_income": debt_to_income,
                    "num_credit_cards": num_credit_cards,
                    "num_dependents": num_dependents,
                    "has_property": int(has_property),
                    "gender": gender,
                    "education": education,
                    "marital_status": marital_status,
                    "employment_status": employment_status,
                }
            ]
        )

        model = models[model_choice]
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        st.markdown("---")
        result_col, gauge_col = st.columns([1, 1])

        with result_col:
            if pred == 1:
                st.success(f"### ✅ Approved\n**Confidence: {proba * 100:.1f}%**")
            else:
                st.error(f"### ❌ Denied\n**Confidence: {(1 - proba) * 100:.1f}%**")

            st.markdown(f"**Model used:** {model_choice}")
            st.markdown(f"**Debt-to-income ratio:** {debt_to_income:.2f}")

        with gauge_col:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    title={"text": "Approval Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#2ca02c" if pred == 1 else "#d62728"},
                        "steps": [
                            {"range": [0, 40], "color": "#fde0dd"},
                            {"range": [40, 60], "color": "#fee8c8"},
                            {"range": [60, 100], "color": "#e5f5e0"},
                        ],
                    },
                )
            )
            fig.update_layout(height=300, margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
