"""
flask_app.py
------------
Flask version of the Credit Card Approval Prediction dashboard
(no Streamlit required).

Run with:
    python flask_app.py

Then open http://localhost:5000 in your browser.
"""

import os
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils

from flask import Flask, render_template, request, jsonify

from data_generator import generate_data

app = Flask(__name__)

MODEL_DIR = "models"
DATA_PATH = os.path.join("data", "credit_card_data.csv")

CATEGORY_OPTIONS = {
    "gender": ["Male", "Female"],
    "education": ["High School", "Bachelors", "Masters", "PhD"],
    "marital_status": ["Single", "Married", "Divorced"],
    "employment_status": ["Employed", "Self-Employed", "Unemployed", "Retired"],
}


# ------------------------------------------------------------------ #
# Data / model loading (loaded once at startup)
# ------------------------------------------------------------------ #
def load_data():
    if not os.path.exists(DATA_PATH):
        df = generate_data(5000)
        os.makedirs("data", exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)
    return df


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


def load_metrics():
    path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


DF = load_data()
MODELS = load_models()
METADATA = load_metrics()


def fig_to_json(fig):
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


# ------------------------------------------------------------------ #
# Routes
# ------------------------------------------------------------------ #
@app.route("/")
def overview():
    stats = {
        "total": f"{len(DF):,}",
        "approval_rate": f"{DF['approved'].mean() * 100:.1f}%",
        "avg_credit_score": f"{DF['credit_score'].mean():.0f}",
        "avg_income": f"${DF['annual_income'].mean():,.0f}",
    }

    fig1 = px.histogram(
        DF, x="credit_score", color="approved", barmode="overlay", nbins=40,
        title="Credit Score Distribution by Approval",
        color_discrete_map={0: "#d62728", 1: "#2ca02c"},
    )
    fig2 = px.histogram(
        DF, x="annual_income", color="approved", barmode="overlay", nbins=40,
        title="Annual Income Distribution by Approval",
        color_discrete_map={0: "#d62728", 1: "#2ca02c"},
    )
    emp_approval = DF.groupby("employment_status")["approved"].mean().reset_index()
    fig3 = px.bar(
        emp_approval, x="employment_status", y="approved",
        title="Approval Rate by Employment Status",
        labels={"approved": "Approval Rate"},
    )
    edu_approval = DF.groupby("education")["approved"].mean().reset_index()
    fig4 = px.bar(
        edu_approval, x="education", y="approved",
        title="Approval Rate by Education Level",
        labels={"approved": "Approval Rate"},
    )

    sample_table = DF.head(15).to_dict(orient="records")
    columns = list(DF.columns)

    return render_template(
        "overview.html",
        active="overview",
        stats=stats,
        fig1=fig_to_json(fig1),
        fig2=fig_to_json(fig2),
        fig3=fig_to_json(fig3),
        fig4=fig_to_json(fig4),
        sample_table=sample_table,
        columns=columns,
    )


@app.route("/model-comparison")
def model_comparison():
    results = METADATA["results"]
    metrics_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    metrics_df = metrics_df[["Model", "accuracy", "precision", "recall", "f1", "roc_auc"]]

    melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    bar_fig = px.bar(
        melted, x="Metric", y="Score", color="Model", barmode="group",
        title="Model Performance Comparison", range_y=[0, 1],
    )

    cm_figs = {}
    for name, res in results.items():
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
        cm_figs[name] = fig_to_json(fig)

    return render_template(
        "model_comparison.html",
        active="comparison",
        best_model=METADATA["best_model"],
        metrics_table=metrics_df.to_dict(orient="records"),
        bar_fig=fig_to_json(bar_fig),
        cm_figs=cm_figs,
    )


@app.route("/feature-importance")
def feature_importance():
    fi_df = pd.DataFrame(METADATA["feature_importance"])
    fi_df = fi_df.sort_values("importance", ascending=True).tail(20)

    fig = px.bar(
        fi_df, x="importance", y="feature", orientation="h",
        title="Top Features Driving Approval Decisions",
    )
    fig.update_layout(height=600)

    return render_template("feature_importance.html", active="importance", fig=fig_to_json(fig))


@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template(
        "predict.html",
        active="predict",
        models=list(MODELS.keys()),
        category_options=CATEGORY_OPTIONS,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json()

    model_choice = payload.get("model")
    if model_choice not in MODELS:
        return jsonify({"error": "Unknown model"}), 400

    annual_income = float(payload["annual_income"])
    existing_debt = float(payload["existing_debt"])
    debt_to_income = existing_debt / (annual_income + 1)

    input_df = pd.DataFrame(
        [
            {
                "age": int(payload["age"]),
                "years_employed": float(payload["years_employed"]),
                "annual_income": annual_income,
                "credit_score": int(payload["credit_score"]),
                "existing_debt": existing_debt,
                "debt_to_income": debt_to_income,
                "num_credit_cards": int(payload["num_credit_cards"]),
                "num_dependents": int(payload["num_dependents"]),
                "has_property": int(payload["has_property"]),
                "gender": payload["gender"],
                "education": payload["education"],
                "marital_status": payload["marital_status"],
                "employment_status": payload["employment_status"],
            }
        ]
    )

    model = MODELS[model_choice]
    pred = int(model.predict(input_df)[0])
    proba = float(model.predict_proba(input_df)[0][1])

    return jsonify(
        {
            "approved": pred == 1,
            "probability": round(proba * 100, 1),
            "debt_to_income": round(debt_to_income, 3),
            "model_used": model_choice,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
