# Credit Card Approval Prediction

A complete machine learning project that predicts whether a credit card
application would be **approved** or **denied**, with an interactive
Streamlit dashboard for exploring the data, comparing models, and getting
live predictions.


## Project Structure

```
credit_card_approval/
├── data_generator.py     # Generates the synthetic dataset
├── train_models.py        # Trains & evaluates 3 ML models
├── app.py                 # Streamlit dashboard (optional)
├── flask_app.py            # Flask dashboard (no Streamlit needed)
├── templates/              # HTML templates for the Flask app
│   ├── base.html
│   ├── overview.html
│   ├── model_comparison.html
│   ├── feature_importance.html
│   └── predict.html
├── static/
│   └── style.css
├── requirements.txt
├── data/
│   └── credit_card_data.csv
└── models/
    ├── logistic_regression.joblib
    ├── random_forest.joblib
    ├── xgboost.joblib
    └── metrics.json
```

## Setup

```bash
# 1. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate data + train models (only needed once, or after code changes)
python data_generator.py
python train_models.py

# 4. Launch the dashboard - choose ONE of the following:

# Option A: Flask dashboard (no Streamlit required)
python flask_app.py
# then open http://localhost:5000

# Option B: Streamlit dashboard
streamlit run app.py
# then open http://localhost:8501
```

Both dashboards offer the same four views: Overview, Model Comparison,
Feature Importance, and Live Prediction. Use whichever framework you
prefer — they run independently and share the same data/model files.

## What's Inside

### 1. Synthetic Data Generation (`data_generator.py`)
Creates 5,000 realistic applicant records with features like age, income,
credit score, employment status, education, existing debt, and more. The
approval label is generated from a weighted, noisy scoring function so the
patterns feel realistic without using any real customer data.

### 2. Model Training (`train_models.py`)
Trains and compares three classifiers on an 80/20 train-test split:
- **Logistic Regression** — simple, interpretable baseline
- **Random Forest** — captures non-linear feature interactions
- **XGBoost** — gradient-boosted trees, typically strongest performance

Each model is wrapped in a `scikit-learn` `Pipeline` with preprocessing
(scaling numeric features, one-hot encoding categorical ones), so raw
applicant data can be fed straight to the model in the dashboard. Metrics
(accuracy, precision, recall, F1, ROC-AUC, confusion matrix) and feature
importances are saved to `models/metrics.json`.

### 3. Streamlit Dashboard (`app.py`)
Four pages, accessible from the sidebar:
- **Overview** — dataset stats and distribution charts
- **Model Comparison** — side-by-side metrics and confusion matrices
- **Feature Importance** — which factors most influence approval
- **Live Prediction** — fill in an applicant's profile and get an instant
  approval prediction + confidence score from any of the three models

## Retraining with New Data

If you want to regenerate the dataset with a different size or plug in
real (properly anonymized) data, just update `data_generator.py` or point
`train_models.py` at your own CSV — as long as it has the same column
names, everything downstream (training + dashboard) will work unchanged.

## Notes

- All models are cached with `@st.cache_resource` so the dashboard loads
  instantly without retraining on every interaction.
- Debt-to-income ratio is automatically computed from income and debt
  inputs in the live prediction form.
