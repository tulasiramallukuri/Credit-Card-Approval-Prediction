"""
data_generator.py
------------------
Generates a realistic synthetic dataset for the Credit Card Approval
Prediction project. No real customer data is used anywhere.

Run directly to create data/credit_card_data.csv:
    python data_generator.py
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)


def generate_data(n_samples: int = 5000) -> pd.DataFrame:
    """Generate a synthetic credit card application dataset."""

    # ---- Demographic & financial features -----------------------------
    age = np.random.randint(21, 70, n_samples)

    gender = np.random.choice(["Male", "Female"], n_samples, p=[0.55, 0.45])

    education = np.random.choice(
        ["High School", "Bachelors", "Masters", "PhD"],
        n_samples,
        p=[0.30, 0.45, 0.20, 0.05],
    )

    employment_status = np.random.choice(
        ["Employed", "Self-Employed", "Unemployed", "Retired"],
        n_samples,
        p=[0.60, 0.20, 0.10, 0.10],
    )

    # Annual income correlates loosely with education/employment
    base_income = np.random.normal(50000, 20000, n_samples)
    edu_bonus = education.astype(object)
    edu_map = {"High School": 0, "Bachelors": 15000, "Masters": 30000, "PhD": 45000}
    edu_bonus = np.array([edu_map[e] for e in education])
    employment_penalty = np.array(
        [0 if e in ("Employed", "Self-Employed") else -20000 for e in employment_status]
    )
    annual_income = np.clip(base_income + edu_bonus + employment_penalty, 8000, None)

    years_employed = np.where(
        employment_status == "Unemployed",
        0,
        np.clip(np.random.normal(6, 5, n_samples), 0, 40),
    )

    credit_score = np.clip(
        np.random.normal(650, 90, n_samples) + (annual_income - 50000) / 2000,
        300,
        850,
    ).astype(int)

    existing_debt = np.clip(np.random.normal(8000, 6000, n_samples), 0, None)

    num_credit_cards = np.random.poisson(2, n_samples)

    num_dependents = np.random.poisson(1, n_samples)

    has_property = np.random.choice([0, 1], n_samples, p=[0.45, 0.55])

    marital_status = np.random.choice(
        ["Single", "Married", "Divorced"], n_samples, p=[0.40, 0.45, 0.15]
    )

    debt_to_income = existing_debt / (annual_income + 1)

    # ---- Target: Approval -----------------------------------------------
    # Build an approval "score" from weighted factors, then threshold + noise
    score = (
        0.0035 * (credit_score - 300)
        + 0.000012 * annual_income
        - 4.0 * debt_to_income
        + 0.05 * years_employed
        + 0.6 * has_property
        - 0.15 * num_dependents
        - (employment_status == "Unemployed").astype(int) * 1.5
        + np.random.normal(0, 0.6, n_samples)  # noise
    )

    approval_prob = 1 / (1 + np.exp(-(score - 2.2)))
    approved = (approval_prob > 0.5).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "education": education,
            "marital_status": marital_status,
            "employment_status": employment_status,
            "years_employed": years_employed.round(1),
            "annual_income": annual_income.round(2),
            "credit_score": credit_score,
            "existing_debt": existing_debt.round(2),
            "debt_to_income": debt_to_income.round(3),
            "num_credit_cards": num_credit_cards,
            "num_dependents": num_dependents,
            "has_property": has_property,
            "approved": approved,
        }
    )

    return df


if __name__ == "__main__":
    df = generate_data(5000)
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "credit_card_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df["approved"].value_counts(normalize=True))
