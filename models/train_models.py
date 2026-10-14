"""Train the TalentPulse salary and role models.

Run from the project root:   python models/train_models.py
Prints held-out test metrics and saves models/models.pkl.
"""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE, "data", "jobs.csv"))

# ---- Encoders -------------------------------------------------------------
le_role, le_city, le_exp, le_edu, le_skill = (LabelEncoder() for _ in range(5))
df["role_enc"] = le_role.fit_transform(df["role"])
df["city_enc"] = le_city.fit_transform(df["city"])
df["exp_enc"] = le_exp.fit_transform(df["experience"])
df["edu_enc"] = le_edu.fit_transform(df["education"])
df["skill_enc"] = le_skill.fit_transform(df["primary_skill"])

# NOTE: demand_score is deliberately NOT a feature. In this dataset it is a
# fixed value per role, so using it would leak the answer to the role model.
features = ["city_enc", "exp_enc", "edu_enc", "skill_enc"]
X = df[features]

# ---- Salary model (regression) -------------------------------------------
Xtr, Xte, ytr, yte = train_test_split(X, df["salary_lpa"], test_size=0.2, random_state=42)
sal_eval = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=42).fit(Xtr, ytr)
pred = sal_eval.predict(Xte)
r2 = r2_score(yte, pred)
mae = mean_absolute_error(yte, pred)

# ---- Role model (classification) -----------------------------------------
Xtr, Xte, ytr, yte = train_test_split(
    X, df["role_enc"], test_size=0.2, random_state=42, stratify=df["role_enc"]
)
role_eval = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, random_state=42).fit(Xtr, ytr)
acc = accuracy_score(yte, role_eval.predict(Xte))
baseline = 1 / df["role"].nunique()

print(f"Salary model  - R2: {r2:.3f} | MAE: {mae:.2f} LPA (20% held-out test set)")
print(f"Role model    - accuracy: {acc:.1%} vs {baseline:.0%} random guessing ({df['role'].nunique()} roles)")

# ---- Refit on all data for the app ---------------------------------------
sal_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=42).fit(X, df["salary_lpa"])
role_model = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, random_state=42).fit(X, df["role_enc"])

models = {
    "salary_model": sal_model,
    "role_model": role_model,
    "le_role": le_role,
    "le_city": le_city,
    "le_exp": le_exp,
    "le_edu": le_edu,
    "le_skill": le_skill,
    "features": features,
    "metrics": {"salary_r2": r2, "salary_mae_lpa": mae, "role_accuracy": acc, "role_baseline": baseline},
}
with open(os.path.join(BASE, "models", "models.pkl"), "wb") as f:
    pickle.dump(models, f)

print("Models trained and saved to models/models.pkl")
