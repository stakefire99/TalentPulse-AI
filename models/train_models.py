import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

df = pd.read_csv("/home/claude/talentpulse/data/jobs.csv")

# Encoders
le_role = LabelEncoder()
le_city = LabelEncoder()
le_exp = LabelEncoder()
le_edu = LabelEncoder()
le_skill = LabelEncoder()

df["role_enc"] = le_role.fit_transform(df["role"])
df["city_enc"] = le_city.fit_transform(df["city"])
df["exp_enc"] = le_exp.fit_transform(df["experience"])
df["edu_enc"] = le_edu.fit_transform(df["education"])
df["skill_enc"] = le_skill.fit_transform(df["primary_skill"])

features = ["city_enc", "exp_enc", "edu_enc", "skill_enc", "demand_score"]

# Salary model
X_sal = df[features]
y_sal = df["salary_lpa"]
sal_model = RandomForestRegressor(n_estimators=100, random_state=42)
sal_model.fit(X_sal, y_sal)

# Role classifier
X_role = df[features]
y_role = df["role_enc"]
role_model = RandomForestClassifier(n_estimators=100, random_state=42)
role_model.fit(X_role, y_role)

# Save everything
os.makedirs("/home/claude/talentpulse/models", exist_ok=True)
models = {
    "salary_model": sal_model,
    "role_model": role_model,
    "le_role": le_role,
    "le_city": le_city,
    "le_exp": le_exp,
    "le_edu": le_edu,
    "le_skill": le_skill,
    "features": features,
}
with open("/home/claude/talentpulse/models/models.pkl", "wb") as f:
    pickle.dump(models, f)

print("Models trained and saved.")
print("Roles:", list(le_role.classes_))
print("Cities:", list(le_city.classes_))
print("Experience:", list(le_exp.classes_))
