# ⚡ TalentPulse AI — Job Market Intelligence Platform

> Track hiring trends, skill demand, salary insights and market opportunities across India's tech landscape.

---

## 🚀 Overview

TalentPulse AI is a 7-page Streamlit data product that gives students, job seekers, recruiters, and career counselors a real-time intelligence view of India's tech job market.

Built with **Python + Streamlit + Plotly + scikit-learn** on a synthetic 3,000-record dataset modeled on real India tech hiring patterns.

---

## 📸 Pages

| # | Page | Description |
|---|------|-------------|
| 1 | **Executive Dashboard** | KPI cards, market pulse storyline, hiring trends |
| 2 | **Location Intelligence** | Hiring heatmap, city-wise salary & role breakdown |
| 3 | **Role Intelligence** | Per-role deep-dive: salary progression, skills, city split |
| 4 | **Skill Intelligence** | Top skills, trending skills, role-skill heatmap, correlation |
| 5 | **Salary Intelligence** | Salary by experience, role, city, distribution violin plots |
| 6 | **Career Predictor** | ML-powered role prediction + salary forecast + skill gap |
| 7 | **Opportunity Finder** | Curated roles, skills, cities + opportunity matrix scatter |

---

## 🛠 Tech Stack

- **Frontend**: Streamlit (custom CSS — Space Grotesk + Inter, Deep Blue/Purple/Cyan theme)
- **Charts**: Plotly (bar, violin, pie, heatmap, scatter, line)
- **ML Models**: RandomForestRegressor (salary), RandomForestClassifier (role)
- **Data**: Synthetic 3,000-record dataset (Kaggle-inspired India tech job patterns)

---

## ⚡ Setup & Run

```bash
# 1. Clone / unzip
cd talentpulse

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate data + train models (first time only)
python data/generate_data.py
python models/train_models.py

# 4. Launch app
streamlit run app.py
```

### Live Demo

Click here:- [https://talentpulse-ai-nur9vuy5jesmnrerwdzfyo.streamlit.app/](url)
---

## 📂 File Structure

```
talentpulse/
├── app.py                  # Main Streamlit app (all 7 pages)
├── requirements.txt
├── README.md
├── data/
│   ├── generate_data.py    # Synthetic data generator
│   └── jobs.csv            # Generated dataset (3,000 records)
└── models/
    ├── train_models.py     # Model training script
    └── models.pkl          # Trained RandomForest models + encoders
```

---

## 🎨 Design System

| Token | Value |
|-------|-------|
| Canvas | `#0A0F1E` |
| Panel | `#1E2D5A` |
| Primary accent | `#6C3FD4` (purple) |
| Highlight | `#00C9FF` (cyan) |
| Text | `#E8EDF8` |
| Muted | `#94A3C4` |
| Display font | Space Grotesk |
| Body font | Inter |

---

## 🤖 ML Models

**Salary Predictor** — `RandomForestRegressor`
- Features: city, experience, education, primary skill, demand score
- Target: salary in LPA

**Role Classifier** — `RandomForestClassifier`
- Same features → predicts best-fit role
- Rule-based skill gap analysis overlaid on top

---

## 👩‍💻 Author

**Lakshita** — MCA Data Analytics, Jain University  
GitHub: [stakefire99](https://github.com/stakefire99) · LinkedIn: [lakshita-047439224](https://linkedin.com/in/lakshita-047439224/)
