import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

ROLES = {
    "Data Analyst": {"base_salary": 650000, "growth": 22, "demand": 92},
    "Business Analyst": {"base_salary": 720000, "growth": 18, "demand": 87},
    "Data Scientist": {"base_salary": 950000, "growth": 28, "demand": 95},
    "Python Developer": {"base_salary": 800000, "growth": 24, "demand": 89},
    "ML Engineer": {"base_salary": 1100000, "growth": 35, "demand": 96},
    "Data Engineer": {"base_salary": 920000, "growth": 30, "demand": 93},
    "QA Engineer": {"base_salary": 580000, "growth": 10, "demand": 72},
    "Product Analyst": {"base_salary": 780000, "growth": 20, "demand": 85},
    "BI Developer": {"base_salary": 700000, "growth": 15, "demand": 80},
    "AI/ML Researcher": {"base_salary": 1300000, "growth": 42, "demand": 98},
}

CITIES = {
    "Bangalore": {"weight": 0.35, "salary_mult": 1.15},
    "Hyderabad": {"weight": 0.18, "salary_mult": 1.05},
    "Pune": {"weight": 0.14, "salary_mult": 0.98},
    "Mumbai": {"weight": 0.12, "salary_mult": 1.10},
    "Noida": {"weight": 0.08, "salary_mult": 0.95},
    "Gurgaon": {"weight": 0.07, "salary_mult": 1.00},
    "Chennai": {"weight": 0.04, "salary_mult": 0.92},
    "Kolkata": {"weight": 0.02, "salary_mult": 0.88},
}

COMPANIES = [
    "Infosys", "TCS", "Wipro", "HCL Technologies", "Tech Mahindra",
    "Cognizant", "Accenture", "IBM India", "Capgemini", "Deloitte",
    "Amazon India", "Microsoft India", "Google India", "Flipkart", "Swiggy",
    "Razorpay", "CRED", "Zomato", "PhonePe", "Paytm",
    "Mu Sigma", "Fractal Analytics", "Tiger Analytics", "LatentView",
    "EXL Service", "Genpact", "WNS Global", "Mphasis", "Hexaware",
]

SKILLS_BY_ROLE = {
    "Data Analyst": ["SQL", "Python", "Power BI", "Excel", "Statistics", "Tableau", "R"],
    "Business Analyst": ["SQL", "Excel", "Power BI", "JIRA", "Agile", "Stakeholder Management", "Tableau"],
    "Data Scientist": ["Python", "Machine Learning", "Statistics", "SQL", "TensorFlow", "PyTorch", "Scikit-learn", "Deep Learning"],
    "Python Developer": ["Python", "Django", "FastAPI", "REST APIs", "SQL", "Docker", "Git", "AWS"],
    "ML Engineer": ["Python", "TensorFlow", "PyTorch", "MLOps", "Docker", "Kubernetes", "AWS", "CI/CD"],
    "Data Engineer": ["Python", "SQL", "Spark", "Airflow", "AWS", "Kafka", "Hadoop", "ETL"],
    "QA Engineer": ["Selenium", "Python", "JIRA", "Manual Testing", "API Testing", "SQL", "Git"],
    "Product Analyst": ["SQL", "Python", "Power BI", "Google Analytics", "A/B Testing", "Excel", "Mixpanel"],
    "BI Developer": ["Power BI", "SQL", "DAX", "Tableau", "Excel", "SSRS", "Azure"],
    "AI/ML Researcher": ["Python", "Deep Learning", "LLMs", "Generative AI", "PyTorch", "Research", "Mathematics"],
}

EXP_LEVELS = {
    "Fresher (0-1 yr)": {"mult": 0.55, "weight": 0.25},
    "Junior (1-3 yrs)": {"mult": 0.80, "weight": 0.35},
    "Mid (3-5 yrs)": {"mult": 1.10, "weight": 0.25},
    "Senior (5+ yrs)": {"mult": 1.55, "weight": 0.15},
}

EDUCATION = ["B.Tech", "MCA", "MBA", "M.Tech", "BCA", "BSc", "MSc"]

n = 3000
records = []

for _ in range(n):
    role = random.choices(list(ROLES.keys()), weights=[1]*len(ROLES))[0]
    city = random.choices(list(CITIES.keys()), weights=[v["weight"] for v in CITIES.values()])[0]
    company = random.choice(COMPANIES)
    exp = random.choices(list(EXP_LEVELS.keys()), weights=[v["weight"] for v in EXP_LEVELS.values()])[0]
    edu = random.choice(EDUCATION)
    
    base = ROLES[role]["base_salary"]
    salary = int(base * CITIES[city]["salary_mult"] * EXP_LEVELS[exp]["mult"] * np.random.uniform(0.85, 1.20))
    
    skills = random.sample(SKILLS_BY_ROLE[role], k=random.randint(3, min(5, len(SKILLS_BY_ROLE[role]))))
    
    days_ago = random.randint(0, 180)
    post_date = datetime.now() - timedelta(days=days_ago)
    quarter = f"Q{((post_date.month - 1) // 3) + 1} {post_date.year}"
    
    records.append({
        "role": role,
        "company": company,
        "city": city,
        "experience": exp,
        "education": edu,
        "salary": salary,
        "salary_lpa": round(salary / 100000, 1),
        "skills": "|".join(skills),
        "primary_skill": skills[0],
        "posted_days_ago": days_ago,
        "quarter": quarter,
        "growth_rate": ROLES[role]["growth"],
        "demand_score": ROLES[role]["demand"],
    })

df = pd.DataFrame(records)
df.to_csv("/home/claude/talentpulse/data/jobs.csv", index=False)
print(f"Generated {len(df)} records")
print(df.head())
print(df.dtypes)
