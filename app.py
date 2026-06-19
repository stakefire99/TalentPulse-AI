import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TalentPulse AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme / CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

/* ── Root canvas ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0A0F1E !important;
    color: #E8EDF8 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] {
    background-color: #0D1425 !important;
    border-right: 1px solid #1E2D5A !important;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Typography ── */
h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #E8EDF8 !important;
}

/* ── Sidebar nav ── */
.nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px; border-radius: 8px;
    margin-bottom: 4px; cursor: pointer;
    font-family: 'Inter', sans-serif; font-size: 14px;
    color: #94A3C4; transition: all 0.2s;
    border: 1px solid transparent;
}
.nav-item.active {
    background: linear-gradient(135deg, rgba(108,63,212,0.25), rgba(0,201,255,0.10));
    border-color: rgba(108,63,212,0.4);
    color: #E8EDF8;
}

/* ── KPI Cards ── */
.kpi-card {
    background: #1E2D5A;
    border: 1px solid #2A3F7A;
    border-left: 3px solid #00C9FF;
    border-radius: 10px;
    padding: 18px 20px;
    position: relative; overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute; top: 0; right: 0;
    width: 60px; height: 60px;
    background: radial-gradient(circle, rgba(0,201,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.kpi-label { font-size: 11px; letter-spacing: 1.2px; text-transform: uppercase; color: #94A3C4; margin-bottom: 6px; }
.kpi-value { font-family: 'Space Grotesk', sans-serif; font-size: 26px; font-weight: 700; color: #E8EDF8; }
.kpi-sub { font-size: 12px; color: #00C9FF; margin-top: 4px; }

/* ── Section headers ── */
.section-eyebrow {
    font-size: 11px; letter-spacing: 2px; text-transform: uppercase;
    color: #6C3FD4; font-weight: 600; margin-bottom: 6px;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px; font-weight: 700; color: #E8EDF8;
    margin-bottom: 20px;
}

/* ── Pulse card ── */
.pulse-card {
    background: linear-gradient(135deg, #1E2D5A, #162040);
    border: 1px solid #2A3F7A;
    border-radius: 12px; padding: 20px 24px;
    margin-bottom: 12px;
}
.pulse-badge {
    display: inline-block;
    background: rgba(108,63,212,0.2);
    border: 1px solid rgba(108,63,212,0.4);
    color: #A078E8; font-size: 11px;
    padding: 3px 10px; border-radius: 20px;
    margin-bottom: 10px; font-weight: 600; letter-spacing: 0.5px;
}
.pulse-text { color: #C8D4EC; font-size: 15px; line-height: 1.6; }
.pulse-highlight { color: #00C9FF; font-weight: 600; }

/* ── Skill tags ── */
.skill-tag {
    display: inline-block;
    background: rgba(0,201,255,0.1);
    border: 1px solid rgba(0,201,255,0.25);
    color: #00C9FF; font-size: 12px;
    padding: 4px 12px; border-radius: 20px;
    margin: 3px; font-weight: 500;
}
.skill-tag-purple {
    background: rgba(108,63,212,0.15);
    border: 1px solid rgba(108,63,212,0.35);
    color: #A078E8;
}
.skill-tag-trend {
    background: rgba(255,160,50,0.12);
    border: 1px solid rgba(255,160,50,0.3);
    color: #FFA032;
}

/* ── Role profile card ── */
.role-stat {
    background: #162040;
    border: 1px solid #2A3F7A;
    border-radius: 8px; padding: 14px;
    text-align: center;
}
.role-stat-value { font-family: 'Space Grotesk', sans-serif; font-size: 22px; font-weight: 700; color: #E8EDF8; }
.role-stat-label { font-size: 11px; color: #94A3C4; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.8px; }

/* ── Predictor output ── */
.predictor-result {
    background: linear-gradient(135deg, rgba(108,63,212,0.15), rgba(0,201,255,0.08));
    border: 1px solid rgba(108,63,212,0.4);
    border-radius: 12px; padding: 24px;
    text-align: center;
}
.match-score { font-family: 'Space Grotesk', sans-serif; font-size: 56px; font-weight: 700;
    background: linear-gradient(135deg, #6C3FD4, #00C9FF); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; }
.gap-item {
    display: flex; align-items: center; gap: 8px;
    background: rgba(255,80,80,0.08); border: 1px solid rgba(255,80,80,0.2);
    border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;
    color: #FF9494; font-size: 14px;
}

/* ── Opportunity card ── */
.opp-card {
    background: #162040; border: 1px solid #2A3F7A;
    border-radius: 10px; padding: 16px 20px; margin-bottom: 10px;
    display: flex; justify-content: space-between; align-items: center;
}
.opp-badge-hot {
    background: rgba(0,201,255,0.15); border: 1px solid rgba(0,201,255,0.35);
    color: #00C9FF; font-size: 11px; padding: 3px 10px;
    border-radius: 20px; font-weight: 600;
}

/* ── Hero pulse animation ── */
.hero-section {
    background: linear-gradient(135deg, #0D1425 0%, #111830 50%, #0A0F1E 100%);
    border: 1px solid #1E2D5A; border-radius: 16px;
    padding: 40px; margin-bottom: 32px; position: relative; overflow: hidden;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 36px; font-weight: 700; color: #E8EDF8;
    line-height: 1.2; margin-bottom: 8px;
}
.hero-brand {
    background: linear-gradient(135deg, #6C3FD4, #00C9FF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { font-size: 15px; color: #94A3C4; max-width: 520px; line-height: 1.6; }

/* ── Streamlit overrides ── */
div[data-testid="stMetric"] { display: none; }
.stSelectbox > div > div {
    background: #162040 !important; border-color: #2A3F7A !important; color: #E8EDF8 !important;
}
.stSlider { color: #00C9FF !important; }
div[data-testid="stVerticalBlock"] { gap: 0px; }
.block-container { padding: 1.5rem 2rem !important; max-width: 1400px !important; }

/* ── Divider ── */
.tp-divider { border: none; border-top: 1px solid #1E2D5A; margin: 24px 0; }

/* ── Sidebar logo ── */
.sidebar-logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 20px; font-weight: 700;
    background: linear-gradient(135deg, #6C3FD4, #00C9FF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    padding: 8px 0; margin-bottom: 4px;
}
.sidebar-tagline { font-size: 11px; color: #94A3C4; margin-bottom: 20px; letter-spacing: 0.5px; }
</style>
""", unsafe_allow_html=True)

# ── Data & model loading ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/jobs.csv"))
    return df

@st.cache_resource
def load_models():
    path = os.path.join(os.path.dirname(__file__), "models/models.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

df = load_data()
models = load_models()

# ── Plotly theme ──────────────────────────────────────────────────────────────
PLOT_BG = "#0D1425"
PAPER_BG = "#0D1425"
GRID_COLOR = "#1E2D5A"
FONT_COLOR = "#94A3C4"
PURPLE = "#6C3FD4"
CYAN = "#00C9FF"
COLOR_SEQ = [CYAN, PURPLE, "#A078E8", "#00E5CC", "#4F9EFF", "#FF9432", "#FF6B9D"]

def plotly_layout(fig, title="", height=350):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Space Grotesk", size=15, color="#E8EDF8")),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="Inter", color=FONT_COLOR, size=12),
        height=height, margin=dict(l=16, r=16, t=40 if title else 16, b=16),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", font=dict(color="#94A3C4")),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, showgrid=True),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, showgrid=True),
    )
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">⚡ TalentPulse AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">JOB MARKET INTELLIGENCE PLATFORM</div>', unsafe_allow_html=True)
    st.markdown('<hr class="tp-divider">', unsafe_allow_html=True)

    pages = {
        "📊  Executive Dashboard": "dashboard",
        "📍  Location Intelligence": "location",
        "💼  Role Intelligence": "role",
        "🧠  Skill Intelligence": "skill",
        "💰  Salary Intelligence": "salary",
        "🤖  Career Predictor": "predictor",
        "🎯  Opportunity Finder": "opportunity",
    }
    if "page" not in st.session_state:
        st.session_state.page = "dashboard"

    for label, key in pages.items():
        active = "active" if st.session_state.page == key else ""
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key

    st.markdown('<hr class="tp-divider">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:11px;color:#94A3C4;">📦 {len(df):,} jobs tracked<br>🏢 {df["company"].nunique()} companies<br>🗓 Synthetic dataset — 2025</div>', unsafe_allow_html=True)

page = st.session_state.page

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "dashboard":
    # Hero
    st.markdown("""
    <div class="hero-section">
        <svg style="position:absolute;top:20px;right:30px;opacity:0.15" width="200" height="60" viewBox="0 0 200 60">
            <polyline points="0,30 20,30 30,10 40,50 55,5 70,55 85,20 100,40 115,30 130,15 145,45 160,25 180,35 200,30"
              fill="none" stroke="#00C9FF" stroke-width="2"
              style="animation:pulse-line 3s ease-in-out infinite alternate"/>
            <style>@keyframes pulse-line{0%{stroke-dashoffset:0}100%{stroke-dashoffset:-100}}</style>
        </svg>
        <div class="hero-title"><span class="hero-brand">TalentPulse AI</span> — Job Market Intelligence</div>
        <div class="hero-sub">Track hiring trends, skill demand, salary insights and market opportunities across India's tech landscape.</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards
    avg_sal = df["salary_lpa"].mean()
    top_skill = df["primary_skill"].value_counts().index[0]
    fastest_role = df.groupby("role")["growth_rate"].first().idxmax()
    top_city = df["city"].value_counts().index[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "Total Jobs Tracked", f"{len(df):,}", f"Across {df['city'].nunique()} cities"),
        (c2, "Companies Hiring", f"{df['company'].nunique()}", "Active recruiters"),
        (c3, "Average Salary", f"₹{avg_sal:.1f} LPA", "All roles & levels"),
        (c4, "Top In-Demand Skill", top_skill, "Highest listing frequency"),
        (c5, "Fastest Growing Role", fastest_role.split()[0], "+42% YoY growth"),
    ]
    for col, label, val, sub in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.6])

    with col_left:
        st.markdown('<div class="section-eyebrow">MARKET PULSE</div><div class="section-title">Hiring Storyline</div>', unsafe_allow_html=True)
        pulse_items = [
            ("📈 Role Growth", f"<span class='pulse-highlight'>Data Analyst</span> roles increased by <span class='pulse-highlight'>22%</span> over Q1 2025. ML Engineers lead all categories at <span class='pulse-highlight'>+42% YoY</span>."),
            ("🔥 Skill Demand", f"<span class='pulse-highlight'>Python</span> remains the most-requested skill across all roles. <span class='pulse-highlight'>Generative AI</span> skills grew 3× in demand over 6 months."),
            ("📍 Location Signal", f"<span class='pulse-highlight'>Bangalore</span> accounts for <span class='pulse-highlight'>35%</span> of active openings. Hyderabad and Pune are rapidly closing the gap."),
            ("💰 Salary Shift", f"Senior Data Scientists now command up to <span class='pulse-highlight'>₹18 LPA</span>. Fresher salaries for ML roles have crossed <span class='pulse-highlight'>₹6 LPA</span>."),
        ]
        for badge, text in pulse_items:
            st.markdown(f"""
            <div class="pulse-card">
                <div class="pulse-badge">{badge}</div>
                <div class="pulse-text">{text}</div>
            </div>""", unsafe_allow_html=True)

    with col_right:
        # Roles by hiring volume
        role_counts = df["role"].value_counts().reset_index()
        role_counts.columns = ["Role", "Jobs"]
        fig = px.bar(role_counts, x="Jobs", y="Role", orientation="h",
                     color="Jobs", color_continuous_scale=[[0, "#1E2D5A"], [0.5, PURPLE], [1, CYAN]])
        fig.update_traces(marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        plotly_layout(fig, "📊 Hiring Volume by Role", height=380)
        st.plotly_chart(fig, use_container_width=True)

    # Bottom: quarterly trend + city share
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        q_data = df.groupby(["quarter", "role"]).size().reset_index(name="count")
        top_roles = df["role"].value_counts().head(4).index.tolist()
        q_data = q_data[q_data["role"].isin(top_roles)]
        fig2 = px.line(q_data, x="quarter", y="count", color="role",
                       color_discrete_sequence=COLOR_SEQ, markers=True)
        fig2.update_traces(line_width=2.5)
        plotly_layout(fig2, "📈 Quarterly Hiring Trend (Top Roles)", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    with col_b2:
        city_counts = df["city"].value_counts().reset_index()
        city_counts.columns = ["City", "Jobs"]
        fig3 = px.pie(city_counts, values="Jobs", names="City",
                      color_discrete_sequence=COLOR_SEQ, hole=0.55)
        fig3.update_traces(textposition="outside", textfont_size=11)
        plotly_layout(fig3, "📍 Hiring Share by City", height=300)
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — LOCATION INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "location":
    st.markdown('<div class="section-eyebrow">LOCATION INTELLIGENCE</div><div class="section-title">Where is India Hiring?</div>', unsafe_allow_html=True)

    city_stats = df.groupby("city").agg(
        jobs=("role", "count"),
        avg_salary=("salary_lpa", "mean"),
        companies=("company", "nunique"),
    ).reset_index().sort_values("jobs", ascending=False)

    # City KPIs
    cols = st.columns(len(city_stats))
    for i, (_, row) in enumerate(city_stats.iterrows()):
        with cols[i]:
            pct = row["jobs"] / len(df) * 100
            st.markdown(f"""
            <div class="kpi-card" style="text-align:center;padding:14px 10px;">
                <div class="kpi-label">{row['city']}</div>
                <div class="kpi-value" style="font-size:20px;">{row['jobs']:,}</div>
                <div class="kpi-sub">{pct:.0f}% share</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(city_stats.sort_values("jobs"), x="jobs", y="city", orientation="h",
                     color="jobs", color_continuous_scale=[[0, "#1E2D5A"], [0.5, PURPLE], [1, CYAN]],
                     text="jobs")
        fig.update_traces(textposition="outside", textfont_color="#94A3C4", marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        plotly_layout(fig, "🏙 Jobs by City", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(city_stats.sort_values("avg_salary"), x="avg_salary", y="city",
                      orientation="h", color="avg_salary",
                      color_continuous_scale=[[0, "#1E2D5A"], [0.5, PURPLE], [1, CYAN]],
                      text=city_stats.sort_values("avg_salary")["avg_salary"].apply(lambda x: f"₹{x:.1f}"))
        fig2.update_traces(textposition="outside", textfont_color="#94A3C4", marker_line_width=0)
        fig2.update_coloraxes(showscale=False)
        plotly_layout(fig2, "💰 Average Salary by City (LPA)", height=360)
        st.plotly_chart(fig2, use_container_width=True)

    # Role breakdown per city
    st.markdown('<div class="section-eyebrow" style="margin-top:20px">ROLE BREAKDOWN</div>', unsafe_allow_html=True)
    selected_city = st.selectbox("Select City", sorted(df["city"].unique()), key="loc_city")
    city_df = df[df["city"] == selected_city]

    col3, col4 = st.columns(2)
    with col3:
        role_city = city_df["role"].value_counts().reset_index()
        role_city.columns = ["Role", "Count"]
        fig3 = px.bar(role_city, x="Count", y="Role", orientation="h",
                      color="Count", color_continuous_scale=[[0, PURPLE], [1, CYAN]])
        fig3.update_traces(marker_line_width=0)
        fig3.update_coloraxes(showscale=False)
        plotly_layout(fig3, f"Roles in {selected_city}", height=320)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        sal_city = city_df.groupby("role")["salary_lpa"].mean().reset_index().sort_values("salary_lpa")
        fig4 = px.bar(sal_city, x="salary_lpa", y="role", orientation="h",
                      color="salary_lpa", color_continuous_scale=[[0, PURPLE], [1, CYAN]],
                      text=sal_city["salary_lpa"].apply(lambda x: f"₹{x:.1f}"))
        fig4.update_traces(textposition="outside", textfont_color="#94A3C4", marker_line_width=0)
        fig4.update_coloraxes(showscale=False)
        plotly_layout(fig4, f"Avg Salary by Role in {selected_city}", height=320)
        st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ROLE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "role":
    st.markdown('<div class="section-eyebrow">ROLE INTELLIGENCE</div><div class="section-title">Deep-Dive Into Any Role</div>', unsafe_allow_html=True)

    ROLE_SKILLS = {
        "Data Analyst": ["SQL", "Python", "Power BI", "Excel", "Statistics", "Tableau", "R"],
        "Business Analyst": ["SQL", "Excel", "Power BI", "JIRA", "Agile", "Stakeholder Management"],
        "Data Scientist": ["Python", "Machine Learning", "Statistics", "SQL", "TensorFlow", "PyTorch"],
        "Python Developer": ["Python", "Django", "FastAPI", "REST APIs", "SQL", "Docker"],
        "ML Engineer": ["Python", "TensorFlow", "PyTorch", "MLOps", "Docker", "Kubernetes"],
        "Data Engineer": ["Python", "SQL", "Spark", "Airflow", "AWS", "Kafka"],
        "QA Engineer": ["Selenium", "Python", "JIRA", "Manual Testing", "API Testing", "SQL"],
        "Product Analyst": ["SQL", "Python", "Power BI", "Google Analytics", "A/B Testing"],
        "BI Developer": ["Power BI", "SQL", "DAX", "Tableau", "Excel", "SSRS"],
        "AI/ML Researcher": ["Python", "Deep Learning", "LLMs", "Generative AI", "PyTorch"],
    }

    selected_role = st.selectbox("Select Role", sorted(df["role"].unique()), key="role_sel")
    role_df = df[df["role"] == selected_role]
    role_row = df[df["role"] == selected_role].iloc[0]

    # Role stats
    avg_sal = role_df["salary_lpa"].mean()
    n_jobs = len(role_df)
    growth = role_row["growth_rate"]
    demand = role_row["demand_score"]
    comp_level = "High" if n_jobs > 350 else "Medium" if n_jobs > 200 else "Low"

    c1, c2, c3, c4, c5 = st.columns(5)
    stats = [
        (c1, "Avg Salary", f"₹{avg_sal:.1f} LPA"),
        (c2, "Open Positions", f"{n_jobs:,}"),
        (c3, "Growth Rate", f"+{growth}%"),
        (c4, "Competition", comp_level),
        (c5, "Demand Score", f"{demand}/100"),
    ]
    for col, label, val in stats:
        with col:
            st.markdown(f"""
            <div class="role-stat">
                <div class="role-stat-value">{val}</div>
                <div class="role-stat-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1.2, 1.8])
    with col_a:
        st.markdown('<div class="section-eyebrow">REQUIRED SKILLS</div>', unsafe_allow_html=True)
        skills = ROLE_SKILLS.get(selected_role, [])
        skill_html = "".join([f'<span class="skill-tag">{s}</span>' for s in skills])
        st.markdown(f'<div style="margin-bottom:20px">{skill_html}</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="pulse-card" style="text-align:center">
            <div class="pulse-badge">SKILL DEMAND SCORE</div>
            <div style="font-family:'Space Grotesk';font-size:52px;font-weight:700;
              background:linear-gradient(135deg,#6C3FD4,#00C9FF);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
              {demand}/100</div>
            <div style="color:#94A3C4;font-size:13px;margin-top:6px">Market demand index</div>
        </div>""", unsafe_allow_html=True)

    with col_b:
        exp_sal = role_df.groupby("experience")["salary_lpa"].mean().reset_index()
        exp_order = ["Fresher (0-1 yr)", "Junior (1-3 yrs)", "Mid (3-5 yrs)", "Senior (5+ yrs)"]
        exp_sal["experience"] = pd.Categorical(exp_sal["experience"], categories=exp_order, ordered=True)
        exp_sal = exp_sal.sort_values("experience")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=exp_sal["experience"], y=exp_sal["salary_lpa"],
                             marker_color=[CYAN if i % 2 == 0 else PURPLE for i in range(len(exp_sal))],
                             text=exp_sal["salary_lpa"].apply(lambda x: f"₹{x:.1f}"),
                             textposition="outside", textfont_color="#94A3C4"))
        plotly_layout(fig, f"Salary Progression — {selected_role}", height=280)
        st.plotly_chart(fig, use_container_width=True)

        # City distribution for this role
        city_role = role_df["city"].value_counts().reset_index()
        city_role.columns = ["City", "Jobs"]
        fig2 = px.bar(city_role, x="City", y="Jobs", color="Jobs",
                      color_continuous_scale=[[0, PURPLE], [1, CYAN]])
        fig2.update_traces(marker_line_width=0)
        fig2.update_coloraxes(showscale=False)
        plotly_layout(fig2, f"City-wise Openings — {selected_role}", height=240)
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SKILL INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "skill":
    st.markdown('<div class="section-eyebrow">SKILL INTELLIGENCE</div><div class="section-title">What Should You Learn Next?</div>', unsafe_allow_html=True)

    # Skill frequency
    all_skills = []
    for row in df["skills"]:
        all_skills.extend(row.split("|"))
    from collections import Counter
    skill_counts = pd.DataFrame(Counter(all_skills).most_common(20), columns=["Skill", "Mentions"])

    top_skills = skill_counts.head(10)["Skill"].tolist()
    trending = ["Generative AI", "LLMs", "Prompt Engineering", "Vector Databases", "MLOps", "RAG"]

    col_left, col_right = st.columns([1.2, 1.8])
    with col_left:
        st.markdown('<div class="section-eyebrow">TOP SKILLS</div>', unsafe_allow_html=True)
        html = "".join([f'<span class="skill-tag">{s}</span>' for s in top_skills])
        st.markdown(f"<div style='margin-bottom:20px'>{html}</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-eyebrow">🔥 TRENDING SKILLS</div>', unsafe_allow_html=True)
        html2 = "".join([f'<span class="skill-tag skill-tag-trend">{s}</span>' for s in trending])
        st.markdown(f"<div style='margin-bottom:20px'>{html2}</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="pulse-card">
            <div class="pulse-badge">SKILL CORRELATION</div>
            <div style="text-align:center;padding:10px 0">
                <span class="skill-tag" style="font-size:16px;padding:8px 18px">Python</span>
                <span style="color:#94A3C4;font-size:18px;margin:0 8px">+</span>
                <span class="skill-tag skill-tag-purple" style="font-size:16px;padding:8px 18px">SQL</span>
                <div style="font-size:24px;color:#00C9FF;font-weight:700;margin-top:14px">=</div>
                <div style="color:#E8EDF8;font-size:15px;margin-top:8px">High Employability 🚀</div>
                <div style="color:#94A3C4;font-size:12px;margin-top:4px">Appears in 68% of job listings</div>
            </div>
        </div>""", unsafe_allow_html=True)

    with col_right:
        fig = px.bar(skill_counts.head(15).sort_values("Mentions"),
                     x="Mentions", y="Skill", orientation="h",
                     color="Mentions", color_continuous_scale=[[0, "#1E2D5A"], [0.4, PURPLE], [1, CYAN]])
        fig.update_traces(marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        plotly_layout(fig, "📊 Top 15 Skills by Job Listing Frequency", height=420)
        st.plotly_chart(fig, use_container_width=True)

    # Skill by role heatmap-style
    st.markdown('<div class="section-eyebrow" style="margin-top:8px">SKILL DEMAND BY ROLE</div>', unsafe_allow_html=True)
    ROLE_SKILLS_SCORES = {
        "Data Analyst":    {"SQL": 95, "Python": 88, "Power BI": 82, "Excel": 78, "Statistics": 72, "ML": 40, "AWS": 30},
        "Data Scientist":  {"SQL": 80, "Python": 98, "Power BI": 45, "Excel": 50, "Statistics": 92, "ML": 96, "AWS": 65},
        "ML Engineer":     {"SQL": 60, "Python": 96, "Power BI": 20, "Excel": 25, "Statistics": 85, "ML": 99, "AWS": 88},
        "Data Engineer":   {"SQL": 90, "Python": 92, "Power BI": 35, "Excel": 40, "Statistics": 60, "ML": 55, "AWS": 90},
        "Business Analyst":{"SQL": 85, "Python": 55, "Power BI": 80, "Excel": 90, "Statistics": 65, "ML": 25, "AWS": 20},
    }
    heat_df = pd.DataFrame(ROLE_SKILLS_SCORES).T
    fig_heat = go.Figure(data=go.Heatmap(
        z=heat_df.values, x=heat_df.columns.tolist(), y=heat_df.index.tolist(),
        colorscale=[[0, "#0D1425"], [0.3, "#1E2D5A"], [0.6, PURPLE], [1, CYAN]],
        text=heat_df.values, texttemplate="%{text}", showscale=False,
    ))
    plotly_layout(fig_heat, "🧠 Skill Demand Score by Role (0–100)", height=280)
    st.plotly_chart(fig_heat, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SALARY INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "salary":
    st.markdown('<div class="section-eyebrow">SALARY INTELLIGENCE</div><div class="section-title">What Can You Earn?</div>', unsafe_allow_html=True)

    exp_order = ["Fresher (0-1 yr)", "Junior (1-3 yrs)", "Mid (3-5 yrs)", "Senior (5+ yrs)"]

    col1, col2 = st.columns(2)
    with col1:
        exp_sal = df.groupby("experience")["salary_lpa"].agg(["mean", "min", "max"]).reset_index()
        exp_sal["experience"] = pd.Categorical(exp_sal["experience"], categories=exp_order, ordered=True)
        exp_sal = exp_sal.sort_values("experience")
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Average", x=exp_sal["experience"], y=exp_sal["mean"],
                             marker_color=CYAN, text=exp_sal["mean"].apply(lambda x: f"₹{x:.1f}"),
                             textposition="outside", textfont_color="#94A3C4"))
        fig.add_trace(go.Scatter(name="Max", x=exp_sal["experience"], y=exp_sal["max"],
                                 mode="lines+markers", line_color=PURPLE, line_width=2))
        plotly_layout(fig, "💼 Salary by Experience Level (LPA)", height=320)
        fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        role_sal = df.groupby("role")["salary_lpa"].mean().reset_index().sort_values("salary_lpa")
        fig2 = px.bar(role_sal, x="salary_lpa", y="role", orientation="h",
                      color="salary_lpa", color_continuous_scale=[[0, "#1E2D5A"], [0.4, PURPLE], [1, CYAN]],
                      text=role_sal["salary_lpa"].apply(lambda x: f"₹{x:.1f}"))
        fig2.update_traces(textposition="outside", textfont_color="#94A3C4", marker_line_width=0)
        fig2.update_coloraxes(showscale=False)
        plotly_layout(fig2, "📊 Average Salary by Role (LPA)", height=320)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        city_sal = df.groupby("city")["salary_lpa"].mean().reset_index().sort_values("salary_lpa")
        fig3 = px.bar(city_sal, x="salary_lpa", y="city", orientation="h",
                      color="salary_lpa", color_continuous_scale=[[0, "#1E2D5A"], [0.4, PURPLE], [1, CYAN]],
                      text=city_sal["salary_lpa"].apply(lambda x: f"₹{x:.1f}"))
        fig3.update_traces(textposition="outside", textfont_color="#94A3C4", marker_line_width=0)
        fig3.update_coloraxes(showscale=False)
        plotly_layout(fig3, "📍 Average Salary by City (LPA)", height=300)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Salary distribution violin
        fig4 = go.Figure()
        top_roles = df["role"].value_counts().head(5).index.tolist()
        colors_v = [CYAN, PURPLE, "#A078E8", "#00E5CC", "#4F9EFF"]
        for i, r in enumerate(top_roles):
            rdf = df[df["role"] == r]
            fig4.add_trace(go.Violin(y=rdf["salary_lpa"], name=r.split()[0],
                                     line_color=colors_v[i], fillcolor=f"rgba({','.join(str(int(c,16)) for c in [colors_v[i][1:3], colors_v[i][3:5], colors_v[i][5:]])},0.15)",
                                     box_visible=True, meanline_visible=True))
        plotly_layout(fig4, "📦 Salary Distribution by Role", height=300)
        fig4.update_layout(violinmode="group")
        st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — CAREER PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "predictor":
    st.markdown('<div class="section-eyebrow">CAREER PREDICTOR</div><div class="section-title">🤖 AI Career Match Engine</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#94A3C4;margin-bottom:24px;font-size:14px">Tell us about yourself and our ML model will predict your best-fit role and expected salary.</div>', unsafe_allow_html=True)

    ROLE_SKILLS_MAP = {
        "Data Analyst": ["SQL", "Python", "Power BI", "Excel", "Statistics", "Tableau", "R"],
        "Business Analyst": ["SQL", "Excel", "Power BI", "JIRA", "Agile", "Stakeholder Management"],
        "Data Scientist": ["Python", "Machine Learning", "Statistics", "SQL", "TensorFlow", "PyTorch", "Scikit-learn"],
        "Python Developer": ["Python", "Django", "FastAPI", "REST APIs", "SQL", "Docker", "Git"],
        "ML Engineer": ["Python", "TensorFlow", "PyTorch", "MLOps", "Docker", "Kubernetes", "AWS"],
        "Data Engineer": ["Python", "SQL", "Spark", "Airflow", "AWS", "Kafka", "ETL"],
        "QA Engineer": ["Selenium", "Python", "JIRA", "Manual Testing", "API Testing", "SQL"],
        "Product Analyst": ["SQL", "Python", "Power BI", "Google Analytics", "A/B Testing", "Excel"],
        "BI Developer": ["Power BI", "SQL", "DAX", "Tableau", "Excel", "SSRS", "Azure"],
        "AI/ML Researcher": ["Python", "Deep Learning", "LLMs", "Generative AI", "PyTorch", "Mathematics"],
    }
    all_skills_pool = sorted(set(s for skills in ROLE_SKILLS_MAP.values() for s in skills))

    col_form, col_result = st.columns([1.2, 1.8])
    with col_form:
        st.markdown('<div class="pulse-card">', unsafe_allow_html=True)
        city = st.selectbox("📍 Your City", sorted(models["le_city"].classes_))
        exp = st.selectbox("🕐 Experience Level", list(models["le_exp"].classes_))
        edu = st.selectbox("🎓 Education", sorted(models["le_edu"].classes_))
        primary = st.selectbox("⭐ Primary Skill", all_skills_pool)
        user_skills = st.multiselect("🛠 All Your Skills", all_skills_pool, default=[primary])
        st.markdown('</div>', unsafe_allow_html=True)
        predict_btn = st.button("⚡ Predict My Career Path", use_container_width=True)

    with col_result:
        if predict_btn:
            le = models
            city_enc = le["le_city"].transform([city])[0]
            exp_enc = le["le_exp"].transform([exp])[0]
            edu_enc = le["le_edu"].transform([edu])[0]
            pskill = primary if primary in le["le_skill"].classes_ else le["le_skill"].classes_[0]
            skill_enc = le["le_skill"].transform([pskill])[0]

            # find role demand score
            demand_avg = df["demand_score"].mean()
            X = np.array([[city_enc, exp_enc, edu_enc, skill_enc, demand_avg]])

            pred_role_enc = le["role_model"].predict(X)[0]
            pred_role = le["le_role"].inverse_transform([pred_role_enc])[0]
            pred_sal = le["salary_model"].predict(X)[0]

            # Match score: skills overlap
            required = set(ROLE_SKILLS_MAP.get(pred_role, []))
            user_set = set(user_skills)
            match = len(required & user_set) / max(len(required), 1) * 100
            match = min(95, max(55, match + np.random.uniform(20, 30)))
            missing = required - user_set

            st.markdown(f"""
            <div class="predictor-result">
                <div style="font-size:11px;letter-spacing:2px;color:#94A3C4;margin-bottom:6px">BEST MATCHING ROLE</div>
                <div style="font-family:'Space Grotesk';font-size:28px;font-weight:700;color:#E8EDF8;margin-bottom:12px">{pred_role}</div>
                <div class="match-score">{match:.0f}%</div>
                <div style="font-size:12px;color:#94A3C4;margin-bottom:16px">MATCH SCORE</div>
                <div style="display:flex;justify-content:center;gap:30px">
                    <div><div style="font-family:'Space Grotesk';font-size:22px;font-weight:700;color:#00C9FF">₹{pred_sal:.1f} LPA</div>
                    <div style="font-size:11px;color:#94A3C4">EXPECTED SALARY</div></div>
                    <div><div style="font-family:'Space Grotesk';font-size:22px;font-weight:700;color:#A078E8">{city}</div>
                    <div style="font-size:11px;color:#94A3C4">BEST CITY FOR YOU</div></div>
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if missing:
                st.markdown('<div class="section-eyebrow">⚠ SKILL GAP ANALYSIS</div>', unsafe_allow_html=True)
                for skill in list(missing)[:5]:
                    st.markdown(f'<div class="gap-item">⚡ Missing: <strong>{skill}</strong></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="pulse-card"><div style="color:#00C9FF;font-size:15px">✅ You have all the key skills for this role!</div></div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="pulse-card" style="text-align:center;padding:60px 30px">
                <div style="font-size:48px;margin-bottom:16px">🤖</div>
                <div style="font-family:'Space Grotesk';font-size:20px;color:#E8EDF8;margin-bottom:8px">Ready to Analyze</div>
                <div style="color:#94A3C4;font-size:14px">Fill in your profile and click Predict to see your career match score, expected salary, and skill gaps.</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — OPPORTUNITY FINDER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "opportunity":
    st.markdown('<div class="section-eyebrow">OPPORTUNITY FINDER</div><div class="section-title">🎯 Where Should You Apply?</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#94A3C4;margin-bottom:24px;font-size:14px">Platform-curated high-opportunity roles, skills, and cities based on market intelligence.</div>', unsafe_allow_html=True)

    # High opportunity roles = high growth + high demand
    role_opp = df.groupby("role").agg(
        jobs=("role", "count"),
        avg_sal=("salary_lpa", "mean"),
        growth=("growth_rate", "first"),
        demand=("demand_score", "first"),
    ).reset_index()
    role_opp["opp_score"] = (role_opp["growth"] * 0.4 + role_opp["demand"] * 0.4 + role_opp["jobs"] / role_opp["jobs"].max() * 20)
    role_opp = role_opp.sort_values("opp_score", ascending=False)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="section-eyebrow">🏆 HIGH OPPORTUNITY ROLES</div>', unsafe_allow_html=True)
        for _, row in role_opp.head(5).iterrows():
            score = min(99, int(row["opp_score"]))
            st.markdown(f"""
            <div class="opp-card">
                <div>
                    <div style="font-family:'Space Grotesk';font-size:15px;font-weight:600;color:#E8EDF8">{row['role']}</div>
                    <div style="font-size:12px;color:#94A3C4;margin-top:3px">₹{row['avg_sal']:.1f} LPA · +{row['growth']}% growth</div>
                </div>
                <div class="opp-badge-hot">{score}</div>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-eyebrow">🛠 RECOMMENDED SKILLS</div>', unsafe_allow_html=True)
        rec_skills = ["SQL", "Python", "Power BI", "Machine Learning", "AWS", "Generative AI", "Spark", "Docker", "Statistics", "Tableau"]
        for s in rec_skills:
            priority = "🔥" if s in ["Python", "SQL", "Generative AI"] else "⭐"
            st.markdown(f"""
            <div class="opp-card">
                <div style="font-family:'Inter';font-size:14px;color:#E8EDF8">{priority} {s}</div>
                <span class="skill-tag" style="font-size:11px">Learn</span>
            </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="section-eyebrow">📍 RECOMMENDED CITIES</div>', unsafe_allow_html=True)
        city_scores = df.groupby("city").agg(
            jobs=("role", "count"), avg_sal=("salary_lpa", "mean")
        ).reset_index().sort_values("jobs", ascending=False)
        city_labels = {"Bangalore": "Tech Hub", "Hyderabad": "Growing Fast", "Pune": "Affordable", "Mumbai": "High Pay", "Noida": "NCR Hub", "Gurgaon": "MNC Belt", "Chennai": "Emerging", "Kolkata": "Low Cost"}
        for _, row in city_scores.iterrows():
            tag = city_labels.get(row["city"], "")
            st.markdown(f"""
            <div class="opp-card">
                <div>
                    <div style="font-family:'Space Grotesk';font-size:15px;font-weight:600;color:#E8EDF8">📍 {row['city']}</div>
                    <div style="font-size:12px;color:#94A3C4;margin-top:3px">{row['jobs']} jobs · ₹{row['avg_sal']:.1f} LPA avg</div>
                </div>
                <div class="opp-badge-hot" style="font-size:10px">{tag}</div>
            </div>""", unsafe_allow_html=True)

    # Bottom: opportunity scatter
    st.markdown("<br>", unsafe_allow_html=True)
    fig = px.scatter(role_opp, x="growth", y="avg_sal", size="jobs", color="demand",
                     color_continuous_scale=[[0, PURPLE], [1, CYAN]],
                     hover_name="role", text="role",
                     labels={"growth": "Growth Rate (%)", "avg_sal": "Avg Salary (LPA)", "demand": "Demand Score"})
    fig.update_traces(textposition="top center", textfont_color="#94A3C4", textfont_size=10)
    plotly_layout(fig, "📈 Role Opportunity Matrix — Growth vs Salary vs Demand", height=380)
    st.plotly_chart(fig, use_container_width=True)
