import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import requests
import joblib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Configuration ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ROI Intelligence Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Design System ──────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">

<style>
/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg-base:       #080C14;
  --bg-surface:    #0D1220;
  --bg-elevated:   #111827;
  --bg-card:       #131B2E;
  --border-subtle: rgba(255,255,255,0.06);
  --border-muted:  rgba(255,255,255,0.10);
  --border-accent: rgba(37,99,235,0.40);

  --text-primary:  #F8FAFC;
  --text-secondary:#94A3B8;
  --text-muted:    #475569;
  --text-accent:   #60A5FA;

  --accent-blue:   #2563EB;
  --accent-blue-h: #1D4ED8;
  --accent-emerald:#059669;
  --accent-amber:  #D97706;
  --accent-rose:   #E11D48;

  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 16px;
  --radius-xl: 24px;

  --shadow-card: 0 1px 3px rgba(0,0,0,0.4), 0 4px 16px rgba(0,0,0,0.3);
  --shadow-elevated: 0 8px 32px rgba(0,0,0,0.5);

  --font-display: 'Syne', sans-serif;
  --font-body:    'DM Sans', sans-serif;
  --font-mono:    'DM Mono', monospace;
}

/* ── App Shell ── */
.stApp {
  background: var(--bg-base) !important;
  font-family: var(--font-body) !important;
  color: var(--text-primary) !important;
}
.stApp > header { display: none !important; }
#MainMenu, footer, .stDeployButton { display: none !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { display: none !important; }
.stApp [data-testid="collapsedControl"] { display: none !important; }

/* ── Main container ── */
.main .block-container {
  padding: 0 3rem !important;
  max-width: 100% !important;
}

/* ── Tabs ── */
.stTabs { background: transparent !important; }
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg-surface) !important;
  border-bottom: 1px solid var(--border-subtle) !important;
  padding: 0 48px !important;
  gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text-muted) !important;
  font-family: var(--font-body) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
  padding: 18px 28px !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  transition: all 0.2s ease !important;
  margin-bottom: -1px !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--text-secondary) !important;
  background: rgba(255,255,255,0.03) !important;
}
.stTabs [aria-selected="true"] {
  color: var(--text-primary) !important;
  border-bottom-color: var(--accent-blue) !important;
  background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
  background: transparent !important;
  padding: 0 !important;
}
/* Assure que les éléments natifs Streamlit ont aussi la marge */
.stTabs [data-baseweb="tab-panel"] > div > div {
  padding-left: 0 !important;
}

/* ── Sliders ── */
.stSlider > div > div > div { background: var(--accent-blue) !important; }
.stSlider [data-baseweb="slider"] { padding: 0 !important; }
.stSlider p { color: var(--text-secondary) !important; font-family: var(--font-body) !important; font-size: 13px !important; }

/* ── Selects & inputs ── */
.stSelectbox label, .stNumberInput label {
  color: var(--text-secondary) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  font-family: var(--font-body) !important;
}
.stSelectbox [data-baseweb="select"] > div {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border-muted) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-body) !important;
}

/* ── Buttons ── */
.stButton > button {
  background: var(--accent-blue) !important;
  color: white !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-family: var(--font-body) !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  letter-spacing: 0.04em !important;
  padding: 10px 24px !important;
  transition: all 0.2s ease !important;
}
.stButton > button:hover {
  background: var(--accent-blue-h) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 16px rgba(37,99,235,0.35) !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: var(--radius-md) !important; overflow: hidden !important; }
.stDataFrame iframe { background: var(--bg-card) !important; }

/* ── Code ── */
.stCode { background: var(--bg-elevated) !important; border-radius: var(--radius-sm) !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-secondary) !important;
  font-family: var(--font-body) !important;
  font-size: 13px !important;
}

/* ── Custom Components ── */

/* Header */
.platform-header {
  background: var(--bg-surface);
  border-bottom: 1px solid var(--border-subtle);
  padding: 20px 48px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: 100;
}
.platform-logo {
  font-family: var(--font-display);
  font-size: 18px;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.02em;
}
.platform-logo span {
  color: var(--accent-blue);
}
.platform-status {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--text-muted);
  font-family: var(--font-body);
}
.status-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--accent-emerald);
  box-shadow: 0 0 8px var(--accent-emerald);
  animation: pulse 2s infinite;
}
.status-dot.offline { background: var(--accent-rose); box-shadow: 0 0 8px var(--accent-rose); animation: none; }
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* Page padding */
.page-content { padding: 40px 48px; }

/* Section header */
.section-header { margin-bottom: 32px; }
.section-label {
  font-family: var(--font-body);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent-blue);
  margin-bottom: 8px;
}
.section-title {
  font-family: var(--font-display);
  font-size: 28px;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.02em;
  line-height: 1.2;
}
.section-subtitle {
  font-family: var(--font-body);
  font-size: 15px;
  font-weight: 300;
  color: var(--text-secondary);
  margin-top: 8px;
  line-height: 1.6;
}

/* KPI Grid */
.kpi-grid { display: grid; gap: 16px; margin-bottom: 40px; }
.kpi-card {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 24px 28px;
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(37,99,235,0.5), transparent);
}
.kpi-card:hover {
  border-color: var(--border-accent);
  box-shadow: 0 4px 24px rgba(37,99,235,0.1);
}
.kpi-label {
  font-family: var(--font-body);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: 12px;
}
.kpi-value {
  font-family: var(--font-display);
  font-size: 32px;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.03em;
  line-height: 1;
}
.kpi-value.accent  { color: var(--accent-blue); }
.kpi-value.emerald { color: #34D399; }
.kpi-value.amber   { color: #FCD34D; }
.kpi-sub {
  font-family: var(--font-body);
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 8px;
  font-weight: 400;
}
.kpi-badge {
  position: absolute;
  top: 20px; right: 20px;
  background: rgba(5,150,105,0.15);
  border: 1px solid rgba(5,150,105,0.3);
  border-radius: 20px;
  padding: 3px 10px;
  font-size: 11px;
  color: #34D399;
  font-family: var(--font-mono);
  font-weight: 500;
}
.kpi-badge.blue {
  background: rgba(37,99,235,0.15);
  border-color: rgba(37,99,235,0.3);
  color: #93C5FD;
}

/* Divider */
.divider {
  height: 1px;
  background: var(--border-subtle);
  margin: 40px 0;
}

/* Chart card */
.chart-card {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 28px;
  margin-bottom: 20px;
}
.chart-title {
  font-family: var(--font-display);
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 4px;
}
.chart-subtitle {
  font-family: var(--font-body);
  font-size: 12px;
  color: var(--text-muted);
  margin-bottom: 20px;
}

/* Simulation panel */
.sim-panel {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-lg);
  padding: 32px;
}
.sim-panel-title {
  font-family: var(--font-display);
  font-size: 14px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 28px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border-subtle);
}

/* Result hero */
.result-hero {
  background: linear-gradient(135deg, #0A2540 0%, #0D3070 100%);
  border: 1px solid rgba(37,99,235,0.35);
  border-radius: var(--radius-lg);
  padding: 36px 28px;
  text-align: center;
  position: relative;
  overflow: hidden;
  margin-bottom: 20px;
}
.result-hero::before {
  content: '';
  position: absolute;
  top: 0; left: 50%; transform: translateX(-50%);
  width: 200px; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(96,165,250,0.8), transparent);
}
.result-hero-value {
  font-family: var(--font-display);
  font-size: 52px;
  font-weight: 800;
  color: #FFFFFF;
  letter-spacing: -0.04em;
  line-height: 1;
}
.result-hero-unit {
  font-family: var(--font-mono);
  font-size: 20px;
  color: #93C5FD;
  margin-top: 4px;
}
.result-hero-label {
  font-family: var(--font-body);
  font-size: 12px;
  font-weight: 500;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: rgba(147,197,253,0.7);
  margin-top: 12px;
}

/* Mini KPI */
.mini-kpi {
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 20px;
  text-align: center;
}
.mini-kpi-value {
  font-family: var(--font-mono);
  font-size: 24px;
  font-weight: 500;
  color: var(--text-primary);
  line-height: 1;
}
.mini-kpi-value.positive { color: #34D399; }
.mini-kpi-value.negative { color: #FB7185; }
.mini-kpi-value.neutral  { color: #93C5FD; }
.mini-kpi-label {
  font-family: var(--font-body);
  font-size: 11px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-top: 8px;
  font-weight: 500;
}

/* Alert */
.alert {
  border-radius: var(--radius-sm);
  padding: 12px 16px;
  font-family: var(--font-body);
  font-size: 13px;
  line-height: 1.5;
  margin-top: 12px;
}
.alert-info    { background: rgba(37,99,235,0.1); border: 1px solid rgba(37,99,235,0.25); color: #BFDBFE; }
.alert-success { background: rgba(5,150,105,0.1); border: 1px solid rgba(5,150,105,0.25); color: #6EE7B7; }
.alert-warn    { background: rgba(217,119,6,0.1); border: 1px solid rgba(217,119,6,0.25); color: #FCD34D; }
.alert-strong  { font-weight: 600; }

/* Budget row */
.budget-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 0;
  border-bottom: 1px solid var(--border-subtle);
  font-family: var(--font-body);
  font-size: 13px;
}
.budget-row:last-child { border-bottom: none; }
.budget-row-label { color: var(--text-secondary); font-weight: 400; }
.budget-row-value { color: var(--text-primary); font-family: var(--font-mono); font-weight: 500; }

/* Scenario table */
.scenario-badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 11px;
  font-family: var(--font-mono);
  font-weight: 500;
}
.badge-best { background: rgba(5,150,105,0.15); color: #34D399; border: 1px solid rgba(5,150,105,0.3); }
.badge-good { background: rgba(37,99,235,0.12); color: #93C5FD; border: 1px solid rgba(37,99,235,0.25); }
.badge-low  { background: rgba(217,119,6,0.12); color: #FCD34D; border: 1px solid rgba(217,119,6,0.25); }

/* Tech tab */
.tech-section-title {
  font-family: var(--font-display);
  font-size: 18px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 32px 0 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border-subtle);
}

/* Footer */
.platform-footer {
  background: var(--bg-surface);
  border-top: 1px solid var(--border-subtle);
  padding: 24px 48px;
  text-align: center;
  font-family: var(--font-body);
  font-size: 12px;
  color: var(--text-muted);
  letter-spacing: 0.02em;
  margin-top: 64px;
}

/* Utility */
.text-mono  { font-family: var(--font-mono) !important; }
.text-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); font-weight: 600; }
.mb-4  { margin-bottom: 16px; }
.mb-8  { margin-bottom: 32px; }
.gap-between { display: flex; flex-direction: column; gap: 16px; }
</style>
""", unsafe_allow_html=True)

# ─── Constants & Config ─────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"
MODELS = ["GradientBoosting", "XGBoost", "RandomForest", "LinearRegression", "MLP"]
MODEL_METRICS = {
    "GradientBoosting": {"r2": 0.9989, "rmse": 3.12, "label": "Recommande"},
    "XGBoost":          {"r2": 0.9988, "rmse": 3.20, "label": "Excellent"},
    "RandomForest":     {"r2": 0.9986, "rmse": 3.47, "label": "Excellent"},
    "LinearRegression": {"r2": 0.9956, "rmse": 6.16, "label": "Referentiel"},
    "MLP":              {"r2": 0.9962, "rmse": 5.73, "label": "Deep Learning"},
}

def CHART_STYLE():
    plt.rcParams.update({
        "figure.facecolor":  "#131B2E",
        "axes.facecolor":    "#131B2E",
        "axes.edgecolor":    (1,1,1,0.06),
        "axes.labelcolor":   "#64748B",
        "xtick.color":       "#475569",
        "ytick.color":       "#475569",
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "text.color":        "#F8FAFC",
        "grid.color":        (1,1,1,0.05),
        "grid.alpha":        1,
        "axes.grid":         True,
        "axes.grid.axis":    "y",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.spines.left":  False,
        "font.family":       "sans-serif",
    })

# ─── Data & API helpers ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/raw/dummy_data_hss.csv")

@st.cache_data
def load_shap():
    p = Path("outputs/reports/shap_importance_GradientBoosting.csv")
    return pd.read_csv(p) if p.exists() else None

def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200, r.json() if r.status_code == 200 else {}
    except:
        return False, {}

def call_predict(tv, radio, social, influencer, model):
    try:
        r = requests.post(f"{API_URL}/predict", json={
            "tv": float(tv), "radio": float(radio), "social_media": float(social),
            "influencer": influencer, "model_name": model
        }, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def fmt_M(v):
    if v >= 1000: return f"{v/1000:.1f}B$"
    return f"{v:.1f}M$"

# ─── State & Data ───────────────────────────────────────────────────────────────
df = load_data()
shap_df = load_shap()
api_ok, api_data = check_api()
tv_corr = df[["TV","Sales"]].dropna().corr().iloc[0,1]

# ─── Header ─────────────────────────────────────────────────────────────────────
status_dot = "status-dot" if api_ok else "status-dot offline"
status_text = "Systeme operationnel" if api_ok else "Systeme hors ligne"

st.markdown(f"""
<div class="platform-header">
  <div class="platform-logo">ROI<span>Intelligence</span></div>
  <div class="platform-status">
    <div class="{status_dot}"></div>
    <span>{status_text}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Main Tabs ──────────────────────────────────────────────────────────────────
TAB_OVERVIEW, TAB_SIMULATION, TAB_TECH = st.tabs([
    "Vue d'ensemble",
    "Simulation & ROI",
    "Espace Technique",
])

# ══════════════════════════════════════════════════════
# TAB 1 — VUE D'ENSEMBLE
# ══════════════════════════════════════════════════════
with TAB_OVERVIEW:
    st.markdown('<div class="page-content">', unsafe_allow_html=True)

    # Section header
    st.markdown("""
    <div class="section-header">
      <div class="section-label">Tableau de bord</div>
      <div class="section-title">Performance Marketing</div>
      <div class="section-subtitle">
        Analyse de 4 572 campagnes publicitaires — budgets TV, Radio et Social Media
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row ──
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Campagnes analysees</div>
          <div class="kpi-value">{len(df):,}</div>
          <div class="kpi-sub">Dataset marketing</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Ventes medianes</div>
          <div class="kpi-value emerald">{df['Sales'].median():.0f}<small style="font-size:16px;font-family:var(--font-mono);color:#6EE7B7"> M$</small></div>
          <div class="kpi-sub">Moy. {df['Sales'].mean():.0f} M$</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Budget TV moyen</div>
          <div class="kpi-value accent">{df['TV'].mean():.0f}<small style="font-size:16px;font-family:var(--font-mono);color:#60A5FA"> M$</small></div>
          <div class="kpi-sub">Levier dominant</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Correlation TV / Ventes</div>
          <div class="kpi-value" style="color:#A78BFA">{tv_corr:.4f}</div>
          <div class="kpi-sub">Relation quasi-parfaite</div>
          <div class="kpi-badge blue">r = {tv_corr:.4f}</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Precision du modele</div>
          <div class="kpi-value emerald">99.89<small style="font-size:16px;font-family:var(--font-mono);color:#34D399">%</small></div>
          <div class="kpi-sub">GradientBoosting R2</div>
          <div class="kpi-badge">Valide</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Charts Row 1 ──
    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Budgets publicitaires vs Ventes generees</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Relation entre investissement par canal et performance commerciale</div>', unsafe_allow_html=True)
        CHART_STYLE()
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.patch.set_facecolor("#131B2E")
        channel_colors = {"TV": "#2563EB", "Radio": "#059669", "Social Media": "#D97706"}
        for ax, (col, color) in zip(axes, channel_colors.items()):
            d = df[[col, "Sales"]].dropna()
            ax.scatter(d[col], d["Sales"], alpha=0.15, s=8, color=color, rasterized=True)
            z = np.polyfit(d[col], d["Sales"], 1)
            x_line = np.linspace(d[col].min(), d[col].max(), 200)
            ax.plot(x_line, np.poly1d(z)(x_line), color="white", linewidth=2, alpha=0.9)
            r = d[col].corr(d["Sales"])
            ax.set_title(col, fontsize=11, fontweight="600", pad=12, color="#F8FAFC")
            ax.set_xlabel("Budget (M$)", fontsize=9)
            ax.text(0.05, 0.92, f"r = {r:.4f}", transform=ax.transAxes,
                    fontsize=10, color=color, fontweight="600",
                    fontfamily="DM Mono, monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=(0, 0, 0, 0.4), edgecolor="none"))
            ax.spines["bottom"].set_color((1,1,1,0.08))
        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Matrice de correlation</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Coefficients de Pearson entre variables</div>', unsafe_allow_html=True)
        CHART_STYLE()
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        fig.patch.set_facecolor("#131B2E")
        corr = df[["TV","Radio","Social Media","Sales"]].dropna().corr()
        cmap = plt.cm.get_cmap("Blues")
        im = ax.imshow(corr.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        labels = ["TV","Radio","Social","Ventes"]
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(labels, fontsize=9, color="#94A3B8")
        ax.set_yticklabels(labels, fontsize=9, color="#94A3B8")
        ax.grid(False)
        for i in range(4):
            for j in range(4):
                v = corr.values[i,j]
                tc = "white" if v > 0.7 else "#94A3B8"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=9, color=tc, fontweight="600",
                        fontfamily="DM Mono, monospace")
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02).ax.tick_params(labelsize=8, colors="#64748B")
        plt.tight_layout(pad=1)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Charts Row 2 ──
    col_a, col_b = st.columns([2, 3], gap="large")

    with col_a:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Repartition du budget marketing</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Part relative de chaque canal dans l\'investissement total moyen</div>', unsafe_allow_html=True)
        CHART_STYLE()
        fig, ax = plt.subplots(figsize=(5.5, 4))
        fig.patch.set_facecolor("#131B2E")
        budgets = {"TV": df["TV"].mean(), "Radio": df["Radio"].mean(), "Social Media": df["Social Media"].mean()}
        total_b = sum(budgets.values())
        sizes = [v/total_b*100 for v in budgets.values()]
        colors_pie = ["#2563EB","#059669","#D97706"]
        wedges, texts = ax.pie(sizes, colors=colors_pie, startangle=90,
                               wedgeprops={"edgecolor":"#131B2E","linewidth":4,"width":0.65})
        for i, (wedge, (k, v)) in enumerate(zip(wedges, budgets.items())):
            ang = (wedge.theta2 + wedge.theta1) / 2.0
            r = 0.78
            x = r * np.cos(np.radians(ang))
            y = r * np.sin(np.radians(ang))
            pct = v / total_b * 100
            ax.text(x, y, f"{k}\n{pct:.1f}%\n({v:.0f}M$)",
                    ha="center", va="center",
                    color="white", fontsize=8, fontweight="500",
                    linespacing=1.4)
        ax.text(0, 0, f"{total_b:.0f}\nM$", ha="center", va="center",
                fontsize=13, fontweight="700", color="white")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Distribution des ventes par canal et par type d\'influenceur</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Analyse des performances selon les modalites d\'investissement</div>', unsafe_allow_html=True)
        CHART_STYLE()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor("#131B2E")
        # Distribution ventes
        data = df["Sales"].dropna()
        axes[0].hist(data, bins=40, color="#2563EB", alpha=0.8, edgecolor="none")
        axes[0].axvline(data.median(), color="#34D399", linewidth=1.5, linestyle="--", alpha=0.8, label=f"Med. {data.median():.0f}")
        axes[0].axvline(data.mean(), color="#FCD34D", linewidth=1.5, linestyle="--", alpha=0.8, label=f"Moy. {data.mean():.0f}")
        axes[0].set_title("Distribution Ventes (M$)", fontsize=10, fontweight="600", color="#F8FAFC")
        axes[0].set_xlabel("Ventes (M$)", fontsize=9)
        axes[0].legend(fontsize=8, framealpha=0)
        axes[0].spines["bottom"].set_color((1, 1, 1, 0.08))
        # Ventes par influenceur
        order = df.groupby("Influencer")["Sales"].median().sort_values(ascending=False).index
        inf_colors = ["#2563EB","#059669","#D97706","#7C3AED"]
        for i, inf in enumerate(order):
            d = df[df["Influencer"]==inf]["Sales"].dropna()
            axes[1].boxplot(d, positions=[i], widths=0.5, patch_artist=True,
                           boxprops=dict(facecolor=inf_colors[i], alpha=0.7),
                           medianprops=dict(color="white", linewidth=2),
                           whiskerprops=dict(color="#475569"), capprops=dict(color="#475569"),
                           flierprops=dict(marker=".", color="#334155", markersize=3, alpha=0.5))
        axes[1].set_xticks(range(len(order))); axes[1].set_xticklabels(order, fontsize=9)
        axes[1].set_title("Ventes par type d'influenceur", fontsize=10, fontweight="600", color="#F8FAFC")
        axes[1].set_ylabel("Ventes (M$)", fontsize=9)
        axes[1].spines["bottom"].set_color((1, 1, 1, 0.08))
        plt.tight_layout(pad=1.5); st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 2 — SIMULATION & ROI
# ══════════════════════════════════════════════════════
with TAB_SIMULATION:
    st.markdown('<div class="page-content">', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header">
      <div class="section-label">Optimisation</div>
      <div class="section-title">Simulation Budgetaire</div>
      <div class="section-subtitle">
        Projetez vos ventes et estimez le retour sur investissement de vos campagnes marketing
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="sim-panel">', unsafe_allow_html=True)
        st.markdown('<div class="sim-panel-title">Parametres de la campagne</div>', unsafe_allow_html=True)

        tv     = st.slider("Budget Television (M$)",    min_value=0.0,  max_value=5000.0, value=50.0,  step=1.0)
        radio  = st.slider("Budget Radio (M$)",          min_value=0.0,  max_value=2000.0, value=15.0,  step=0.5)
        social = st.slider("Budget Reseaux Sociaux (M$)",min_value=0.0,  max_value=500.0,  value=3.0,   step=0.5)

        total = tv + radio + social

        st.markdown(f"""
        <div style="margin-top: 24px; padding-top: 20px; border-top: 1px solid var(--border-subtle);">
          <div class="text-label mb-4">Repartition du budget</div>
          <div class="budget-row">
            <span class="budget-row-label">Television</span>
            <span class="budget-row-value">{tv:.1f} M$ &nbsp;<span style="color:var(--text-muted);font-size:11px">({tv/max(total,0.01)*100:.1f}%)</span></span>
          </div>
          <div class="budget-row">
            <span class="budget-row-label">Radio</span>
            <span class="budget-row-value">{radio:.1f} M$ &nbsp;<span style="color:var(--text-muted);font-size:11px">({radio/max(total,0.01)*100:.1f}%)</span></span>
          </div>
          <div class="budget-row">
            <span class="budget-row-label">Reseaux Sociaux</span>
            <span class="budget-row-value">{social:.1f} M$ &nbsp;<span style="color:var(--text-muted);font-size:11px">({social/max(total,0.01)*100:.1f}%)</span></span>
          </div>
          <div class="budget-row" style="margin-top:8px; border-top: 1px solid var(--border-muted); padding-top: 12px;">
            <span style="color:var(--text-primary);font-weight:600;font-size:14px;">Total investi</span>
            <span style="color:var(--text-primary);font-family:var(--font-mono);font-size:16px;font-weight:600;">{total:.1f} M$</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        influencer   = st.selectbox("Type d'influenceur", ["Mega","Micro","Nano","Macro"])
        model_choice = st.selectbox("Modele de prediction", MODELS,
                                    format_func=lambda m: f"{m}  —  R2 {MODEL_METRICS[m]['r2']}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        if not api_ok:
            st.markdown("""
            <div class="result-hero" style="background:linear-gradient(135deg,#1A0A0A,#3D1515);">
              <div class="result-hero-value" style="font-size:32px;color:#FB7185">Systeme hors ligne</div>
              <div class="result-hero-label">Executer : bash launch.sh</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            res = call_predict(tv, radio, social, influencer, model_choice)
            if res:
                sales  = res["predicted_sales"]
                roi    = res["roi_estimate"]
                profit = round(sales - total, 1)
                ratio  = sales / max(total, 0.01)

                roi_color = "#34D399" if roi >= 0 else "#FB7185"
                roi_sign  = "+" if roi >= 0 else ""

                st.markdown(f"""
                <div class="result-hero">
                  <div class="result-hero-value">{sales:.1f}</div>
                  <div class="result-hero-unit">millions de dollars</div>
                  <div class="result-hero-label">Ventes projetees</div>
                </div>
                """, unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    sign_class = "positive" if roi >= 0 else "negative"
                    st.markdown(f"""
                    <div class="mini-kpi">
                      <div class="mini-kpi-value {sign_class}">{roi_sign}{roi:.1f}%</div>
                      <div class="mini-kpi-label">Retour sur invest.</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    pcls = "positive" if profit >= 0 else "negative"
                    st.markdown(f"""
                    <div class="mini-kpi">
                      <div class="mini-kpi-value {pcls}">{profit:+.1f} M$</div>
                      <div class="mini-kpi-label">Profit brut</div>
                    </div>""", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class="mini-kpi">
                      <div class="mini-kpi-value neutral">{ratio:.2f}x</div>
                      <div class="mini-kpi-label">Multiplicateur</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="alert alert-info" style="margin-top:16px">
                  Modele utilise : <span class="alert-strong">{model_choice}</span> &nbsp;|&nbsp;
                  Precision : <span class="alert-strong">{res['model_r2']*100:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)

                if tv < 20 and total > 10:
                    st.markdown("""
                    <div class="alert alert-warn" style="margin-top:8px">
                      Le budget Television est faible. L'analyse SHAP indique que la Television 
                      represente <span class="alert-strong">94.9%</span> de l'impact sur les ventes.
                      Reallouer une part du budget vers la Television pourrait significativement 
                      ameliorer la performance.
                    </div>
                    """, unsafe_allow_html=True)

                if roi > 200:
                    st.markdown(f"""
                    <div class="alert alert-success" style="margin-top:8px">
                      Scenario a fort potentiel. Ce mix budgetaire genere un ROI projete 
                      de <span class="alert-strong">{roi:.0f}%</span>, soit un retour de 
                      <span class="alert-strong">{ratio:.1f}x</span> l'investissement initial.
                    </div>
                    """, unsafe_allow_html=True)

        # ── Analyse comparative de scenarios ──
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="border-top: 1px solid var(--border-subtle); padding-top: 28px;">
          <div class="text-label mb-4">Analyse comparative de scenarios</div>
        </div>
        """, unsafe_allow_html=True)

        scenarios = [
            ("Television prioritaire",   200, 40, 10),
            ("Mix equilibre",            100, 60, 30),
            ("Budget standard",           50, 15,  3),
            ("Investissement eleve",      500,120, 50),
            ("Tres grand compte",        2000,500,200),
            ("Social & Influence",        30, 20, 80),
            ("Budget minimal",            20,  5,  2),
        ]

        if api_ok:
            rows = []
            for name, t, r, s in scenarios:
                res2 = call_predict(t, r, s, "Mega", "GradientBoosting")
                if res2:
                    tot = t+r+s
                    roi2 = res2["roi_estimate"]
                    rows.append({
                        "Scenario": name,
                        "TV (M$)": t, "Radio (M$)": r, "Social (M$)": s,
                        "Investissement (M$)": tot,
                        "Ventes predites (M$)": res2["predicted_sales"],
                        "ROI (%)": round(roi2, 1),
                        "Profit (M$)": round(res2["predicted_sales"]-tot, 1),
                    })
            if rows:
                df_sc = pd.DataFrame(rows).set_index("Scenario")
                best_roi = df_sc["ROI (%)"].max()
                st.dataframe(
                    df_sc.style
                    .format({"Ventes predites (M$)": "{:.1f}", "ROI (%)": "{:+.1f}", "Profit (M$)": "{:+.1f}"})
                    .highlight_max(subset=["Ventes predites (M$)","ROI (%)","Profit (M$)"], color="#0A2A1A")
                    .highlight_min(subset=["ROI (%)"], color="#2A0A12"),
                    use_container_width=True
                )

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 3 — ESPACE TECHNIQUE
# ══════════════════════════════════════════════════════
with TAB_TECH:
    st.markdown('<div class="page-content">', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header">
      <div class="section-label">Documentation</div>
      <div class="section-title">Espace Technique</div>
      <div class="section-subtitle">
        Analyse exploratoire, performances des modeles, interpretabilite et acces API
      </div>
    </div>
    """, unsafe_allow_html=True)

    sub1, sub2, sub3, sub4 = st.tabs([
        "Donnees & EDA",
        "Modeles ML / DL",
        "Interpretabilite",
        "API",
    ])

    # ── EDA ──
    with sub1:
        st.markdown('<div class="tech-section-title">Analyse Exploratoire des Donnees</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        stats = [
            ("Observations",    f"{len(df):,}", "apres nettoyage : 4 566"),
            ("Variables",       "5",            "TV, Radio, Social, Influencer, Sales"),
            ("Valeurs manq.",   "<0.25%",       "traitees par imputation"),
            ("Doublons",        "0",            "dataset propre"),
        ]
        for col, (lbl, val, sub) in zip([c1,c2,c3,c4], stats):
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                  <div class="kpi-label">{lbl}</div>
                  <div class="kpi-value" style="font-size:26px">{val}</div>
                  <div class="kpi-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Distributions des variables</div>', unsafe_allow_html=True)
        CHART_STYLE()
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
        fig.patch.set_facecolor("#131B2E")
        cols_info = [("TV","#2563EB"),("Radio","#059669"),("Social Media","#D97706"),("Sales","#7C3AED")]
        for ax, (col, color) in zip(axes, cols_info):
            data = df[col].dropna()
            ax.hist(data, bins=35, color=color, alpha=0.85, edgecolor="none")
            ax.axvline(data.median(), color="white", linewidth=1.5, linestyle="--", alpha=0.6)
            ax.set_title(col, fontsize=10, fontweight="600", pad=10, color="#F8FAFC")
            ax.set_xlabel("Valeur (M$)", fontsize=8)
            ax.spines["bottom"].set_color((1,1,1,0.08))
            ax.text(0.97, 0.94, f"med={data.median():.1f}", transform=ax.transAxes,
                    fontsize=7, color="white", ha="right", va="top",
                    fontfamily="DM Mono, monospace")
        plt.tight_layout(pad=1.2)
        st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("Donnees brutes (50 premieres lignes)"):
            st.dataframe(df.head(50), use_container_width=True)

    # ── MODELES ──
    with sub2:
        st.markdown('<div class="tech-section-title">Performances comparatives</div>', unsafe_allow_html=True)

        model_table = {
            "Modele":        ["GradientBoosting","XGBoost","RandomForest","LinearRegression","MLP PyTorch"],
            "CV R2":         ["0.9972 ± 0.0015","0.9967 ± 0.0015","0.9972 ± 0.0013","0.9940 ± 0.0033","—"],
            "Test R2":       [0.9989,0.9988,0.9986,0.9956,0.9962],
            "Test RMSE (M$)":[3.12,3.20,3.47,6.16,5.73],
            "Test MAE (M$)": [2.45,2.49,2.61,2.96,4.07],
            "Statut":        ["Recommande","Excellent","Excellent","Referentiel","Deep Learning"],
        }
        st.dataframe(pd.DataFrame(model_table).set_index("Modele"), use_container_width=True)

        cl, cr = st.columns(2, gap="large")
        with cl:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">R2 sur le jeu de test</div>', unsafe_allow_html=True)
            CHART_STYLE()
            fig, ax = plt.subplots(figsize=(7, 3.8))
            fig.patch.set_facecolor("#131B2E")
            mnames = ["GradientBoosting","XGBoost","RandomForest","MLP PyTorch","LinearRegression"]
            r2vals = [0.9989,0.9988,0.9986,0.9962,0.9956]
            mcolors = ["#059669","#2563EB","#2563EB","#7C3AED","#D97706"]
            bars = ax.barh(mnames, r2vals, color=mcolors, edgecolor="none", height=0.5)
            ax.set_xlim(0.990, 1.001)
            for bar, val, c in zip(bars, r2vals, mcolors):
                ax.text(val+0.00005, bar.get_y()+bar.get_height()/2,
                        f"{val:.4f}", va="center", fontsize=9,
                        color="white", fontfamily="DM Mono, monospace", fontweight="500")
            ax.set_xlabel("R2 Score", fontsize=9)
            ax.spines["bottom"].set_color((1,1,1,0.08))
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        with cr:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">RMSE sur le jeu de test (M$)</div>', unsafe_allow_html=True)
            CHART_STYLE()
            fig, ax = plt.subplots(figsize=(7, 3.8))
            fig.patch.set_facecolor("#131B2E")
            rmsevals = [3.12,3.20,3.47,5.73,6.16]
            bars2 = ax.barh(mnames, rmsevals, color=mcolors, edgecolor="none", height=0.5)
            for bar, val in zip(bars2, rmsevals):
                ax.text(val+0.05, bar.get_y()+bar.get_height()/2,
                        f"{val:.2f} M$", va="center", fontsize=9,
                        color="white", fontfamily="DM Mono, monospace", fontweight="500")
            ax.set_xlabel("RMSE (M$)", fontsize=9)
            ax.spines["bottom"].set_color((1,1,1,0.08))
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        mlp_fig = Path("outputs/figures/09_mlp_training.png")
        if mlp_fig.exists():
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Courbe d\'entrainement MLP PyTorch</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-subtitle">Train loss et validation loss sur 111 epochs (early stopping)</div>', unsafe_allow_html=True)
            st.image(str(mlp_fig), use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── INTERPRETABILITE ──
    with sub3:
        st.markdown('<div class="tech-section-title">Analyse SHAP — GradientBoosting</div>', unsafe_allow_html=True)

        if shap_df is not None:
            cl, cr = st.columns([1,1], gap="large")
            with cl:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.markdown('<div class="chart-title">Importance des variables (SHAP)</div>', unsafe_allow_html=True)
                st.markdown('<div class="chart-subtitle">Valeur absolue moyenne des contributions SHAP par feature</div>', unsafe_allow_html=True)
                CHART_STYLE()
                fig, ax = plt.subplots(figsize=(7, 5))
                fig.patch.set_facecolor("#131B2E")
                shap_s = shap_df.sort_values("shap_importance", ascending=True)
                max_v = shap_s["shap_importance"].max()
                bar_colors = ["#2563EB" if v/max_v > 0.05 else "#1E3A5F" for v in shap_s["shap_importance"]]
                bars = ax.barh(shap_s["feature"], shap_s["shap_importance"],
                               color=bar_colors, edgecolor="none", height=0.55)
                for bar, val in zip(bars, shap_s["shap_importance"]):
                    ax.text(val+0.2, bar.get_y()+bar.get_height()/2,
                            f"{val:.3f}", va="center", fontsize=8.5,
                            color="white", fontfamily="DM Mono, monospace")
                ax.set_xlabel("Importance SHAP moyenne (|valeur|)", fontsize=9)
                ax.spines["bottom"].set_color((1,1,1,0.08))
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
                st.markdown('</div>', unsafe_allow_html=True)

            with cr:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.markdown('<div class="chart-title">Repartition de l\'impact sur les ventes</div>', unsafe_allow_html=True)
                CHART_STYLE()
                fig, ax = plt.subplots(figsize=(6, 5))
                fig.patch.set_facecolor("#131B2E")
                total_shap = shap_df["shap_importance"].sum()
                tv_s = shap_df[shap_df["feature"]=="TV"]["shap_importance"].values[0]
                radio_s = shap_df[shap_df["feature"]=="Radio"]["shap_importance"].values[0]
                other_s = total_shap - tv_s - radio_s
                sizes = [tv_s/total_shap*100, radio_s/total_shap*100, other_s/total_shap*100]
                colors_d = ["#2563EB","#059669","#1E293B"]
                labels_d = [f"Television\n{sizes[0]:.1f}%", f"Radio\n{sizes[1]:.1f}%", f"Autres\n{sizes[2]:.1f}%"]
                wedges, _ = ax.pie(sizes, colors=colors_d, startangle=90,
                                   wedgeprops={"edgecolor":"#131B2E","linewidth":4,"width":0.62})
                for i, (label, wedge) in enumerate(zip(labels_d, wedges)):
                    ang = (wedge.theta2 + wedge.theta1) / 2
                    x = 0.82 * np.cos(np.radians(ang))
                    y = 0.82 * np.sin(np.radians(ang))
                    ax.text(x, y, label, ha="center", va="center", fontsize=9,
                            color="white", fontweight="500")
                ax.text(0, 0, "SHAP\nImpact", ha="center", va="center",
                        fontsize=11, color="white", fontweight="600")
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="alert alert-info" style="margin-top: 0">
              La Television concentre <strong>94.9%</strong> de l'impact total sur les ventes selon 
              l'analyse SHAP TreeExplainer. La Radio contribue pour <strong>2.5%</strong>. 
              Le type d'influenceur presente une contribution negligeable de <strong>0.04%</strong>.
            </div>
            """, unsafe_allow_html=True)

        shap_s_path = Path("outputs/figures/10_shap_summary_GradientBoosting.png")
        perm_path   = Path("outputs/figures/12_permutation_importance_GradientBoosting.png")
        if shap_s_path.exists() and perm_path.exists():
            cl2, cr2 = st.columns(2, gap="large")
            with cl2:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.markdown('<div class="chart-title">SHAP Summary Plot</div>', unsafe_allow_html=True)
                st.image(str(shap_s_path), use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with cr2:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.markdown('<div class="chart-title">Permutation Importance</div>', unsafe_allow_html=True)
                st.image(str(perm_path), use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # ── API ──
    with sub4:
        st.markdown('<div class="tech-section-title">Documentation API REST</div>', unsafe_allow_html=True)

        status_label = "Operationnel" if api_ok else "Hors ligne"
        loaded = len(api_data.get("models_loaded", []))
        st.markdown(f"""
        <div class="alert {'alert-success' if api_ok else 'alert-warn'}" style="margin-bottom:24px">
          Statut : <strong>{status_label}</strong> &nbsp;|&nbsp;
          Modeles charges : <strong>{loaded}</strong> &nbsp;|&nbsp;
          URL : <strong>http://localhost:8000</strong> &nbsp;|&nbsp;
          Documentation : <strong>http://localhost:8000/docs</strong>
        </div>
        """, unsafe_allow_html=True)

        endpoints = [
            ("GET",  "/",           "Page d'accueil et liens de navigation"),
            ("GET",  "/health",     "Statut du systeme et modeles charges"),
            ("POST", "/predict",    "Prediction des ventes et estimation du ROI"),
            ("GET",  "/model-info", "Performances et metadonnees des modeles"),
        ]
        ep_data = pd.DataFrame(endpoints, columns=["Methode","Endpoint","Description"])
        st.dataframe(ep_data.set_index("Methode"), use_container_width=True)

        st.markdown('<div class="tech-section-title">Test en direct</div>', unsafe_allow_html=True)
        cl, cr = st.columns(2, gap="large")
        with cl:
            st.markdown('<div class="sim-panel">', unsafe_allow_html=True)
            st.markdown('<div class="sim-panel-title">Requete POST /predict</div>', unsafe_allow_html=True)
            t_api = st.number_input("Television (M$)", 0.0, value=50.0, key="api_tv")
            r_api = st.number_input("Radio (M$)",      0.0, value=15.0, key="api_radio")
            s_api = st.number_input("Social Media (M$)",0.0,value=3.0,  key="api_social")
            i_api = st.selectbox("Influenceur", ["Mega","Micro","Nano","Macro"], key="api_inf")
            m_api = st.selectbox("Modele", MODELS, key="api_model")
            if st.button("Executer la requete", type="primary", use_container_width=True):
                if api_ok:
                    res3 = call_predict(t_api, r_api, s_api, i_api, m_api)
                    if res3:
                        st.success("HTTP 200 OK")
                        st.json(res3)
                else:
                    st.error("API hors ligne — executer : bash launch.sh")
            st.markdown('</div>', unsafe_allow_html=True)

        with cr:
            st.markdown('<div class="sim-panel">', unsafe_allow_html=True)
            st.markdown('<div class="sim-panel-title">Commandes equivalentes</div>', unsafe_allow_html=True)
            st.code(f"""# Prediction
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{{
    "tv": 50.0,
    "radio": 15.0,
    "social_media": 3.0,
    "influencer": "Mega",
    "model_name": "GradientBoosting"
  }}'

# Health check
curl http://localhost:8000/health

# Informations modeles
curl http://localhost:8000/model-info""", language="bash")

            if st.button("Tester /health", use_container_width=True):
                ok2, data2 = check_api()
                if ok2: st.success("HTTP 200 OK"); st.json(data2)
                else: st.error("Connexion refusee")

            if st.button("Tester /model-info", use_container_width=True):
                try:
                    r4 = requests.get(f"{API_URL}/model-info", timeout=3)
                    if r4.status_code == 200: st.success("HTTP 200 OK"); st.json(r4.json())
                except: st.error("API hors ligne")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ─── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="platform-footer">
  Realise par Lorin Kakahoun et Degnon CAPO CHI CHI — Tous droits reserves.
</div>
""", unsafe_allow_html=True)

