import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import joblib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Marketing ROI Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"

st.markdown("""
<style>
    /* Theme global */
    .stApp { background-color: #0F172A; }
    section[data-testid="stSidebar"] { background-color: #1E293B; }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1E40AF, #1D4ED8);
        border-radius: 12px; padding: 1.2rem 1rem;
        text-align: center; border: 1px solid #3B82F6;
        margin-bottom: 0.5rem;
    }
    .kpi-value { font-size: 2rem; font-weight: 800; color: #FFFFFF; line-height: 1.1; }
    .kpi-label { font-size: 0.78rem; color: #93C5FD; margin-top: 4px; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; }

    .kpi-green { background: linear-gradient(135deg, #065F46, #047857); border-color: #10B981; }
    .kpi-orange { background: linear-gradient(135deg, #92400E, #B45309); border-color: #F59E0B; }
    .kpi-purple { background: linear-gradient(135deg, #4C1D95, #6D28D9); border-color: #8B5CF6; }

    /* Section titles */
    .section-title {
        font-size: 1.15rem; font-weight: 700; color: #F1F5F9;
        border-left: 4px solid #3B82F6; padding-left: 12px;
        margin: 1.5rem 0 0.8rem 0;
    }

    /* Result card prediction */
    .result-card {
        background: linear-gradient(135deg, #064E3B, #065F46);
        border: 2px solid #10B981; border-radius: 16px;
        padding: 1.5rem; text-align: center; margin: 0.5rem 0;
    }
    .result-value { font-size: 3rem; font-weight: 900; color: #34D399; line-height: 1; }
    .result-label { font-size: 0.9rem; color: #6EE7B7; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.08em; }

    .roi-positive { background: linear-gradient(135deg, #064E3B, #065F46); border-color: #10B981; }
    .roi-negative { background: linear-gradient(135deg, #7F1D1D, #991B1B); border-color: #EF4444; }

    /* Badge modele */
    .model-badge {
        display: inline-block; background: #1E40AF;
        color: #BFDBFE; border-radius: 20px;
        padding: 3px 12px; font-size: 0.78rem; font-weight: 600;
        border: 1px solid #3B82F6;
    }

    /* Tableau */
    .dataframe { background-color: #1E293B !important; }
    thead tr th { background-color: #1E3A5F !important; color: #BFDBFE !important; }
    tbody tr:nth-child(even) { background-color: #0F172A !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { color: #94A3B8; font-weight: 500; }
    .stTabs [aria-selected="true"] { color: #3B82F6 !important; border-bottom-color: #3B82F6 !important; }

    /* Slider */
    .stSlider > div > div > div { background: #3B82F6 !important; }

    /* Sidebar */
    .sidebar-section { background: #0F172A; border-radius: 8px; padding: 0.8rem; margin: 0.5rem 0; border: 1px solid #334155; }
    .sidebar-title { color: #60A5FA; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem; }

    /* Alerts */
    .alert-info { background: #1E3A5F; border: 1px solid #3B82F6; border-radius: 8px; padding: 0.8rem 1rem; color: #BFDBFE; font-size: 0.88rem; }
    .alert-success { background: #064E3B; border: 1px solid #10B981; border-radius: 8px; padding: 0.8rem 1rem; color: #6EE7B7; font-size: 0.88rem; }
    .alert-warning { background: #78350F; border: 1px solid #F59E0B; border-radius: 8px; padding: 0.8rem 1rem; color: #FCD34D; font-size: 0.88rem; }

    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/raw/dummy_data_hss.csv")

@st.cache_data
def load_comparison():
    p = Path("outputs/reports/model_comparison.csv")
    return pd.read_csv(p, index_col=0) if p.exists() else None

@st.cache_data
def load_shap():
    p = Path("outputs/reports/shap_importance_GradientBoosting.csv")
    return pd.read_csv(p) if p.exists() else None

def api_status():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200, r.json() if r.status_code == 200 else {}
    except:
        return False, {}

def predict(tv, radio, social, influencer, model):
    try:
        r = requests.post(f"{API_URL}/predict", json={
            "tv": tv, "radio": radio, "social_media": social,
            "influencer": influencer, "model_name": model
        }, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def kpi(value, label, css_class=""):
    return f'<div class="kpi-card {css_class}"><div class="kpi-value">{value}</div><div class="kpi-label">{label}</div></div>'

def fig_style():
    plt.rcParams.update({
        "figure.facecolor": "#1E293B", "axes.facecolor": "#1E293B",
        "axes.edgecolor": "#334155", "axes.labelcolor": "#94A3B8",
        "xtick.color": "#64748B", "ytick.color": "#64748B",
        "text.color": "#F1F5F9", "grid.color": "#334155",
        "grid.alpha": 0.5, "font.family": "sans-serif",
    })

# ─── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Marketing ROI Optimizer")
    st.markdown("---")

    ok, health = api_status()
    if ok:
        loaded = len(health.get("models_loaded", []))
        st.markdown(f'<div class="alert-success">API connectee — {loaded} modeles charges</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-warning">API hors ligne — lancer launch.sh</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-section"><div class="sidebar-title">Projet</div>Mastere Data Engineering & AI<br>EFREI Paris — 2024/2026<br><br><b>Auteur :</b> Lorin Kakahoun<br><b>Dataset :</b> Dummy Marketing HSS<br><b>Lignes :</b> 4 572 campagnes</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section"><div class="sidebar-title">Meilleur modele</div><b>GradientBoosting</b><br>Test R2 : 0.9989<br>RMSE : 3.12 M$<br>CV : 0.9972 +/- 0.0015</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section"><div class="sidebar-title">Liens rapides</div><a href="http://localhost:8000/docs" target="_blank">Swagger API</a> | <a href="http://localhost:8000/health" target="_blank">Health</a></div>', unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────
st.markdown("# 📊 Marketing ROI Optimizer")
st.markdown("*Systeme intelligent de prediction des ventes et optimisation du ROI marketing — EFREI Paris 2026*")
st.markdown("---")

df = load_data()
comparison = load_comparison()
shap_df = load_shap()

# ─── KPIs globaux ───────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
tv_corr = df[["TV","Sales"]].dropna().corr().iloc[0,1]
with c1: st.markdown(kpi(f"{len(df):,}", "Campagnes"), unsafe_allow_html=True)
with c2: st.markdown(kpi(f"{df['Sales'].mean():.0f} M$", "Ventes moyennes", "kpi-green"), unsafe_allow_html=True)
with c3: st.markdown(kpi(f"{df['TV'].mean():.0f} M$", "Budget TV moyen"), unsafe_allow_html=True)
with c4: st.markdown(kpi(f"{tv_corr:.4f}", "Corr. TV / Sales", "kpi-purple"), unsafe_allow_html=True)
with c5: st.markdown(kpi("0.9989", "Best Model R2", "kpi-green"), unsafe_allow_html=True)

st.markdown("")

# ─── Tabs ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈  Donnees & EDA",
    "🤖  Modeles ML/DL",
    "💰  Simulation ROI",
    "🔍  Interpretabilite",
    "⚡  API Live",
])

# ═══════════════════════════════════════
# TAB 1 : EDA
# ═══════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Distributions des variables</div>', unsafe_allow_html=True)
    fig_style()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    colors = ["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"]
    for ax, col, c in zip(axes, ["TV","Radio","Social Media","Sales"], colors):
        data = df[col].dropna()
        ax.hist(data, bins=35, color=c, alpha=0.85, edgecolor="none")
        ax.axvline(data.mean(), color="white", linewidth=1.5, linestyle="--", alpha=0.7)
        ax.set_title(col, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Valeur (M$)", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(pad=1.5)
    st.pyplot(fig); plt.close()

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-title">Matrice de correlation</div>', unsafe_allow_html=True)
        fig_style()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        corr = df[["TV","Radio","Social Media","Sales"]].dropna().corr()
        mask = np.zeros_like(corr, dtype=bool)
        sns.heatmap(corr, annot=True, fmt=".3f", cmap="Blues", ax=ax,
                    linewidths=1, linecolor="#0F172A", annot_kws={"size": 12, "weight": "bold"},
                    vmin=0, vmax=1, cbar_kws={"shrink": 0.8})
        ax.set_title("Correlations Pearson", fontsize=12, fontweight="bold", pad=10)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_r:
        st.markdown('<div class="section-title">Budgets vs Ventes (scatter)</div>', unsafe_allow_html=True)
        fig_style()
        fig, axes = plt.subplots(1, 3, figsize=(9, 4))
        for ax, col, c in zip(axes, ["TV","Radio","Social Media"], ["#3B82F6","#10B981","#F59E0B"]):
            d = df[[col,"Sales"]].dropna()
            ax.scatter(d[col], d["Sales"], alpha=0.25, s=6, color=c)
            z = np.polyfit(d[col], d["Sales"], 1)
            x_l = np.linspace(d[col].min(), d[col].max(), 100)
            ax.plot(x_l, np.poly1d(z)(x_l), color="white", linewidth=1.8)
            r = d[col].corr(d["Sales"])
            ax.set_title(f"{col}\nr = {r:.3f}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Budget M$", fontsize=8); ax.grid(alpha=0.2)
            ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">Analyse par type d\'influenceur</div>', unsafe_allow_html=True)
    fig_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    order = df.groupby("Influencer")["Sales"].median().sort_values(ascending=False).index.tolist()
    palette = {"Mega":"#3B82F6","Micro":"#10B981","Nano":"#F59E0B","Macro":"#8B5CF6"}
    counts = df["Influencer"].value_counts()
    axes[0].bar(counts.index, counts.values, color=[palette.get(x,"#64748B") for x in counts.index])
    axes[0].set_title("Repartition des types", fontweight="bold"); axes[0].grid(axis="y", alpha=0.3)
    axes[0].spines[["top","right"]].set_visible(False)
    means = df.groupby("Influencer")["Sales"].mean().reindex(order)
    axes[1].bar(means.index, means.values, color=[palette.get(x,"#64748B") for x in means.index])
    axes[1].set_title("Ventes moyennes par type", fontweight="bold"); axes[1].grid(axis="y", alpha=0.3)
    axes[1].spines[["top","right"]].set_visible(False)
    for inf_type in order:
        data = df[df["Influencer"]==inf_type]["Sales"].dropna()
        axes[2].boxplot(data, positions=[order.index(inf_type)], widths=0.5, patch_artist=True,
                        boxprops=dict(facecolor=palette.get(inf_type,"#64748B"), alpha=0.7),
                        medianprops=dict(color="white", linewidth=2), whiskerprops=dict(color="#64748B"),
                        capprops=dict(color="#64748B"), flierprops=dict(marker=".", color="#64748B", alpha=0.3))
    axes[2].set_xticks(range(len(order))); axes[2].set_xticklabels(order)
    axes[2].set_title("Distribution Sales", fontweight="bold"); axes[2].grid(axis="y", alpha=0.3)
    axes[2].spines[["top","right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ═══════════════════════════════════════
# TAB 2 : MODELES
# ═══════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Performances comparatives (CV 5-fold + Test)</div>', unsafe_allow_html=True)

    model_data = {
        "Modele": ["GradientBoosting", "XGBoost", "RandomForest", "LinearRegression", "MLP PyTorch"],
        "CV R2": [0.9972, 0.9967, 0.9972, 0.9940, "-"],
        "CV RMSE": [4.80, 5.19, 4.76, 6.92, "-"],
        "Test R2": [0.9989, 0.9988, 0.9986, 0.9956, 0.9962],
        "Test RMSE": [3.12, 3.20, 3.47, 6.16, 5.73],
        "Test MAE": [2.45, 2.49, 2.61, 2.96, 4.07],
        "Statut": ["Meilleur", "Excellent", "Excellent", "Bon", "Bon"],
    }
    df_models = pd.DataFrame(model_data)
    st.dataframe(df_models.set_index("Modele"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">R2 par modele (Test)</div>', unsafe_allow_html=True)
        fig_style()
        fig, ax = plt.subplots(figsize=(7, 4))
        models = ["GradientBoosting", "XGBoost", "RandomForest", "MLP PyTorch", "LinearRegression"]
        r2s = [0.9989, 0.9988, 0.9986, 0.9962, 0.9956]
        colors_bar = ["#10B981","#3B82F6","#3B82F6","#8B5CF6","#F59E0B"]
        bars = ax.barh(models, r2s, color=colors_bar, edgecolor="none", height=0.55)
        ax.set_xlim(0.99, 1.001)
        for bar, val in zip(bars, r2s):
            ax.text(val + 0.00005, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", fontsize=10, fontweight="bold", color="white")
        ax.set_xlabel("R2 Score"); ax.set_title("Test R2 (plus haut = meilleur)", fontweight="bold")
        ax.grid(axis="x", alpha=0.3); ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<div class="section-title">RMSE par modele (Test)</div>', unsafe_allow_html=True)
        fig_style()
        fig, ax = plt.subplots(figsize=(7, 4))
        rmses = [3.12, 3.20, 3.47, 5.73, 6.16]
        colors_bar2 = ["#10B981","#3B82F6","#3B82F6","#8B5CF6","#F59E0B"]
        bars2 = ax.barh(models, rmses, color=colors_bar2, edgecolor="none", height=0.55)
        for bar, val in zip(bars2, rmses):
            ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                    f"{val:.2f} M$", va="center", fontsize=10, fontweight="bold", color="white")
        ax.set_xlabel("RMSE (M$)"); ax.set_title("Test RMSE (plus bas = meilleur)", fontweight="bold")
        ax.grid(axis="x", alpha=0.3); ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-title">Courbe d\'entrainement MLP PyTorch</div>', unsafe_allow_html=True)
    fig_path = Path("outputs/figures/09_mlp_training.png")
    if fig_path.exists():
        st.image(str(fig_path), use_column_width=True)

# ═══════════════════════════════════════
# TAB 3 : SIMULATION ROI
# ═══════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Simulateur budgetaire en temps reel</div>', unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown("**Parametres de la campagne**")
        tv = st.slider("Budget TV (M$)", 0.0, 100.0, 50.0, 0.5,
                       help="La TV est le levier dominant : r=0.9995 avec les ventes")
        radio = st.slider("Budget Radio (M$)", 0.0, 49.0, 15.0, 0.5)
        social = st.slider("Budget Social Media (M$)", 0.0, 14.0, 3.0, 0.1)
        influencer = st.selectbox("Type d'influenceur", ["Mega","Micro","Nano","Macro"],
                                  help="Impact negligeable sur les ventes (SHAP=0.03)")
        model_choice = st.selectbox("Modele de prediction",
                                    ["GradientBoosting","XGBoost","RandomForest","LinearRegression","MLP"])

        total = tv + radio + social
        st.markdown(f"**Budget total investi : {total:.1f} M$**")

        # Mix pie chart
        if total > 0:
            fig_style()
            fig, ax = plt.subplots(figsize=(5, 3))
            sizes = [tv/total*100, radio/total*100, social/total*100]
            labels = [f"TV\n{tv/total*100:.1f}%", f"Radio\n{radio/total*100:.1f}%", f"Social\n{social/total*100:.1f}%"]
            colors_pie = ["#3B82F6","#10B981","#F59E0B"]
            wedges, texts = ax.pie(sizes, labels=labels, colors=colors_pie,
                                   startangle=90, textprops={"color":"white","fontsize":10},
                                   wedgeprops={"edgecolor":"#0F172A","linewidth":2})
            ax.set_title("Mix marketing", color="white", fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_out:
        if not ok:
            st.markdown('<div class="alert-warning">API hors ligne. Lancer : bash launch.sh</div>', unsafe_allow_html=True)
        else:
            result = predict(tv, radio, social, influencer, model_choice)
            if result:
                sales = result["predicted_sales"]
                roi = result["roi_estimate"]
                profit = sales - total

                st.markdown(f'''
                <div class="result-card">
                    <div class="result-value">{sales:.1f} M$</div>
                    <div class="result-label">Ventes predites</div>
                </div>''', unsafe_allow_html=True)

                st.markdown("")
                c_roi, c_profit, c_ratio = st.columns(3)
                roi_class = "kpi-green" if roi > 0 else "kpi-orange"
                with c_roi: st.markdown(kpi(f"{roi:.0f}%", "ROI estime", roi_class), unsafe_allow_html=True)
                with c_profit: st.markdown(kpi(f"{profit:.1f}M$", "Profit brut", "kpi-green" if profit>0 else "kpi-orange"), unsafe_allow_html=True)
                with c_ratio: st.markdown(kpi(f"{sales/max(total,0.01):.2f}x", "Multiplicateur", "kpi-purple"), unsafe_allow_html=True)

                st.markdown("")
                st.markdown(f'<div class="alert-info">Modele : <b>{model_choice}</b> | R2 test : <b>{result["model_r2"]:.4f}</b> | Precision : ±{result["model_r2"]*100:.2f}%</div>', unsafe_allow_html=True)

                if tv < 15:
                    st.markdown('<div class="alert-warning">Budget TV faible — TV represente 94.9% de l\'impact (SHAP). Augmenter ce budget pour maximiser les ventes.</div>', unsafe_allow_html=True)

    # Comparaison scenarios
    st.markdown('<div class="section-title">Comparaison de scenarios budgetaires</div>', unsafe_allow_html=True)
    scenarios = {
        "TV dominante (80/10/2)":    (80, 10, 2),
        "Mix equilibre (40/20/8)":   (40, 20, 8),
        "Budget standard (50/15/3)": (50, 15, 3),
        "Social First (20/10/14)":   (20, 10, 14),
        "Budget minimal (20/5/2)":   (20,  5, 2),
        "Budget maximal (100/49/14)":(100,49,14),
    }
    if ok:
        rows = []
        for name, (t, r, s) in scenarios.items():
            res = predict(t, r, s, "Mega", "GradientBoosting")
            if res:
                rows.append({
                    "Scenario": name,
                    "TV": t, "Radio": r, "Social": s,
                    "Budget total (M$)": t+r+s,
                    "Ventes predites (M$)": res["predicted_sales"],
                    "ROI (%)": res["roi_estimate"],
                    "Profit (M$)": round(res["predicted_sales"]-(t+r+s), 1),
                })
        if rows:
            df_sc = pd.DataFrame(rows).set_index("Scenario")
            st.dataframe(df_sc.style.highlight_max(subset=["Ventes predites (M$)","ROI (%)","Profit (M$)"], color="#064E3B")
                         .highlight_min(subset=["Ventes predites (M$)"], color="#7F1D1D"),
                         use_container_width=True)

# ═══════════════════════════════════════
# TAB 4 : INTERPRETABILITE
# ═══════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Importance des variables — SHAP GradientBoosting</div>', unsafe_allow_html=True)

    if shap_df is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            fig_style()
            fig, ax = plt.subplots(figsize=(7, 5))
            shap_s = shap_df.sort_values("shap_importance", ascending=True)
            colors_shap = ["#3B82F6" if v > 1 else "#1E3A5F" for v in shap_s["shap_importance"]]
            bars = ax.barh(shap_s["feature"], shap_s["shap_importance"], color=colors_shap, edgecolor="none", height=0.6)
            ax.axvline(x=0, color="#334155", linewidth=0.5)
            for bar, val in zip(bars, shap_s["shap_importance"]):
                ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                        f"{val:.2f}", va="center", fontsize=9, color="white")
            ax.set_xlabel("Importance SHAP (|valeur moyenne|)")
            ax.set_title("SHAP Feature Importance", fontweight="bold")
            ax.grid(axis="x", alpha=0.2); ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            # Donut chart SHAP
            fig_style()
            fig, ax = plt.subplots(figsize=(6, 5))
            total_shap = shap_df["shap_importance"].sum()
            tv_shap = shap_df[shap_df["feature"]=="TV"]["shap_importance"].values[0]
            radio_shap = shap_df[shap_df["feature"]=="Radio"]["shap_importance"].values[0]
            other_shap = total_shap - tv_shap - radio_shap
            sizes = [tv_shap/total_shap*100, radio_shap/total_shap*100, other_shap/total_shap*100]
            labels = [f"TV\n{sizes[0]:.1f}%", f"Radio\n{sizes[1]:.1f}%", f"Autres\n{sizes[2]:.1f}%"]
            colors_d = ["#3B82F6","#10B981","#334155"]
            wedges, texts = ax.pie(sizes, labels=labels, colors=colors_d, startangle=90,
                                   textprops={"color":"white","fontsize":11,"fontweight":"bold"},
                                   wedgeprops={"edgecolor":"#0F172A","linewidth":3, "width":0.6})
            ax.text(0, 0, "SHAP\nImpact", ha="center", va="center", fontsize=12, color="white", fontweight="bold")
            ax.set_title("Repartition de l'impact\nsur les ventes (SHAP)", fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown('<div class="section-title">Interpretation business</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(kpi("94.9%", "Impact TV sur ventes", "kpi-purple"), unsafe_allow_html=True)
            st.markdown('<div class="alert-info">La TV est le moteur principal. Chaque million investi en TV a un impact 38x superieur a la Radio.</div>', unsafe_allow_html=True)
        with c2:
            st.markdown(kpi("2.5%", "Impact Radio", "kpi-green"), unsafe_allow_html=True)
            st.markdown('<div class="alert-info">La Radio est le 2eme levier. Maintenir un budget >15M$ pour un effet complementaire significatif.</div>', unsafe_allow_html=True)
        with c3:
            st.markdown(kpi("0.04%", "Impact Influenceur", "kpi-orange"), unsafe_allow_html=True)
            st.markdown('<div class="alert-warning">Le type d\'influenceur (Mega/Micro/Nano/Macro) n\'a pas d\'impact mesurable sur les ventes globales.</div>', unsafe_allow_html=True)

    fig1 = Path("outputs/figures/10_shap_summary_GradientBoosting.png")
    fig2 = Path("outputs/figures/12_permutation_importance_GradientBoosting.png")
    if fig1.exists() and fig2.exists():
        st.markdown('<div class="section-title">SHAP Summary Plot & Permutation Importance</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1: st.image(str(fig1), use_column_width=True)
        with col2: st.image(str(fig2), use_column_width=True)

# ═══════════════════════════════════════
# TAB 5 : API LIVE
# ═══════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">Test de l\'API en direct</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Endpoint POST /predict**")
        api_tv = st.number_input("TV budget (M$)", 0.0, 500.0, 50.0)
        api_radio = st.number_input("Radio budget (M$)", 0.0, 200.0, 15.0)
        api_social = st.number_input("Social Media budget (M$)", 0.0, 100.0, 3.0)
        api_inf = st.selectbox("Influencer", ["Mega","Micro","Nano","Macro"], key="api_inf")
        api_model = st.selectbox("Modele", ["GradientBoosting","XGBoost","RandomForest","LinearRegression","MLP"], key="api_model")

        if st.button("Envoyer la requete POST /predict", type="primary", use_container_width=True):
            if ok:
                res = predict(api_tv, api_radio, api_social, api_inf, api_model)
                if res:
                    st.success("Reponse HTTP 200 OK")
                    st.json(res)
                else:
                    st.error("Erreur API")
            else:
                st.error("API hors ligne")

    with c2:
        st.markdown("**Status des endpoints**")
        if st.button("Tester GET /health", use_container_width=True):
            if ok:
                st.success("HTTP 200 OK")
                st.json(health)
            else:
                st.error("HTTP — API hors ligne")

        if st.button("Tester GET /model-info", use_container_width=True):
            try:
                r = requests.get(f"{API_URL}/model-info", timeout=3)
                if r.status_code == 200:
                    st.success("HTTP 200 OK")
                    st.json(r.json())
            except:
                st.error("API hors ligne")

        st.markdown("**Requete curl equivalente :**")
        curl_cmd = f"""curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"tv": {api_tv}, "radio": {api_radio}, "social_media": {api_social}, "influencer": "{api_inf}", "model_name": "{api_model}"}}'"""
        st.code(curl_cmd, language="bash")

        st.markdown("**Documentation interactive :**")
        st.markdown("Ouvrir [Swagger UI](http://localhost:8000/docs) dans un nouvel onglet.")
