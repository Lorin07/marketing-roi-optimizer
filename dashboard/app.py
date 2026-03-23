import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import requests
import joblib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration de la page
st.set_page_config(
    page_title="Marketing ROI Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"

# Style CSS personnalise
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1565C0, #1976D2);
        padding: 1rem; border-radius: 10px; color: white;
        text-align: center; margin: 0.3rem 0;
    }
    .metric-value { font-size: 2rem; font-weight: bold; }
    .metric-label { font-size: 0.85rem; opacity: 0.85; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("data/raw/dummy_data_hss.csv")
    return df

@st.cache_data
def load_model_comparison():
    path = Path("outputs/reports/model_comparison.csv")
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return None

@st.cache_data
def load_shap_importance():
    path = Path("outputs/reports/shap_importance_GradientBoosting.csv")
    if path.exists():
        return pd.read_csv(path)
    return None

def call_predict_api(tv, radio, social_media, influencer, model_name):
    try:
        resp = requests.post(f"{API_URL}/predict", json={
            "tv": tv, "radio": radio, "social_media": social_media,
            "influencer": influencer, "model_name": model_name
        }, timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None

def check_api_status():
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False

# Sidebar
with st.sidebar:
    st.title("📊 Marketing ROI Optimizer")
    st.markdown("---")
    
    api_ok = check_api_status()
    if api_ok:
        st.success("API : Connectee")
    else:
        st.error("API : Hors ligne")
        st.info("Lancer : uvicorn api.main:app --port 8000")
    
    st.markdown("---")
    st.markdown("**Projet ML/DL - EFREI Paris**")
    st.markdown("Mastere Data Engineering & AI")
    st.markdown("**Dataset** : Dummy Marketing HSS")
    st.markdown("**Modeles** : LR | RF | GB | XGB | MLP")

# Chargement des donnees
df = load_data()
comparison = load_model_comparison()
shap_df = load_shap_importance()

# Titre principal
st.title("Marketing ROI Optimizer")
st.markdown("Systeme intelligent de prediction des ventes et optimisation du budget marketing")
st.markdown("---")

# Onglets
tab1, tab2, tab3, tab4 = st.tabs([
    "Apercu des donnees",
    "Comparaison des modeles",
    "Simulation budgetaire",
    "Interpretabilite",
])

# TAB 1 : DONNEES
with tab1:
    st.header("Analyse Exploratoire des Donnees")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Campagnes</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{df['Sales'].mean():.0f}M$</div>
            <div class="metric-label">Ventes moyennes</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{df['TV'].mean():.0f}M$</div>
            <div class="metric-label">Budget TV moyen</div></div>""", unsafe_allow_html=True)
    with col4:
        tv_corr = df[["TV","Sales"]].dropna().corr().iloc[0,1]
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{tv_corr:.4f}</div>
            <div class="metric-label">Correlation TV/Sales</div></div>""", unsafe_allow_html=True)
    
    st.markdown("")
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Distribution des ventes")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df["Sales"].dropna(), bins=50, color="#1976D2", alpha=0.8, edgecolor="white")
        ax.axvline(df["Sales"].mean(), color="red", linestyle="--", linewidth=1.5,
                   label=f"Moyenne : {df['Sales'].mean():.1f}")
        ax.set_xlabel("Sales (M$)")
        ax.set_ylabel("Frequence")
        ax.legend()
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()
    
    with col_right:
        st.subheader("Correlation budgets vs ventes")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        numeric_df = df[["TV","Radio","Social Media","Sales"]].dropna()
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, fmt=".3f", cmap="Blues", ax=ax,
                    linewidths=0.5, vmin=-1, vmax=1)
        ax.set_title("")
        st.pyplot(fig)
        plt.close()
    
    st.subheader("Ventes par type d'influenceur")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    order = df.groupby("Influencer")["Sales"].median().sort_values(ascending=False).index
    for i, inf_type in enumerate(order):
        data = df[df["Influencer"] == inf_type]["Sales"].dropna()
        ax.boxplot(data, positions=[i], widths=0.5,
                   patch_artist=True,
                   boxprops=dict(facecolor=f"#{hex(0x1565C0 + i*0x101010)[2:]}"))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("Sales (M$)")
    ax.spines[["top","right"]].set_visible(False)
    st.pyplot(fig)
    plt.close()
    
    with st.expander("Voir les donnees brutes (50 premieres lignes)"):
        st.dataframe(df.head(50), use_container_width=True)

# TAB 2 : MODELES
with tab2:
    st.header("Comparaison des Modeles ML/DL")
    
    if comparison is not None:
        st.subheader("Tableau des performances")
        st.dataframe(comparison.style.highlight_max(
            subset=["Test R2"], color="#BBDEFB"
        ).highlight_min(
            subset=["Test RMSE", "Test MAE"], color="#BBDEFB"
        ), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_path = Path("outputs/figures/07_model_comparison.png")
        if fig_path.exists():
            st.subheader("Comparaison R2 et RMSE (CV)")
            st.image(str(fig_path), use_container_width=True)
    
    with col2:
        fig_path = Path("outputs/figures/08_predictions_vs_actual.png")
        if fig_path.exists():
            st.subheader("Predictions vs Valeurs Reelles")
            st.image(str(fig_path), use_container_width=True)
    
    st.subheader("Courbe d'entrainement MLP")
    fig_path = Path("outputs/figures/09_mlp_training.png")
    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)

# TAB 3 : SIMULATION
with tab3:
    st.header("Simulation Budgetaire en Temps Reel")
    st.markdown("Ajustez les budgets et observez l'impact predit sur les ventes.")
    
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        st.subheader("Parametres de la campagne")
        
        tv_budget = st.slider("Budget TV (M$)", 0.0, 100.0, 50.0, 0.5)
        radio_budget = st.slider("Budget Radio (M$)", 0.0, 50.0, 15.0, 0.5)
        social_budget = st.slider("Budget Social Media (M$)", 0.0, 15.0, 3.0, 0.1)
        influencer_type = st.selectbox("Type d'influenceur", ["Mega", "Micro", "Nano", "Macro"])
        model_choice = st.selectbox(
            "Modele de prediction",
            ["GradientBoosting", "XGBoost", "RandomForest", "LinearRegression", "MLP"]
        )
        
        total_invest = tv_budget + radio_budget + social_budget
        st.info(f"Budget total investi : **{total_invest:.1f} M$**")
        
        # Mix marketing visuel
        if total_invest > 0:
            fig, ax = plt.subplots(figsize=(5, 3))
            shares = [tv_budget/total_invest*100, radio_budget/total_invest*100,
                      social_budget/total_invest*100]
            ax.pie(shares, labels=["TV","Radio","Social"],
                   autopct="%1.1f%%", colors=["#1565C0","#42A5F5","#90CAF9"],
                   startangle=90)
            ax.set_title("Mix marketing", fontsize=10)
            st.pyplot(fig)
            plt.close()
    
    with col_result:
        st.subheader("Prediction")
        
        if not api_ok:
            st.error("API hors ligne. Lancer uvicorn pour obtenir les predictions.")
        else:
            result = call_predict_api(
                tv_budget, radio_budget, social_budget,
                influencer_type, model_choice
            )
            
            if result:
                sales = result["predicted_sales"]
                roi = result["roi_estimate"]
                r2 = result["model_r2"]
                
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{sales:.1f} M$</div>
                    <div class="metric-label">Ventes predites</div></div>""",
                    unsafe_allow_html=True)
                st.markdown("")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    color = "#27AE60" if roi > 0 else "#E74C3C"
                    st.markdown(f"""<div style="background:{color};padding:1rem;
                        border-radius:10px;color:white;text-align:center;">
                        <div style="font-size:1.8rem;font-weight:bold;">{roi:.1f}%</div>
                        <div style="font-size:0.85rem;">ROI estime</div></div>""",
                        unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"""<div style="background:#1976D2;padding:1rem;
                        border-radius:10px;color:white;text-align:center;">
                        <div style="font-size:1.8rem;font-weight:bold;">{r2:.4f}</div>
                        <div style="font-size:0.85rem;">R2 du modele</div></div>""",
                        unsafe_allow_html=True)
                
                st.markdown("")
                st.markdown("**Analyse du scenario :**")
                profit = sales - total_invest
                st.write(f"- Investissement total : {total_invest:.1f} M$")
                st.write(f"- Ventes predites : {sales:.1f} M$")
                st.write(f"- Profit estime : {profit:.1f} M$")
                st.write(f"- Retour par dollar investi : {sales/max(total_invest,0.01):.2f}x")
                
                if tv_budget < 10:
                    st.warning("Budget TV faible. TV est le levier le plus impactant (SHAP=78).")
            else:
                st.error("Erreur de prediction. Verifier que l'API est bien lancee.")
        
        st.subheader("Comparaison multi-scenarios")
        scenarios = {
            "TV forte (80/10/2)": (80, 10, 2),
            "Mix equilibre (40/20/8)": (40, 20, 8),
            "Social first (20/10/14)": (20, 10, 14),
            "Budget minimal (20/5/2)": (20, 5, 2),
        }
        
        if api_ok:
            scenario_results = []
            for name, (tv, radio, soc) in scenarios.items():
                res = call_predict_api(tv, radio, soc, "Mega", "GradientBoosting")
                if res:
                    scenario_results.append({
                        "Scenario": name,
                        "TV": tv, "Radio": radio, "Social": soc,
                        "Budget": tv+radio+soc,
                        "Ventes predites": res["predicted_sales"],
                        "ROI (%)": res["roi_estimate"],
                    })
            if scenario_results:
                df_scen = pd.DataFrame(scenario_results)
                st.dataframe(df_scen.set_index("Scenario").style.highlight_max(
                    subset=["Ventes predites"], color="#BBDEFB"
                ), use_container_width=True)

# TAB 4 : INTERPRETABILITE
with tab4:
    st.header("Interpretabilite des Modeles")
    
    if shap_df is not None:
        st.subheader("Importance SHAP - GradientBoosting (Meilleur modele)")
        
        fig, ax = plt.subplots(figsize=(9, 5))
        shap_sorted = shap_df.sort_values("shap_importance", ascending=True)
        colors = ["#1565C0" if v > 1 else "#90CAF9" for v in shap_sorted["shap_importance"]]
        ax.barh(shap_sorted["feature"], shap_sorted["shap_importance"], color=colors)
        ax.set_xlabel("Importance SHAP moyenne (|valeur|)")
        ax.set_title("Quelles variables influencent le plus les ventes ?")
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()
        
        st.markdown("""
        **Lecture business :**
        - **TV (78.2)** : Le budget TV est de loin le facteur le plus determinant des ventes
        - **Radio (2.0)** : Second levier, 38x moins impactant que TV
        - **Influencer (0.03)** : Le type d'influenceur a un impact negligeable sur les ventes globales
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        fig_path = Path("outputs/figures/10_shap_summary_GradientBoosting.png")
        if fig_path.exists():
            st.subheader("SHAP Summary Plot")
            st.image(str(fig_path), use_container_width=True)
    with col2:
        fig_path = Path("outputs/figures/12_permutation_importance_GradientBoosting.png")
        if fig_path.exists():
            st.subheader("Permutation Importance")
            st.image(str(fig_path), use_container_width=True)
    
    fig_path = Path("outputs/figures/13_shap_dependence_TV_GradientBoosting.png")
    if fig_path.exists():
        st.subheader("Relation TV Budget -> Impact sur les ventes (SHAP Dependence)")
        st.image(str(fig_path), use_container_width=True)
