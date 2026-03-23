import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger()

# Style global
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})
PALETTE = "Blues_d"
OUTPUT_DIR = Path("outputs/figures/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_fig(name: str) -> None:
    path = OUTPUT_DIR / f"{name}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure sauvegardee : {path}")

def plot_distributions(df: pd.DataFrame) -> None:
    # Distribution de chaque variable numerique avec KDE
    numeric_cols = ["TV", "Radio", "Social Media", "Sales"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Distribution des variables numeriques", fontsize=14, fontweight="bold")
    
    for ax, col in zip(axes.flatten(), numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#2196F3", bins=40)
        ax.axvline(df[col].median(), color="red", linestyle="--", linewidth=1.5, label=f"Mediane : {df[col].median():.1f}")
        ax.axvline(df[col].mean(), color="orange", linestyle="--", linewidth=1.5, label=f"Moyenne : {df[col].mean():.1f}")
        ax.set_title(col)
        ax.set_xlabel("Valeur (M$)")
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    save_fig("01_distributions")

def plot_target_analysis(df: pd.DataFrame) -> None:
    # Analyse approfondie de la variable cible Sales
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Analyse de la cible : Sales (M$)", fontsize=14, fontweight="bold")
    
    # Distribution + KDE
    sns.histplot(df["Sales"].dropna(), kde=True, ax=axes[0], color="#1976D2", bins=50)
    axes[0].set_title("Distribution des ventes")
    axes[0].set_xlabel("Sales (M$)")
    
    # Boxplot global
    axes[1].boxplot(df["Sales"].dropna(), vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#BBDEFB"))
    axes[1].set_title("Boxplot Sales")
    axes[1].set_ylabel("Sales (M$)")
    
    # Sales par type d'influenceur
    order = df.groupby("Influencer")["Sales"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="Influencer", y="Sales", order=order,
                ax=axes[2], palette="Blues")
    axes[2].set_title("Sales par type d'Influenceur")
    axes[2].set_xlabel("Type")
    
    plt.tight_layout()
    save_fig("02_target_analysis")

def plot_correlation_matrix(df: pd.DataFrame) -> None:
    # Matrice de correlation sur les variables numeriques
    numeric_df = df[["TV", "Radio", "Social Media", "Sales"]].dropna()
    corr = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="Blues", ax=ax,
                linewidths=0.5, vmin=-1, vmax=1)
    ax.set_title("Matrice de correlation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig("03_correlation_matrix")
    
    # Affichage des correlations avec la cible
    logger.info("Correlations avec Sales :")
    for col, val in corr["Sales"].drop("Sales").sort_values(ascending=False).items():
        logger.info(f"  {col:20s} : {val:.4f}")

def plot_scatter_features(df: pd.DataFrame) -> None:
    # Scatterplots budgets vs Sales avec regression lineaire
    budget_cols = ["TV", "Radio", "Social Media"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Budgets publicitaires vs Ventes", fontsize=14, fontweight="bold")
    
    for ax, col in zip(axes, budget_cols):
        df_clean = df[[col, "Sales"]].dropna()
        ax.scatter(df_clean[col], df_clean["Sales"], alpha=0.3, color="#1565C0", s=10)
        # Ligne de regression
        z = np.polyfit(df_clean[col], df_clean["Sales"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_clean[col].min(), df_clean[col].max(), 100)
        ax.plot(x_line, p(x_line), "r-", linewidth=2)
        ax.set_xlabel(f"{col} budget (M$)")
        ax.set_ylabel("Sales (M$)")
        ax.set_title(f"{col} vs Sales")
        corr_val = df_clean[col].corr(df_clean["Sales"])
        ax.annotate(f"r = {corr_val:.3f}", xy=(0.05, 0.92), xycoords="axes fraction",
                    fontsize=10, color="red")
    
    plt.tight_layout()
    save_fig("04_scatter_features_vs_sales")

def plot_influencer_analysis(df: pd.DataFrame) -> None:
    # Analyse detaillee par type d'influenceur
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Analyse par type d'Influenceur", fontsize=14, fontweight="bold")
    
    # Repartition des types
    counts = df["Influencer"].value_counts()
    axes[0, 0].bar(counts.index, counts.values, color="#1976D2")
    axes[0, 0].set_title("Repartition des types d'influenceur")
    axes[0, 0].set_ylabel("Nombre de campagnes")
    
    # Sales moyennes par influenceur
    means = df.groupby("Influencer")["Sales"].mean().sort_values(ascending=False)
    axes[0, 1].bar(means.index, means.values, color="#42A5F5")
    axes[0, 1].set_title("Ventes moyennes par influenceur")
    axes[0, 1].set_ylabel("Sales moyen (M$)")
    
    # Budget TV moyen par influenceur
    tv_means = df.groupby("Influencer")["TV"].mean().sort_values(ascending=False)
    axes[1, 0].bar(tv_means.index, tv_means.values, color="#64B5F6")
    axes[1, 0].set_title("Budget TV moyen par influenceur")
    axes[1, 0].set_ylabel("TV budget (M$)")
    
    # Distribution Sales par influenceur (violin)
    sns.violinplot(data=df, x="Influencer", y="Sales", ax=axes[1, 1], palette="Blues")
    axes[1, 1].set_title("Distribution Sales par influenceur")
    
    plt.tight_layout()
    save_fig("05_influencer_analysis")

def plot_budget_mix(df: pd.DataFrame) -> None:
    # Analyse du mix marketing : contribution relative de chaque canal
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Mix Marketing - Analyse des budgets", fontsize=14, fontweight="bold")
    
    # Budget moyen par canal
    budget_means = {"TV": df["TV"].mean(), "Radio": df["Radio"].mean(), "Social Media": df["Social Media"].mean()}
    axes[0].bar(budget_means.keys(), budget_means.values(), color=["#1565C0", "#42A5F5", "#90CAF9"])
    axes[0].set_title("Budget moyen par canal (M$)")
    axes[0].set_ylabel("Budget moyen (M$)")
    for i, (k, v) in enumerate(budget_means.items()):
        axes[0].text(i, v + 0.3, f"{v:.1f}", ha="center", fontweight="bold")
    
    # Part de chaque canal dans le budget total
    total = sum(budget_means.values())
    shares = {k: v/total*100 for k, v in budget_means.items()}
    axes[1].pie(shares.values(), labels=shares.keys(), autopct="%1.1f%%",
                colors=["#1565C0", "#42A5F5", "#90CAF9"], startangle=90)
    axes[1].set_title("Part dans le budget total")
    
    plt.tight_layout()
    save_fig("06_budget_mix")

def run_full_eda(df: pd.DataFrame) -> None:
    # Execute l'EDA complete et sauvegarde tous les graphiques
    logger.info("Demarrage EDA complete...")
    plot_distributions(df)
    plot_target_analysis(df)
    plot_correlation_matrix(df)
    plot_scatter_features(df)
    plot_influencer_analysis(df)
    plot_budget_mix(df)
    logger.info("EDA terminee - tous les graphiques sauvegardes dans outputs/figures/")
