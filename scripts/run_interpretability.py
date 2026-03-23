import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.interpretability import run_interpretability

if __name__ == "__main__":
    # Analyse sur GradientBoosting (meilleur modele test R2)
    run_interpretability("GradientBoosting")
    print("\nInterpretabilite terminee - figures dans outputs/figures/")
