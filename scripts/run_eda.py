import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_raw_data
from src.data.eda import run_full_eda
from src.utils.logger import get_logger

logger = get_logger()

if __name__ == "__main__":
    df = load_raw_data()
    run_full_eda(df)
    print("\nEDA terminee. Figures dans : outputs/figures/")
    import os
    for f in sorted(os.listdir("outputs/figures/")):
        print(f"  {f}")
