import yaml
from pathlib import Path

def load_config(config_path: str = "config/config.yaml") -> dict:
    # Charge le fichier de configuration YAML depuis la racine du projet
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier de config introuvable : {config_path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)
