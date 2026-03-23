import sys
from loguru import logger
from pathlib import Path

def get_logger(name: str = "marketing_roi"):
    # Configure loguru avec rotation de fichier et sortie console
    Path("logs").mkdir(exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}", level="INFO")
    logger.add(f"logs/{name}.log", rotation="10 MB", retention="7 days", level="DEBUG")
    return logger
