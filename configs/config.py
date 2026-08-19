from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "Datasets"

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

BATCH_SIZE = 25