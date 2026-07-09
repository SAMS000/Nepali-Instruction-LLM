from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "configs"

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

INTERIM_DATA_DIR = DATA_DIR / "interim"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

BENCHMARK_DIR = DATA_DIR / "benchmark"

MODELS_DIR = PROJECT_ROOT / "models"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"