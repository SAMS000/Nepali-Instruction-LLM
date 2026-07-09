from src.config.loader import load_yaml
from src.config.paths import CONFIG_DIR


model_cfg = load_yaml(CONFIG_DIR / "model.yaml")

print("=" * 60)

print("Base Model")

print(model_cfg["base_model"])

print("=" * 60)