from pathlib import Path

import yaml


def load_yaml(config_path: str | Path):
    config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)