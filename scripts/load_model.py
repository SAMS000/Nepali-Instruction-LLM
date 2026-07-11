from src.config.loader import load_yaml
from src.config.paths import CONFIG_DIR

from src.llm.load_model import (
    load_model,
    load_tokenizer,
)


def main():

    config = load_yaml(CONFIG_DIR / "model.yaml")

    print("=" * 60)
    print("Loading tokenizer...")
    print("=" * 60)

    tokenizer = load_tokenizer(
        model_name=config["base_model"],
        trust_remote_code=config["trust_remote_code"],
        use_fast=config["use_fast"],
        padding_side=config["padding_side"],
    )

    print("✓ Tokenizer loaded")

    print()

    print("=" * 60)
    print("Loading model...")
    print("=" * 60)

    model = load_model(config)

    print("✓ Model loaded")

    print()

    print("Model dtype:", model.dtype)
    print("Device:", model.device)

    print()

    print("Everything loaded successfully!")
    

if __name__ == "__main__":
    main()