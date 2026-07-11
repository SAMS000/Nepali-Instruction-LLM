from src.config.loader import load_yaml
from src.config.paths import CONFIG_DIR
from src.llm.load_model import load_tokenizer


def main():
    config = load_yaml(CONFIG_DIR / "model.yaml")

    print("=" * 60)
    print("Downloading Tokenizer...")
    print("=" * 60)

    tokenizer = load_tokenizer(
        model_name=config["base_model"],
        trust_remote_code=config["trust_remote_code"],
        use_fast=config["use_fast"],
        padding_side=config["padding_side"],
    )

    print(f"\nModel: {config['base_model']}")
    print(f"Vocabulary Size: {len(tokenizer)}")
    print(f"EOS Token: {tokenizer.eos_token}")
    print(f"PAD Token: {tokenizer.pad_token}")
    print("\nTokenizer downloaded successfully.")


if __name__ == "__main__":
    main()