from transformers import AutoTokenizer


def load_tokenizer(
    model_name: str,
    trust_remote_code: bool = True,
    use_fast: bool = True,
    padding_side: str = "left",
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast,
    )

    tokenizer.padding_side = padding_side

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer