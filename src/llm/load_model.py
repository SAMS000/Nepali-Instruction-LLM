from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

import torch


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


def load_model(config: dict):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config["load_in_4bit"],
        bnb_4bit_quant_type=config["bnb_quant_type"],
        bnb_4bit_use_double_quant=config["bnb_use_double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        device_map=config["device_map"],
        trust_remote_code=config["trust_remote_code"],
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )

    return model