"""Shared model + LoRA utilities."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
LORA_RANK = 8

def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32
    )
    return model, tokenizer

def wrap_with_lora(model, rank=LORA_RANK):
    config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["q_proj", "v_proj"],  # keep small for speed
        lora_dropout=0.0,  # deterministic, per SPEC quantization requirement
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)

def get_lora_state_dict(model):
    """Extract just the LoRA adapter weights (what gets committed)."""
    return {k: v.detach().clone() for k, v in model.named_parameters() if "lora_" in k}

def set_lora_state_dict(model, state_dict):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state_dict:
                param.copy_(state_dict[name])

if __name__ == "__main__":
    # quick smoke test
    model, tok = load_base_model()
    model = wrap_with_lora(model)
    model.print_trainable_parameters()
    sd = get_lora_state_dict(model)
    print(f"LoRA params: {len(sd)} tensors")
    for k, v in list(sd.items())[:3]:
        print(f"  {k}: {v.shape}")