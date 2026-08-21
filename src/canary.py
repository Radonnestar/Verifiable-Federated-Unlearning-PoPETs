"""
Canary seeding and extraction-rate measurement.
Per SPEC.md's out-of-scope list, this covers single-client canaries only
- no cross-client duplication study this week.
"""
import torch
import numpy as np
import secrets
from model_utils import load_base_model, wrap_with_lora, set_lora_state_dict
from data_utils import TextDataset

NUM_CANARIES = 5
CANARY_TEMPLATE = "The secret unlock code for client {cid} is {token}."

def generate_canaries(client_id, n=NUM_CANARIES, seed=None):
    """High-entropy, unique-per-client synthetic sequences."""
    rng = secrets.SystemRandom()
    canaries = []
    for _ in range(n):
        token = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(8))
        canaries.append(CANARY_TEMPLATE.format(cid=client_id, token=token))
    return canaries

def inject_canaries(client_texts, client_id, canaries):
    """Insert canary sequences into a client's local text list."""
    return client_texts + canaries

def canary_loss(model, tokenizer, canary_text, device="cpu"):
    """
    Per-token average loss of the model on a canary sequence.
    Lower loss = model has memorized / can reproduce it more easily.
    This is the extraction-rate proxy: compare before vs after unlearning.
    """
    model.eval()
    enc = tokenizer(canary_text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], labels=enc["input_ids"])
    return out.loss.item()

def measure_canary_suite(model, tokenizer, canaries):
    losses = [canary_loss(model, tokenizer, c) for c in canaries]
    return {
        "mean_loss": float(np.mean(losses)),
        "min_loss": float(np.min(losses)),
        "per_canary": losses,
    }

def load_lora_into_model(base_model, lora_keys, param_arrays, rank=8):
    """Wrap a fresh base model with LoRA and load specific weights (e.g. an
    unlearned aggregate from baselines.py) for evaluation."""
    model = wrap_with_lora(base_model, rank=rank)
    from collections import OrderedDict
    sd = OrderedDict({k: torch.tensor(a) for k, a in zip(lora_keys, param_arrays)})
    set_lora_state_dict(model, sd)
    return model

def compare_before_after(round_ckpt, target_client_id, target_canaries,
                          before_params, after_params_dict, tokenizer):
    lora_keys = round_ckpt["lora_keys"]
    results = {}

    base_model, _ = load_base_model()
    model_before = load_lora_into_model(base_model, lora_keys, before_params)
    results["before_unlearning"] = measure_canary_suite(model_before, tokenizer, target_canaries)

    for variant_name, params in after_params_dict.items():
        base_model, _ = load_base_model()
        model_after = load_lora_into_model(base_model, lora_keys, params)
        results[f"after_{variant_name}"] = measure_canary_suite(model_after, tokenizer, target_canaries)

    return results