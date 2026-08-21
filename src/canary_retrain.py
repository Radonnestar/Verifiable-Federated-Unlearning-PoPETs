"""
Produces a real before/after canary-forgetting result without rerunning
the full 5-round, 20-client federated training. Client 0 is retrained
from the round-5 global state with canaries injected into its data,
then swapped into the round-5 aggregate in place of its original
contribution. This is equivalent to "client 0's final contribution
included canary memorization" -- which is all the experiment needs.
"""
import pickle
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict

from model_utils import load_base_model, wrap_with_lora, get_lora_state_dict, set_lora_state_dict
from data_utils import TextDataset, get_client_texts
from fl_client import state_dict_to_ndarrays, ndarrays_to_state_dict
from canary import generate_canaries, inject_canaries, measure_canary_suite, load_lora_into_model
import baselines as baselines_module
from baselines import (
    aggregate_retained, EPSILON,
    naive_subtraction, unconstrained_correction, circuit_representable_correction,
)

TARGET_CLIENT_ID = 0
CKPT_PATH = "checkpoints/clients20_alpha0.5_r8/round_5.pkl"
LOCAL_EPOCHS = 8
LR = 1e-4
BATCH_SIZE = 4
RANK = 8


def free_model(*objs):
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def retrain_client0_with_canaries(round_ckpt, tokenizer):
    lora_keys = round_ckpt["lora_keys"]

    all_client_texts = get_client_texts(num_clients=20, alpha=0.5)
    client0_texts = all_client_texts[TARGET_CLIENT_ID]

    canaries = generate_canaries(client_id=TARGET_CLIENT_ID)
    texts_with_canaries = inject_canaries(client0_texts, TARGET_CLIENT_ID, canaries)

    dataset = TextDataset(texts_with_canaries, tokenizer)

    base_model, _ = load_base_model()
    model = wrap_with_lora(base_model, rank=RANK)

    global_sd = OrderedDict({k: torch.tensor(a) for k, a in zip(lora_keys, round_ckpt["global_params"])})
    set_lora_state_dict(model, global_sd)

    model.train()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    fisher_accum = {k: torch.zeros_like(v) for k, v in get_lora_state_dict(model).items()}
    n_batches = 0

    for epoch in range(LOCAL_EPOCHS):
        epoch_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                         labels=batch["labels"])
            out.loss.backward()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in fisher_accum and param.grad is not None:
                        fisher_accum[name] += param.grad.detach() ** 2
            n_batches += 1
            optimizer.step()
            epoch_loss += out.loss.item()
        print(f"  canary-retrain epoch {epoch+1}/{LOCAL_EPOCHS}, avg loss: {epoch_loss/len(loader):.4f}", flush=True)

    fisher = {k: (v / max(n_batches, 1)).detach().cpu().numpy() for k, v in fisher_accum.items()}
    new_sd = get_lora_state_dict(model)
    new_delta = state_dict_to_ndarrays(new_sd)

    free_model(model, base_model)
    return new_delta, fisher, canaries


def rebuild_round_with_canary_client(round_ckpt, new_delta_c0, new_fisher_c0):
    updated = dict(round_ckpt)
    idx0 = round_ckpt["client_ids"].index(TARGET_CLIENT_ID)

    client_deltas = list(round_ckpt["client_deltas"])
    client_fisher = list(round_ckpt["client_fisher"])
    client_deltas[idx0] = new_delta_c0
    client_fisher[idx0] = new_fisher_c0

    updated["client_deltas"] = client_deltas
    updated["client_fisher"] = client_fisher

    sizes = round_ckpt["client_sizes"]
    total = sum(sizes)
    weights = [s / total for s in sizes]
    n_layers = len(client_deltas[0])
    new_global = []
    for layer in range(n_layers):
        stacked = np.stack([client_deltas[c][layer] * weights[c] for c in range(len(client_deltas))])
        new_global.append(stacked.sum(axis=0))
    updated["global_params"] = new_global

    return updated


def measure_one_condition(label, lora_keys, params, canaries, tokenizer):
    try:
        base_model, _ = load_base_model()
        model = load_lora_into_model(base_model, lora_keys, params, rank=RANK)
        result = measure_canary_suite(model, tokenizer, canaries)
        print(f"  {label:20s} mean canary loss: {result['mean_loss']:.4f}", flush=True)
        free_model(model, base_model)
        return result
    except Exception as e:
        print(f"  FAILED on '{label}': {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None


def run_full_canary_experiment():
    with open(CKPT_PATH, "rb") as f:
        round_ckpt = pickle.load(f)

    _, tokenizer = load_base_model()

    print("Step 1: retraining client 0 with canaries injected...", flush=True)
    new_delta_c0, new_fisher_c0, canaries = retrain_client0_with_canaries(round_ckpt, tokenizer)

    print("\nStep 2: rebuilding round with canary-holding client 0...", flush=True)
    canary_round = rebuild_round_with_canary_client(round_ckpt, new_delta_c0, new_fisher_c0)

    lora_keys = canary_round["lora_keys"]

    print("\nStep 2b: measuring canary loss on client 0's SOLO contribution (pre-aggregation)...", flush=True)
    idx0 = canary_round["client_ids"].index(TARGET_CLIENT_ID)
    solo_params = canary_round["client_deltas"][idx0]
    results_solo = measure_one_condition(
        "client0_solo", lora_keys, solo_params, canaries, tokenizer
    )

    results = {"client0_solo": results_solo}

    print("\nStep 3: measuring canary loss BEFORE unlearning...", flush=True)
    results["before_unlearning"] = measure_one_condition(
        "before_unlearning", lora_keys, canary_round["global_params"], canaries, tokenizer
    )

    print("\nStep 4: sweeping ETA to find the correction-strength operating point...", flush=True)
    eta_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    eta_results = {}

    for eta_val in eta_values:
        baselines_module.ETA = eta_val
        circuit_params = baselines_module.circuit_representable_correction(canary_round, TARGET_CLIENT_ID)
        r = measure_one_condition(f"circuit_eta{eta_val}", lora_keys, circuit_params, canaries, tokenizer)
        eta_results[f"circuit_eta{eta_val}"] = r

    results.update(eta_results)

    naive_params = naive_subtraction(canary_round, TARGET_CLIENT_ID)
    results["after_naive"] = measure_one_condition("after_naive", lora_keys, naive_params, canaries, tokenizer)

    print("\nStep 4b: computing REAL unconstrained baseline at multiple ETA values...", flush=True)
    for eta_val in [1.0, 5.0]:
        baselines_module.ETA = eta_val
        unconstrained_params = baselines_module.unconstrained_correction(canary_round, TARGET_CLIENT_ID)
        results[f"unconstrained_eta{eta_val}"] = measure_one_condition(
            f"unconstrained_eta{eta_val}", lora_keys, unconstrained_params, canaries, tokenizer
        )

    print("\n=== FORGETTING RESULT ===")
    print(f"{'condition':20s} {'mean_loss':>10s}")
    for k, v in results.items():
        if v is not None:
            print(f"{k:20s} {v['mean_loss']:>10.4f}")
        else:
            print(f"{k:20s} {'FAILED':>10s}")

    out_path = f"checkpoints/canary_experiment_result_client{TARGET_CLIENT_ID}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"results": results, "canaries": canaries, "target_client": TARGET_CLIENT_ID}, f)
    print(f"\nSaved to {out_path}")

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        globals()["TARGET_CLIENT_ID"] = int(sys.argv[1])
    run_full_canary_experiment()