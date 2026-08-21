"""
Federated LoRA training driver.
Deliberately NOT using flwr.simulation.start_simulation — that API
aggregates internally and discards per-client ΔW_i / S_i, which Phase 3
(commitment layer) needs access to. Instead we drive clients manually
round-by-round, which is still 'using Flower' (NumPyClient interface)
but gives us the raw per-client artifacts SPEC.md requires.
"""
import os
import pickle
import numpy as np
import torch
from collections import OrderedDict

from model_utils import load_base_model, wrap_with_lora, get_lora_state_dict
from data_utils import get_client_texts
from fl_client import LoRAClient, state_dict_to_ndarrays, ndarrays_to_state_dict

NUM_ROUNDS = 5
CHECKPOINT_DIR = "checkpoints"

def fedavg(client_arrays_list, client_sizes):
    total = sum(client_sizes)
    weights = [s / total for s in client_sizes]
    n_layers = len(client_arrays_list[0])
    avg = []
    for layer_idx in range(n_layers):
        stacked = np.stack([client_arrays_list[c][layer_idx] * weights[c]
                             for c in range(len(client_arrays_list))])
        avg.append(stacked.sum(axis=0))
    return avg

def run_federated_training(num_clients=20, alpha=0.5, rank=8, num_rounds=NUM_ROUNDS,
                            run_tag=None):
    run_tag = run_tag or f"clients{num_clients}_alpha{alpha}_r{rank}"
    ckpt_dir = os.path.join(CHECKPOINT_DIR, run_tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"[{run_tag}] loading data + partitioning across {num_clients} clients (alpha={alpha})")
    client_texts = get_client_texts(num_clients=num_clients, alpha=alpha)

    # global model init
    global_model, tokenizer = load_base_model()
    global_model = wrap_with_lora(global_model, rank=rank)
    global_sd = get_lora_state_dict(global_model)
    lora_keys = list(global_sd.keys())
    global_params = state_dict_to_ndarrays(global_sd)

    clients = [
        LoRAClient(client_id=i, texts=client_texts[i], tokenizer=tokenizer,
                   shared_model=global_model, rank=rank)
        for i in range(num_clients)
    ]

    for round_num in range(1, num_rounds + 1):
        print(f"[{run_tag}] round {round_num}/{num_rounds}")

        round_arrays = []      # ΔW_i per client, this round
        round_fisher = []      # S_i per client, this round
        round_sizes = []
        round_client_ids = []

        for client in clients:
            new_params, n_samples, meta = client.fit(global_params, config={})
            fisher = client.get_fisher()

            round_arrays.append(new_params)
            round_fisher.append(fisher)
            round_sizes.append(n_samples)
            round_client_ids.append(meta["client_id"])

        # FedAvg aggregation -> new global params
        global_params = fedavg(round_arrays, round_sizes)

        # checkpoint everything Phase 3 needs: per-client ΔW_i, S_i, and the
        # new global aggregate, all keyed by round
        round_ckpt = {
            "round": round_num,
            "lora_keys": lora_keys,
            "client_ids": round_client_ids,
            "client_deltas": round_arrays,     # list of ndarray-lists, per client
            "client_fisher": round_fisher,     # list of dicts, per client
            "client_sizes": round_sizes,
            "global_params": global_params,
        }
        with open(os.path.join(ckpt_dir, f"round_{round_num}.pkl"), "wb") as f:
            pickle.dump(round_ckpt, f)

        # quick eval on a couple of clients as a sanity signal
        eval_loss, _, _ = clients[0].evaluate(global_params, config={})
        print(f"[{run_tag}] round {round_num} eval loss (client 0): {eval_loss:.4f}")

    print(f"[{run_tag}] done. checkpoints in {ckpt_dir}")
    return ckpt_dir


if __name__ == "__main__":
    # primary config
    run_federated_training(num_clients=20, alpha=0.5, rank=8)

    # secondary config — comment out if short on time, per SPEC.md
    # run_federated_training(num_clients=100, alpha=0.5, rank=8)