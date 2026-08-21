"""
Exports an N-element slice of client 0's committed values from a real
checkpoint into circom's input.json format. Used both for single-slice
testing (Phase 5) and the cost sweep across slice sizes (Phase 6).
"""
import pickle
import json
import numpy as np

from commitment import quantize
from baselines import aggregate_retained

ROUND_PATH = "checkpoints/clients20_alpha0.5_r8/round_5.pkl"
TARGET_CLIENT_ID = 0
LAYER_IDX = 3  # confirmed real signal: 0.self_attn.v_proj.lora_A.default.weight
ETA_INT = 5

def fisher_dict_to_list(fisher_dict, lora_keys):
    return [fisher_dict[k] for k in lora_keys]

def export_slice(slice_size, output_path="../circuits/input.json", layer_idx=LAYER_IDX):
    with open(ROUND_PATH, "rb") as f:
        round_ckpt = pickle.load(f)

    client_ids = round_ckpt["client_ids"]
    lora_keys = round_ckpt["lora_keys"]
    target_idx = client_ids.index(TARGET_CLIENT_ID)

    agg, keep_idx = aggregate_retained(
        round_ckpt["client_deltas"], round_ckpt["client_sizes"], client_ids, TARGET_CLIENT_ID,
    )
    fisher_all = [fisher_dict_to_list(f, lora_keys) for f in round_ckpt["client_fisher"]]
    delta_j = round_ckpt["client_deltas"][target_idx]
    fisher_j = fisher_all[target_idx]

    S_retain_flat = sum(fisher_all[i][layer_idx] for i in keep_idx).flatten()
    S_j_flat = fisher_j[layer_idx].flatten()
    dW_j_flat = delta_j[layer_idx].flatten()
    dW_agg_flat = agg[layer_idx].flatten()

    available = len(S_retain_flat)
    if slice_size > available:
        raise ValueError(f"slice_size={slice_size} exceeds available elements={available} "
                          f"in layer_idx={layer_idx}; pick a smaller size or a larger layer")

    S_retain_flat = S_retain_flat[:slice_size]
    S_j_flat = S_j_flat[:slice_size]
    dW_j_flat = dW_j_flat[:slice_size]
    dW_agg_flat = dW_agg_flat[:slice_size]

    q_S_retain = quantize(S_retain_flat)
    q_S_j = quantize(S_j_flat)
    q_dW_j = quantize(dW_j_flat)
    q_dW_agg = quantize(dW_agg_flat)

    q_dW_out = (q_dW_agg.astype(object) * q_S_retain.astype(object)
                - ETA_INT * q_S_j.astype(object) * q_dW_j.astype(object))

    inputs = {
        "S_retain": [int(x) for x in q_S_retain],
        "S_j": [int(x) for x in q_S_j],
        "dW_j": [int(x) for x in q_dW_j],
        "dW_agg": [int(x) for x in q_dW_agg],
        "ETA": ETA_INT,
        "dW_out": [int(x) for x in q_dW_out],
    }

    with open(output_path, "w") as f:
        json.dump(inputs, f)

    return inputs


if __name__ == "__main__":
    inputs = export_slice(slice_size=64)
    print("Exported 64-element slice to circuits/input.json")
    print(f"Sample S_retain[0:3]: {inputs['S_retain'][:3]}")
    print(f"Sample dW_out[0:3]: {inputs['dW_out'][:3]}")