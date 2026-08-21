"""
Three unlearning variants over a checkpointed federated round,
per SPEC.md's Correct() operator and the baseline comparison plan.
"""
import pickle
import numpy as np

EPSILON = 1e-6  # SPEC.md: fixed public constant, not committed
ETA = 5.0       # correction step size (operating point chosen via sweep)

def load_round(ckpt_path):
    with open(ckpt_path, "rb") as f:
        return pickle.load(f)

def fisher_dict_to_list(fisher_dict, lora_keys):
    """client_fisher entries are dicts (from get_fisher()); convert to
    the same ordered-list-by-lora_keys format client_deltas already use."""
    return [fisher_dict[k] for k in lora_keys]

def aggregate_retained(client_deltas, client_sizes, client_ids, exclude_id):
    """Weighted FedAvg over all clients except exclude_id."""
    keep_idx = [i for i, cid in enumerate(client_ids) if cid != exclude_id]
    retained_sizes = [client_sizes[i] for i in keep_idx]
    total = sum(retained_sizes)
    weights = [s / total for s in retained_sizes]

    n_layers = len(client_deltas[0])
    agg = []
    for layer in range(n_layers):
        stacked = np.stack([client_deltas[i][layer] * weights[k]
                             for k, i in enumerate(keep_idx)])
        agg.append(stacked.sum(axis=0))
    return agg, keep_idx

def naive_subtraction(round_ckpt, target_client_id):
    """Baseline 1: just drop the target client from the average, no correction term."""
    agg, _ = aggregate_retained(
        round_ckpt["client_deltas"], round_ckpt["client_sizes"],
        round_ckpt["client_ids"], target_client_id,
    )
    return agg

def unconstrained_correction(round_ckpt, target_client_id):
    """
    Baseline 2 (upper bound on forgetting quality): real cross-parameter
    correction via a small per-column matrix inverse -- NOT restricted to
    being circuit-provable. This is the ceiling our circuit-representable
    version is measured against; the gap between the two is the
    "provability tax" (N4).
    """
    return _fisher_correction(round_ckpt, target_client_id, elementwise=False)

def circuit_representable_correction(round_ckpt, target_client_id):
    """
    Our method: elementwise (Hadamard) correction per SPEC.md's Correct().
    This is the version whose arithmetic is degree-2 and provable.
    """
    return _fisher_correction(round_ckpt, target_client_id, elementwise=True)

def _fisher_correction(round_ckpt, target_client_id, elementwise: bool):
    """
    elementwise=True  -> circuit-representable: diagonal-only Fisher scaling,
        matches Correct() exactly as specified (degree-2, provable).
    elementwise=False -> unconstrained upper bound: uses a small per-column
        rank x rank curvature matrix (diagonal Fisher + empirical outer-product
        cross term from dW_j), inverted directly. This genuinely incorporates
        cross-parameter correlation the circuit version structurally cannot
        represent (matrix inversion is not degree-2 arithmetic), so it is a
        legitimate, different upper bound rather than a copy of the elementwise
        formula. Reshapes each layer to its natural (rank, features) or
        (features, rank) LoRA shape to define "columns" along the rank axis.
    """
    client_ids = round_ckpt["client_ids"]
    lora_keys = round_ckpt["lora_keys"]
    target_idx = client_ids.index(target_client_id)

    agg, keep_idx = aggregate_retained(
        round_ckpt["client_deltas"], round_ckpt["client_sizes"], client_ids, target_client_id,
    )

    fisher_retain = [fisher_dict_to_list(f, lora_keys) for f in round_ckpt["client_fisher"]]
    delta_j = round_ckpt["client_deltas"][target_idx]
    fisher_j = fisher_dict_to_list(round_ckpt["client_fisher"][target_idx], lora_keys)

    n_layers = len(agg)
    corrected = []

    for layer in range(n_layers):
        S_retain_layer = sum(fisher_retain[i][layer] for i in keep_idx)
        S_j_layer = fisher_j[layer]
        dW_j_layer = delta_j[layer]
        agg_layer = agg[layer]

        if elementwise:
            # SPEC.md Correct(): (S_retain + eps)^-1 (Hadamard) S_j (Hadamard) dW_j
            correction = (1.0 / (S_retain_layer + EPSILON)) * S_j_layer * dW_j_layer
            corrected.append(agg_layer + ETA * correction)
            continue

        # --- unconstrained: real cross-term correction via small matrix inverse ---
        shape = agg_layer.shape
        if len(shape) != 2 or min(shape) > 32:
            # fallback for any layer shape too large/odd for a dense per-column
            # inverse to be cheap (shouldn't trigger for standard LoRA A/B mats)
            correction = (1.0 / (S_retain_layer + EPSILON)) * S_j_layer * dW_j_layer
            corrected.append(agg_layer + ETA * correction)
            continue

        # orient so axis 0 is the small "rank" dimension we invert over
        transpose = shape[0] > shape[1]
        S_r = S_retain_layer.T if transpose else S_retain_layer
        S_j = S_j_layer.T if transpose else S_j_layer
        dW_j = dW_j_layer.T if transpose else dW_j_layer
        agg_l = agg_layer.T if transpose else agg_layer

        r, cols = S_r.shape
        out = np.zeros_like(agg_l)

        for c in range(cols):
            s_r_col = S_r[:, c]
            s_j_col = S_j[:, c]
            dw_col = dW_j[:, c]

            # small r x r curvature matrix: diagonal Fisher + empirical
            # outer-product cross term (K-FAC-style Gauss-Newton approximation)
            G = np.diag(s_r_col + EPSILON) + np.outer(dw_col, dw_col)
            correction_col = np.linalg.solve(G, s_j_col * dw_col)

            out[:, c] = agg_l[:, c] + ETA * correction_col

        corrected.append(out.T if transpose else out)

    return corrected


def compare_variants(ckpt_path, target_client_id):
    round_ckpt = load_round(ckpt_path)
    results = {
        "naive": naive_subtraction(round_ckpt, target_client_id),
        "unconstrained": unconstrained_correction(round_ckpt, target_client_id),
        "circuit": circuit_representable_correction(round_ckpt, target_client_id),
    }
    diff_naive_vs_circuit = np.linalg.norm(results["naive"][0] - results["circuit"][0])
    diff_unconstrained_vs_circuit = np.linalg.norm(results["unconstrained"][0] - results["circuit"][0])
    print(f"Target client: {target_client_id}")
    print(f"||naive - circuit|| (layer 0): {diff_naive_vs_circuit:.6f}")
    print(f"||unconstrained - circuit|| (layer 0): {diff_unconstrained_vs_circuit:.6f}")
    return results


if __name__ == "__main__":
    import sys
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/clients20_alpha0.5_r8/round_5.pkl"
    compare_variants(ckpt_path, target_client_id=0)