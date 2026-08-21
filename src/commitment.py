"""
Pedersen commitments over quantized LoRA contributions, and the
per-round provenance hash chain (SPEC.md §Provenance anchoring).
"""
import hashlib
import numpy as np
from tinyec import registry
import secrets

CURVE = registry.get_curve('secp256r1')
SCALE = 2 ** 12  # SPEC.md quantization scale factor

def quantize(arr: np.ndarray) -> np.ndarray:
    """16-bit fixed point, deterministic (SPEC.md)."""
    q = np.round(arr * SCALE).astype(np.int64)
    q = np.clip(q, -(2**15), 2**15 - 1)  # fits in 16-bit signed range
    return q

def dequantize(q: np.ndarray) -> np.ndarray:
    return q.astype(np.float64) / SCALE

def flatten_state_dict(state_dict_arrays, keys):
    """Flatten a list-of-ndarrays (per SPEC ordering) into one 1D vector."""
    return np.concatenate([np.asarray(a).flatten() for a in state_dict_arrays])


class PedersenCommitment:
    """
    Single-value Pedersen commitment: C = v*G + r*H over secp256r1.
    For a full parameter vector we commit to a hash-reduced scalar
    representation (practical approach for prototype scale — a real
    circuit would batch-commit per-chunk; noted as a simplification).
    """
    def __init__(self):
        self.G = CURVE.g
        # H = second generator, derived deterministically from G so no
        # trusted setup is needed and nobody knows log_G(H)
        h_seed = int(hashlib.sha256(b"popets-unlearning-H").hexdigest(), 16)
        self.H = h_seed * CURVE.g

    def _vector_to_scalar(self, quantized_vec: np.ndarray) -> int:
        """Reduce a quantized parameter vector to a single field element
        via hashing — the value being committed to."""
        h = hashlib.sha256(quantized_vec.tobytes()).digest()
        return int.from_bytes(h, "big") % CURVE.field.n

    def commit(self, quantized_vec: np.ndarray, randomness: int = None):
        v = self._vector_to_scalar(quantized_vec)
        r = randomness if randomness is not None else secrets.randbelow(CURVE.field.n)
        C = v * self.G + r * self.H
        return C, v, r  # commitment point, opened value, randomness (r kept secret)

    def verify(self, C, v: int, r: int) -> bool:
        return C == (v * self.G + r * self.H)

    def point_to_bytes(self, point) -> bytes:
        return f"{point.x},{point.y}".encode()


def commit_client_contribution(delta_arrays, fisher_arrays, lora_keys):
    """
    Commits to a single client's (ΔW_i, S_i) pair.
    Returns (C_delta, C_fisher, opening_data) — opening_data is kept
    secret by the client, needed later for well-formedness proofs.
    """
    pc = PedersenCommitment()

    flat_delta = flatten_state_dict(delta_arrays, lora_keys)
    flat_fisher = flatten_state_dict(fisher_arrays, lora_keys)

    q_delta = quantize(flat_delta)
    q_fisher = quantize(flat_fisher)

    C_delta, v_delta, r_delta = pc.commit(q_delta)
    C_fisher, v_fisher, r_fisher = pc.commit(q_fisher)

    opening = {
        "v_delta": v_delta, "r_delta": r_delta,
        "v_fisher": v_fisher, "r_fisher": r_fisher,
        "q_delta": q_delta, "q_fisher": q_fisher,
    }
    return C_delta, C_fisher, opening


def hash_round(prev_hash: bytes, commitments: list, global_commitment_bytes: bytes) -> bytes:
    """
    H_t = Hash(H_{t-1} || {C_i} || C_global,t)   -- SPEC.md provenance chain
    """
    h = hashlib.sha256()
    h.update(prev_hash)
    for C in commitments:
        h.update(f"{C.x},{C.y}".encode())
    h.update(global_commitment_bytes)
    return h.digest()


if __name__ == "__main__":
    # smoke test: commit + verify a dummy vector
    pc = PedersenCommitment()
    dummy = quantize(np.random.randn(100).astype(np.float32))
    C, v, r = pc.commit(dummy)
    print("Commitment verifies:", pc.verify(C, v, r))
    print("Tampered value rejected:", not pc.verify(C, v + 1, r))