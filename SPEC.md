# SPEC — Verifiable Federated Unlearning (frozen, do not revisit mid-week)

## Fixed decisions

- **Model:** Qwen2.5-0.5B (HF + PEFT tooling is easiest of the candidates)
- **Adapter:** LoRA, rank r ∈ {4, 8} — sweep for Phase 6, r=8 default elsewhere
- **Quantization:** 16-bit fixed point, scale factor 2^12, deterministic (no stochastic rounding)
- **Commitment:** Pedersen, over the flattened + quantized ΔW_i vector
- **FL framework:** Flower, Dirichlet non-IID split, α ∈ {0.1, 0.5}
- **Client counts:** 20 (primary), 100 (secondary, if time permits)

## `Correct()` — the unlearning operator

Each client i, during local training, additionally accumulates a diagonal
empirical Fisher estimate over its own adapter parameters:

    S_i = E_{x~D_i} [ (∂L/∂ΔW_i)^2 ]     (elementwise square of local grad, averaged)

S_i is quantized and committed alongside ΔW_i. At unlearning time, given
target client j and retained set R:

    S_retain = Σ_{i∈R} S_i
    Correct(S_retain, S_j, ΔW_j) = (S_retain + ε)^-1 ⊙ S_j ⊙ ΔW_j     (elementwise)

    ΔW_global' = Aggregate({ΔW_i}_{i∈R}) − η · Correct(S_retain, S_j, ΔW_j)

This is deliberately elementwise (Hadamard), not a matrix inverse — a full
inverse-Hessian correction is not low-degree and is not provable at
reasonable cost. Diagonal/elementwise keeps the relation degree-2 in
committed values, which is what Phase 5's circuit depends on.

ε is a small constant to avoid division by zero, fixed at commit time as a
public parameter (not committed, not secret).

## Protocol syntax

- `Setup(λ) → pp`
- `Commit(ΔW_i, S_i, r_i) → C_i`
- `Aggregate({C_i}_{i∈R}) → C_global`
- `Unlearn(C_global, C_j, {S_i}_{i∈R}) → ΔW_global'`
- `Prove(witness) → π`
- `Verify(π, C_global, C_j, ΔW_global') → {0,1}`

## Exclusion soundness (informal game)

- Challenger runs `Setup(λ) → pp`, gives pp to adversary A (playing a
  malicious server).
- A outputs `(π, C_global, C_j, ΔW_global')`.
- A wins if `Verify(π, C_global, C_j, ΔW_global') = 1` AND ΔW_global'
  still reflects client j's committed contribution (i.e. j ∉ excluded
  set despite the proof claiming exclusion).
- **Definition:** the protocol has exclusion soundness if no PPT
  adversary A wins except with negligible probability in λ.
- **Argument (sketch, not a full reduction):** winning requires either
  breaking the binding property of the Pedersen commitment (producing
  a valid opening to two different values) or breaking soundness of the
  underlying proof system. Both are assumed hard. Full reduction is
  future work — this sketch is what goes in the paper for week 1.

## Provenance anchoring

Per-round hash chain, published alongside the global model:

    H_t = Hash(H_{t-1} || {C_i}_{i in round t} || C_global,t)

Unlearning proof takes H_T (latest published root) as a public input,
binding the server to a previously-published history rather than an
invented one. Does not prove honest training — only that the server
can't retroactively fabricate the starting state.

## Explicitly out of scope this week

- Deployment binding (proof that the *served* model matches the proven one)
- Full forging-attack replication (Thudi et al. / ICML'24 fragility attacks)
- Duplication study across clients
- Retrain-from-scratch gold standard (include only if Phase 7 has time)