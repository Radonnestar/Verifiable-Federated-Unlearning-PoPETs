## Results

### Forgetting quality (canary extraction loss)

| Condition | Mean canary loss |
|---|---|
| client0 solo | 4.7553 |
| before unlearning | 5.4786 |
| circuit eta0.5 | 5.5085 |
| circuit eta1.0 | 5.5219 |
| circuit eta2.0 | 5.5510 |
| circuit eta5.0 | 5.7106 |
| circuit eta10.0 | 8.5843 |
| after naive | 5.4955 |

### Cryptographic cost (Groth16, real witness data)

| N | Constraints | Prove (s) | Verify (s) | Proof size (B) |
|---|---|---|---|---|
| 16 | 48 | 0.577 | 0.551 | 805 |
| 32 | 96 | 0.617 | 0.544 | 806 |
| 64 | 192 | 0.654 | 0.568 | 805 |
| 128 | 384 | 0.634 | 0.556 | 806 |
| 256 | 768 | 0.631 | 0.534 | 807 |
| 512 | 1536 | 0.728 | 0.545 | 802 |
| 1024 | 3072 | 0.844 | 0.561 | 804 |
| 2048 | 6144 | 1.057 | 0.576 | 806 |

**Key findings:**

- Proof size and verification time remain effectively constant (~800B, ~0.55s) across a 128x range in slice size (N=16 to N=2048).
- Proving time grows sublinearly with constraint count (~0.58s to ~1.06s, a 1.8x increase over a 128x range in N).
- The naive-subtraction baseline shows negligible forgetting; the Fisher-weighted circuit-representable correction shows controllable, tunable forgetting via the ETA parameter.
- Both adversarial checks (tampered commitment, inconsistent exclusion claim) were rejected -- the exclusion claim was rejected at witness generation, before any proof could be attempted.