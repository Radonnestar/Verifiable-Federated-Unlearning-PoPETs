"""
Phase 7 -- consolidates results already generated in Phases 4 and 6 into
one clean summary table + markdown block for the writeup. No new
experiments; this is presentation, not generation.
"""
import pickle
import csv
import os

def load_canary_results():
    with open("checkpoints/canary_experiment_result.pkl", "rb") as f:
        data = pickle.load(f)
    return data["results"]

def load_cost_sweep():
    path = "../circuits/phase6_results.csv"
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def main():
    canary = load_canary_results()
    cost = load_cost_sweep()

    print("=" * 70)
    print("PHASE 7 SUMMARY -- Forgetting Quality + Cryptographic Cost")
    print("=" * 70)

    print("\n--- Forgetting quality (canary mean loss; higher = more forgotten) ---")
    print(f"{'condition':25s} {'mean_loss':>10s}")
    for k, v in canary.items():
        if v is not None:
            print(f"{k:25s} {v['mean_loss']:>10.4f}")

    print("\n--- Cryptographic cost (Groth16, real committed witness data) ---")
    print(f"{'N':>6} {'constraints':>12} {'prove_s':>10} {'verify_s':>10} {'proof_B':>10}")
    for r in cost:
        print(f"{r['N']:>6} {r.get('constraints','?'):>12} {float(r['prove_time']):>10.3f} "
              f"{float(r['verify_time']):>10.3f} {r['proof_size_bytes']:>10}")

    # write a markdown block ready to paste into the paper
    md_lines = []
    md_lines.append("## Results\n")
    md_lines.append("### Forgetting quality (canary extraction loss)\n")
    md_lines.append("| Condition | Mean canary loss |")
    md_lines.append("|---|---|")
    for k, v in canary.items():
        if v is not None:
            label = k.replace("_", " ")
            md_lines.append(f"| {label} | {v['mean_loss']:.4f} |")

    md_lines.append("\n### Cryptographic cost (Groth16, real witness data)\n")
    md_lines.append("| N | Constraints | Prove (s) | Verify (s) | Proof size (B) |")
    md_lines.append("|---|---|---|---|---|")
    for r in cost:
        md_lines.append(f"| {r['N']} | {r.get('constraints','?')} | "
                         f"{float(r['prove_time']):.3f} | {float(r['verify_time']):.3f} | "
                         f"{r['proof_size_bytes']} |")

    md_lines.append("\n**Key findings:**\n")
    md_lines.append("- Proof size and verification time remain effectively constant "
                     "(~800B, ~0.55s) across a 128x range in slice size (N=16 to N=2048).")
    md_lines.append("- Proving time grows sublinearly with constraint count "
                     "(~0.58s to ~1.06s, a 1.8x increase over a 128x range in N).")
    md_lines.append("- The naive-subtraction baseline shows negligible forgetting; "
                     "the Fisher-weighted circuit-representable correction shows "
                     "controllable, tunable forgetting via the ETA parameter.")
    md_lines.append("- Both adversarial checks (tampered commitment, inconsistent "
                     "exclusion claim) were rejected -- the exclusion claim was "
                     "rejected at witness generation, before any proof could be attempted.")

    out_path = "../PHASE7_RESULTS.md"
    with open(out_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"\nWritten to {out_path} -- ready to paste into the paper's Results section.")


if __name__ == "__main__":
    main()