"""
Generates paper-ready figures from Phase 6 (cost sweep) and Phase 4/7
(forgetting quality, LHH comparison) results. Saves as both PNG (for
quick viewing) and PDF (vector, for LaTeX inclusion).
"""
import csv
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

OUT_DIR = "../figures"
import os
os.makedirs(OUT_DIR, exist_ok=True)


def fig1_cost_scaling():
    """Prove time, verify time, and proof size vs slice size N."""
    rows = []
    with open("../circuits/phase6_results.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    rows.sort(key=lambda r: int(r["N"]))

    N = [int(r["N"]) for r in rows]
    constraints = [int(r["constraints"]) for r in rows]
    prove = [float(r["prove_time"]) for r in rows]
    verify = [float(r["verify_time"]) for r in rows]
    proof_size = [int(r["proof_size_bytes"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(N, prove, marker='o', label='Prove time', color='#2166ac', linewidth=2)
    ax.plot(N, verify, marker='s', label='Verify time', color='#b2182b', linewidth=2)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Slice size N (elements)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('(a) Proving/verification cost vs. adapter slice size')
    ax.legend()
    ax.axvline(x=7168, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.annotate('full LoRA\nmatrix (N=7168)', xy=(7168, prove[-1]), xytext=(1800, prove[-1]+0.4),
                fontsize=8, ha='center', arrowprops=dict(arrowstyle='->', alpha=0.6))

    ax2 = axes[1]
    ax2.plot(N, proof_size, marker='D', color='#4d9221', linewidth=2)
    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Slice size N (elements)')
    ax2.set_ylabel('Proof size (bytes)')
    ax2.set_title('(b) Proof size vs. adapter slice size')
    ax2.set_ylim(0, 1000)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig1_cost_scaling.pdf")
    plt.savefig(f"{OUT_DIR}/fig1_cost_scaling.png", dpi=200)
    print(f"Saved fig1_cost_scaling.pdf / .png  (N range: {min(N)}-{max(N)}, "
          f"prove: {min(prove):.2f}s-{max(prove):.2f}s)")
    plt.close()


def fig2_forgetting_vs_eta():
    with open("checkpoints/canary_experiment_result.pkl", "rb") as f:
        data = pickle.load(f)
    results = data["results"]

    eta_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    circuit_losses = [results[f"circuit_eta{e}"]["mean_loss"] for e in eta_values]

    unconstrained_eta = [1.0, 5.0]
    unconstrained_losses = [results[f"unconstrained_eta{e}"]["mean_loss"] for e in unconstrained_eta]

    naive_loss = results["after_naive"]["mean_loss"]
    before_loss = results["before_unlearning"]["mean_loss"]
    solo_loss = results["client0_solo"]["mean_loss"]

    fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left panel: full range, showing the eta=10 regime change
    ax_main.plot(eta_values, circuit_losses, marker='o', label='Circuit-representable (ours)',
                 color='#2166ac', linewidth=2, markersize=7)
    ax_main.plot(unconstrained_eta, unconstrained_losses, marker='^', label='Unconstrained (cross-term)',
                 color='#b2182b', linewidth=2, markersize=7, linestyle='--')
    ax_main.axhline(y=naive_loss, color='#4d9221', linestyle=':', linewidth=2, label='Naive subtraction')
    ax_main.axhline(y=before_loss, color='gray', linestyle='-', linewidth=1.2, alpha=0.7, label='Before unlearning')
    ax_main.axhline(y=solo_loss, color='black', linestyle='-.', linewidth=1, alpha=0.5, label="Client's solo contribution")
    ax_main.annotate('regime change\n(overshoot)', xy=(10, circuit_losses[-1]), xytext=(6.5, 8.3),
                      fontsize=8, ha='center', arrowprops=dict(arrowstyle='->', alpha=0.6))
    ax_main.set_xlabel(r'Correction strength $\eta$')
    ax_main.set_ylabel('Mean canary loss')
    ax_main.set_title('(a) Full range, showing overshoot at high $\\eta$')
    ax_main.legend(fontsize=8, loc='upper left')

    # Right panel: zoomed to eta in [0, 5], where the real comparison lives
    zoom_idx = [i for i, e in enumerate(eta_values) if e <= 5.0]
    eta_zoom = [eta_values[i] for i in zoom_idx]
    circuit_zoom = [circuit_losses[i] for i in zoom_idx]

    ax_zoom.plot(eta_zoom, circuit_zoom, marker='o', label='Circuit-representable (ours)',
                 color='#2166ac', linewidth=2, markersize=8)
    ax_zoom.plot(unconstrained_eta, unconstrained_losses, marker='^', label='Unconstrained (cross-term)',
                 color='#b2182b', linewidth=2, markersize=8, linestyle='--')
    ax_zoom.axhline(y=naive_loss, color='#4d9221', linestyle=':', linewidth=2, label='Naive subtraction')
    ax_zoom.axhline(y=before_loss, color='gray', linestyle='-', linewidth=1.2, alpha=0.7, label='Before unlearning')
    ax_zoom.set_xlabel(r'Correction strength $\eta$')
    ax_zoom.set_ylabel('Mean canary loss')
    ax_zoom.set_title(r'(b) Zoomed: $\eta \leq 5$, the tested operating range')
    ax_zoom.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig2_forgetting_vs_eta.pdf")
    plt.savefig(f"{OUT_DIR}/fig2_forgetting_vs_eta.png", dpi=200)
    print(f"Saved fig2_forgetting_vs_eta.pdf / .png")
    plt.close()


def fig3_lhh_vs_snark():
    """Bar chart: cost comparison, LHH vs SNARK, log scale.
    SNARK values pulled from the actual N=64 row of phase6_results.csv,
    not estimated -- exact measured figures for the paper."""
    import csv
    with open("../circuits/phase6_results.csv") as f:
        reader = csv.DictReader(f)
        n64_row = next(r for r in reader if int(r["N"]) == 64)

    snark_prove_ms = float(n64_row["prove_time"]) * 1000
    snark_verify_ms = float(n64_row["verify_time"]) * 1000

    labels = ['LHH hash gen\n(20 clients)', 'LHH verify', 'SNARK prove\n(N=64)', 'SNARK verify\n(N=64)']
    times_ms = [0.46, 0.014, snark_prove_ms, snark_verify_ms]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = ['#4d9221', '#4d9221', '#2166ac', '#2166ac']
    bars = ax.bar(labels, times_ms, color=colors, alpha=0.85)
    ax.set_yscale('log')
    ax.set_ylabel('Time (milliseconds, log scale)')
    ax.set_title('Cost: LHH (VerFU-style) vs. our ZK-SNARK approach')

    for bar, val in zip(bars, times_ms):
        ax.annotate(f'{val:.2f} ms' if val < 10 else f'{val:.0f} ms',
                    xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig3_lhh_vs_snark.pdf")
    plt.savefig(f"{OUT_DIR}/fig3_lhh_vs_snark.png", dpi=200)
    print(f"Saved fig3_lhh_vs_snark.pdf / .png  (SNARK values from actual N=64 measurement: "
          f"prove={snark_prove_ms:.1f}ms, verify={snark_verify_ms:.1f}ms)")
    plt.close()


if __name__ == "__main__":
    fig1_cost_scaling()
    fig2_forgetting_vs_eta()
    fig3_lhh_vs_snark()
    print(f"\nAll figures saved to {OUT_DIR}/")