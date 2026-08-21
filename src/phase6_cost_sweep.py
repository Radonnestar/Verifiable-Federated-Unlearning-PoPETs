"""
Phase 6 -- the decisive cost experiment. Sweeps slice size N and times
every stage of the pipeline: compile, setup, witness gen, prove, verify.
This is Figure 1 of the paper: does proving cost scale with N (adapter
dimension) in a tractable way, independent of base-model size.
"""
import subprocess
import time
import os
import csv

CIRCOM_BIN = r"C:\circom\circom.exe"
CIRCUITS_DIR = os.path.abspath("../circuits")
PTAU_FILE = "pot16_final.ptau"
SLICE_SIZES = [7168]
RESULTS_CSV = os.path.join(CIRCUITS_DIR, "phase6_results.csv")

# lora_A layers (e.g. layer_idx=0) have shape [8,896] = 7168 elements,
# enough headroom for any N in SLICE_SIZES. lora_A layers were shown
# earlier to have near-zero Fisher values -- fine here since constraint
# count and proof size depend only on element COUNT, not values; the
# forgetting-quality experiment (Phase 4) already validated signal
# quality separately using layer_idx=3.
LARGE_N_LAYER = 0
SMALL_N_LAYER = 3
LARGE_N_THRESHOLD = 1024  # layer 3 only has 1024 elements available

def run(cmd, cwd=CIRCUITS_DIR, input_text=None, timeout=300):
    start = time.time()
    result = subprocess.run(
        cmd, cwd=cwd, shell=True, capture_output=True, text=True,
        input=input_text, timeout=timeout,
    )
    elapsed = time.time() - start
    return elapsed, result.stdout + result.stderr, result.returncode

def sweep_one_size(n):
    from export_witness import export_slice
    print(f"\n{'='*60}\nSlice size N={n}\n{'='*60}")

    row = {"N": n}

    t, out, rc = run(f"python gen_circuit.py {n}")
    if rc != 0:
        print("gen_circuit failed:", out); return None

    t, out, rc = run(f'"{CIRCOM_BIN}" correction.circom --r1cs --wasm --sym -o build')
    row["compile_time"] = t
    if rc != 0:
        print("compile failed:", out); return None
    for line in out.splitlines():
        if "non-linear constraints" in line:
            row["constraints"] = int(line.split(":")[1].strip())

    t, out, rc = run(f"snarkjs groth16 setup build/correction.r1cs {PTAU_FILE} zkey_0000.zkey")
    row["setup_time"] = t
    if rc != 0:
        print("setup failed:", out); return None

    t, out, rc = run(
        f"snarkjs zkey contribute zkey_0000.zkey zkey_final.zkey --name=sweep",
        input_text="sweepentropy12345\n",
    )
    row["contribute_time"] = t
    if rc != 0:
        print("contribute failed:", out); return None

    t, out, rc = run("snarkjs zkey export verificationkey zkey_final.zkey vkey.json")
    row["export_vkey_time"] = t

    # pick a layer with enough elements for this N
    layer_idx = SMALL_N_LAYER if n <= LARGE_N_THRESHOLD else LARGE_N_LAYER

    os.chdir("../src")
    export_slice(slice_size=n, output_path="../circuits/input.json", layer_idx=layer_idx)
    os.chdir(CIRCUITS_DIR)

    t, out, rc = run(
        "node build/correction_js/generate_witness.js build/correction_js/correction.wasm input.json witness.wtns"
    )
    row["witness_time"] = t
    if rc != 0:
        print("witness gen failed:", out); return None

    t, out, rc = run("snarkjs groth16 prove zkey_final.zkey witness.wtns proof.json public.json")
    row["prove_time"] = t
    if rc != 0:
        print("prove failed:", out); return None

    t, out, rc = run("snarkjs groth16 verify vkey.json public.json proof.json")
    row["verify_time"] = t
    row["verified"] = "OK!" in out

    proof_path = os.path.join(CIRCUITS_DIR, "proof.json")
    row["proof_size_bytes"] = os.path.getsize(proof_path)
    row["layer_used"] = layer_idx

    print(f"N={n}: constraints={row.get('constraints')}, "
          f"prove={row['prove_time']:.2f}s, verify={row['verify_time']:.3f}s, "
          f"proof_size={row['proof_size_bytes']}B, verified={row['verified']}, "
          f"layer={layer_idx}")

    return row


def load_existing_results():
    """Load prior results if the CSV exists, keyed by N, so reruns merge
    instead of destroying earlier data points."""
    existing = {}
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[int(row["N"])] = row
    return existing


def main():
    existing = load_existing_results()
    print(f"Loaded {len(existing)} existing result(s) from {RESULTS_CSV}" if existing else "No existing results found.")

    new_results = {}
    for n in SLICE_SIZES:
        try:
            row = sweep_one_size(n)
            if row:
                # normalize to strings for consistent CSV merge with loaded rows
                new_results[n] = {k: str(v) for k, v in row.items()}
        except subprocess.TimeoutExpired:
            print(f"N={n} timed out, skipping")
        except Exception as e:
            print(f"N={n} failed with exception: {e}")

    # merge: new results overwrite old ones for the same N, everything else preserved
    merged = {**existing, **new_results}

    if not merged:
        print("No results at all.")
        return

    all_keys = set()
    for row in merged.values():
        all_keys.update(row.keys())
    fieldnames = ["N"] + sorted(k for k in all_keys if k != "N")

    sorted_rows = [merged[n] for n in sorted(merged.keys(), key=lambda x: int(x))]

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_rows)

    print(f"\n{'='*60}\nFULL RESULTS ({len(sorted_rows)} points) -> {RESULTS_CSV}\n{'='*60}")
    print(f"{'N':>6} {'constraints':>12} {'prove_s':>10} {'verify_s':>10} {'proof_B':>10}")
    for r in sorted_rows:
        print(f"{r['N']:>6} {r.get('constraints','?'):>12} {float(r['prove_time']):>10.3f} "
              f"{float(r['verify_time']):>10.3f} {r['proof_size_bytes']:>10}")


if __name__ == "__main__":
    main()