"""
Phase 8 -- adversarial sanity checks.
1. Malformed/tampered commitment opening is rejected (Pedersen binding).
2. A proof built with an inconsistent exclusion claim fails -- either at
   witness generation (the constraint is unsatisfiable) or at proof
   verification. Self-contained: recompiles its own circuit/keys for a
   fixed N=64 rather than depending on whatever the sweep script left
   in build/.
"""
import subprocess
import os
import json
import hashlib
import numpy as np

from commitment import PedersenCommitment, quantize, CURVE
from export_witness import export_slice

CIRCOM_BIN = r"C:\circom\circom.exe"
CIRCUITS_DIR = os.path.abspath("../circuits")
N = 64


def run(cmd, cwd=CIRCUITS_DIR, input_text=None, timeout=120):
    result = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True,
                             text=True, input=input_text, timeout=timeout)
    return result.stdout + result.stderr, result.returncode


def check_1_tampered_commitment():
    print("=" * 60)
    print("Check 1: tampered commitment opening is rejected")
    print("=" * 60)

    pc = PedersenCommitment()
    real_vec = quantize(np.random.randn(64).astype(np.float32))
    C, v, r = pc.commit(real_vec)

    honest_ok = pc.verify(C, v, r)
    print(f"  Honest opening verifies:  {honest_ok}")

    tampered_vec = quantize(np.random.randn(64).astype(np.float32))
    h = hashlib.sha256(tampered_vec.tobytes()).digest()
    tampered_v = int.from_bytes(h, "big") % CURVE.field.n

    tampered_ok = pc.verify(C, tampered_v, r)
    print(f"  Tampered opening rejected: {not tampered_ok}")

    passed = honest_ok and not tampered_ok
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def check_2_wrong_client_excluded():
    print("\n" + "=" * 60)
    print("Check 2: proof for wrong exclusion claim fails")
    print("=" * 60)

    print(f"  Recompiling circuit for N={N} (self-contained, ignores build/ state)...")
    out, rc = run(f"python gen_circuit.py {N}")
    if rc != 0:
        print("  gen_circuit failed:", out); return False

    out, rc = run(f'"{CIRCOM_BIN}" correction.circom --r1cs --wasm --sym -o build')
    if rc != 0:
        print("  compile failed:", out); return False

    out, rc = run("snarkjs groth16 setup build/correction.r1cs pot14_final.ptau zkey_0000.zkey")
    if rc != 0:
        print("  setup failed:", out); return False

    out, rc = run("snarkjs zkey contribute zkey_0000.zkey zkey_final.zkey --name=phase8",
                   input_text="phase8entropy999\n")
    if rc != 0:
        print("  contribute failed:", out); return False

    out, rc = run("snarkjs zkey export verificationkey zkey_final.zkey vkey.json")
    if rc != 0:
        print("  export vkey failed:", out); return False

    print("  Circuit + keys ready for N=64.")

    print("\n  Generating HONEST witness (correct target)...")
    os.chdir("../src")
    export_slice(slice_size=N, output_path="../circuits/input_honest.json", layer_idx=3)
    os.chdir(CIRCUITS_DIR)

    out, rc = run("node build/correction_js/generate_witness.js build/correction_js/correction.wasm "
                  "input_honest.json witness_honest.wtns")
    if rc != 0:
        print("  Honest witness gen FAILED unexpectedly:", out)
        return False

    out, rc = run("snarkjs groth16 prove zkey_final.zkey witness_honest.wtns proof_honest.json public_honest.json")
    honest_proved = (rc == 0)
    print(f"  Honest proof generated: {honest_proved}")

    out, rc = run("snarkjs groth16 verify vkey.json public_honest.json proof_honest.json")
    honest_verified = "OK!" in out
    print(f"  Honest proof verifies:  {honest_verified}")

    print("\n  Generating ADVERSARIAL witness (tampered dW_out, inconsistent exclusion claim)...")
    with open("input_honest.json") as f:
        adversarial_inputs = json.load(f)
    adversarial_inputs["dW_out"] = [x + 999999999 for x in adversarial_inputs["dW_out"]]
    with open("input_adversarial.json", "w") as f:
        json.dump(adversarial_inputs, f)

    out, rc = run("node build/correction_js/generate_witness.js build/correction_js/correction.wasm "
                  "input_adversarial.json witness_adversarial.wtns")
    adversarial_witness_rejected_at_gen = (rc != 0)
    print(f"  Adversarial witness rejected at generation (constraint unsatisfiable): "
          f"{adversarial_witness_rejected_at_gen}")

    if adversarial_witness_rejected_at_gen:
        print("  (No proof attempted -- circuit enforces the relation at witness time,")
        print("   which is a stronger guarantee than rejecting at verification.)")
        passed = honest_verified and adversarial_witness_rejected_at_gen
        print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
        return passed

    out, rc = run("snarkjs groth16 prove zkey_final.zkey witness_adversarial.wtns "
                  "proof_adversarial.json public_adversarial.json")
    if rc == 0:
        out, rc = run("snarkjs groth16 verify vkey.json public_adversarial.json proof_adversarial.json")
        adversarial_rejected_at_verify = "OK!" not in out
        print(f"  Adversarial proof rejected at verification: {adversarial_rejected_at_verify}")
        passed = honest_verified and adversarial_rejected_at_verify
    else:
        passed = honest_verified
        print("  Adversarial proof generation failed (also an acceptable rejection point).")

    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    r1 = check_1_tampered_commitment()
    r2 = check_2_wrong_client_excluded()

    print("\n" + "=" * 60)
    print("PHASE 8 SUMMARY")
    print("=" * 60)
    print(f"  Check 1 (tampered commitment rejected):     {'PASS' if r1 else 'FAIL'}")
    print(f"  Check 2 (inconsistent-exclusion rejected):  {'PASS' if r2 else 'FAIL'}")
    print(f"  Overall: {'ALL PASS' if (r1 and r2) else 'FAILURE -- needs investigation'}")


if __name__ == "__main__":
    main()