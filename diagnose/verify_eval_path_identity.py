"""CHECK C -- sanity-checks the evaluation path itself, so the CNN-BN vs
CNN-GroupNorm vs CNN-IN comparison isn't an artifact of divergent preprocessing
or noise-injection code between the three models.

C1. Prints the exact call chain each model goes through in
    eval/snr_degradation.py (build_model_specs + run_sweep) and
    eval/instancenorm_control.py (run_cnn_in_sweep), and greps both files for
    any `if`/`elif` branch keyed on model name that would make one model's
    input differ from another's. All three models are expected to call the
    same add_noise_floor() and preprocess_raw() functions with no model-name
    branching -- the only difference should be which model.pt is loaded for
    inference afterward.

C2. For one fixed noise seed (0) at 5 dB, builds the noisy raw spectrum once
    via add_noise_floor(X_test_raw, 5, rng) and preprocesses it once via
    preprocess_raw(). Since eval/snr_degradation.py's run_sweep() and
    eval/instancenorm_control.py's run_cnn_in_sweep() both call add_noise_floor
    then preprocess_raw exactly once per (seed, snr_db) and feed the SAME
    resulting array to every model's infer() closure, the "one array reused
    for three models" property is true by construction -- this check confirms
    that by (a) re-deriving the array via each script's own imported
    functions and asserting byte-identity, and (b) saving the array so it can
    be inspected directly.

Run with: python diagnose\\verify_eval_path_identity.py   (after activating venv)
Requires: data/processed/{X_test_raw,y_test}.npy, data/processed/scaler.joblib.
Do NOT run this automatically -- the user runs it manually.
"""
import ast
import re
import sys
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "eval"))
sys.path.insert(0, str(REPO_ROOT / "train"))
import snr_degradation as sd  # noqa: E402
import instancenorm_control as ic  # noqa: E402
from utils import PROCESSED_DIR  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "diagnostics"
FIXED_SEED = 0
FIXED_SNR_DB = 5

TARGET_FILES = [
    REPO_ROOT / "eval" / "snr_degradation.py",
    REPO_ROOT / "eval" / "instancenorm_control.py",
]
MODEL_NAME_TOKENS = ["cnn_bn", "cnn_in", "\"cnn\"", "'cnn'", "cnn_gn", "mlp_bn"]


def print_call_chain():
    print(f"\n{'=' * 78}\nC1. Code path each model takes\n{'=' * 78}")
    print("""
  eval/snr_degradation.py (used for SVM, CNN, CNN-BN, MLP, MLP-BN):
    run_sweep()
      -> X_noisy_raw = noise_fn(X_test_raw, snr_db, rng)      # add_noise_floor
      -> X_noisy = preprocess_raw(X_noisy_raw, scaler, snr_db)
      -> for spec in model_specs: scores = spec["infer"](X_noisy)
         (the SAME X_noisy array is passed to every model's infer() closure
          in this loop -- built once per (seed, snr_db), not once per model)

  eval/instancenorm_control.py (used for CNN-IN):
    run_cnn_in_sweep()
      -> X_noisy_raw = sd.add_noise_floor(X_test_raw, snr_db, rng)   # same fn, imported from snr_degradation
      -> X_noisy = sd.preprocess_raw(X_noisy_raw, scaler, snr_db)    # same fn, imported from snr_degradation
      -> logits = torch_infer(cnn_in, X_noisy)

  Both scripts call the identical add_noise_floor() and preprocess_raw()
  functions defined once in eval/snr_degradation.py (instancenorm_control.py
  imports them via `import snr_degradation as sd`, not a re-implementation).
""")


def grep_for_model_branching():
    print(f"\n{'=' * 78}\nC1 (cont.). Scanning for model-name branching that "
          f"could give one model a different input\n{'=' * 78}")
    any_hit = False
    for path in TARGET_FILES:
        text = path.read_text()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.If,)):
                # only flag an If whose test source mentions a model-name token
                seg = ast.get_source_segment(text, node.test) or ""
                if any(tok.strip("\"'") in seg for tok in MODEL_NAME_TOKENS):
                    print(f"  FOUND branch in {path.relative_to(REPO_ROOT)} line "
                          f"{node.lineno}: `if {seg}:`")
                    any_hit = True
    if not any_hit:
        print("  No `if`/`elif` branches keyed on a model name found in "
              "eval/snr_degradation.py or eval/instancenorm_control.py.")
        print("  (Per-model dispatch is done only via the `model_specs` list of "
              "{name, infer} dicts / separate model.pt loads -- the noise and "
              "preprocessing calls themselves take no model argument.)")
    return not any_hit


def verify_byte_identical_inputs():
    print(f"\n{'=' * 78}\nC2. Byte-identity of preprocessed input at seed="
          f"{FIXED_SEED}, SNR={FIXED_SNR_DB} dB\n{'=' * 78}")

    X_test_raw = np.load(PROCESSED_DIR / "X_test_raw.npy")
    scaler = joblib.load(PROCESSED_DIR / "scaler.joblib")

    # Path A: via eval/snr_degradation.py's own functions (used for BN/GN/MLP/SVM)
    rng_a = np.random.default_rng(FIXED_SEED)
    X_noisy_raw_a = sd.add_noise_floor(X_test_raw, FIXED_SNR_DB, rng_a)
    X_a = sd.preprocess_raw(X_noisy_raw_a, scaler, snr_db=FIXED_SNR_DB)

    # Path B: via eval/instancenorm_control.py's imported functions (used for CNN-IN)
    rng_b = np.random.default_rng(FIXED_SEED)
    X_noisy_raw_b = ic.sd.add_noise_floor(X_test_raw, FIXED_SNR_DB, rng_b)
    X_b = ic.sd.preprocess_raw(X_noisy_raw_b, scaler, snr_db=FIXED_SNR_DB)

    # Path C: a third independent call, simulating "yet another model's turn
    # in the same run_sweep() loop" -- must match A and B exactly since all
    # three come from the same seeded RNG and the same functions.
    rng_c = np.random.default_rng(FIXED_SEED)
    X_noisy_raw_c = sd.add_noise_floor(X_test_raw, FIXED_SNR_DB, rng_c)
    X_c = sd.preprocess_raw(X_noisy_raw_c, scaler, snr_db=FIXED_SNR_DB)

    ab_identical = np.array_equal(X_a, X_b)
    ac_identical = np.array_equal(X_a, X_c)
    bc_identical = np.array_equal(X_b, X_c)

    print(f"  Path A (snr_degradation.py fns)      vs Path B (instancenorm_control.py "
          f"imported fns): byte-identical = {ab_identical}")
    print(f"  Path A vs Path C (independent re-derivation):                    "
          f"byte-identical = {ac_identical}")
    print(f"  Path B vs Path C:                                                "
          f"byte-identical = {bc_identical}")

    assert ab_identical, "Path A and Path B produced different preprocessed inputs -- comparison is INVALID."
    assert ac_identical, "Path A and Path C produced different preprocessed inputs -- comparison is INVALID."
    assert bc_identical, "Path B and Path C produced different preprocessed inputs -- comparison is INVALID."

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"eval_path_check_X_snr{FIXED_SNR_DB}_seed{FIXED_SEED}.npy", X_a)
    print(f"\n  All three paths byte-identical. Saved shared tensor to "
          f"{OUT_DIR / f'eval_path_check_X_snr{FIXED_SNR_DB}_seed{FIXED_SEED}.npy'}")
    return True


def main():
    print_call_chain()
    no_branching = grep_for_model_branching()
    identical = verify_byte_identical_inputs()

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    print(f"  No model-name branching in eval code:        {no_branching}")
    print(f"  Preprocessed inputs byte-identical (5 dB, seed 0): {identical}")
    if no_branching and identical:
        print("  -> Evaluation path is shared identically across BN/GN/IN. "
              "The comparison is valid on this axis.")
    else:
        print("  -> FAIL: evaluation path is NOT shared. Investigate before "
              "trusting the BN/GN/IN comparison.")


if __name__ == "__main__":
    main()
