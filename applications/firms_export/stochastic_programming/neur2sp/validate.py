"""Out-of-sample validation of the trained Neur2SP value function.

Generates a held-out test set with high R (accurate labels), then compares
NN predictions vs true expected period-2 values.

Usage:
    python -m neur2sp.validate                               (sequential)
    python -m neur2sp.validate --workers 8                   (parallel)
    python -m neur2sp.validate --model neur2sp/model.pt --n_test 500
"""
import argparse
import time
import numpy as np
import torch
from neur2sp.generate_data import generate_dataset, sample_feasible_b1


def load_nn(model_path):
    """Load trained NN and return a prediction function (numpy in/out)."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    weights = ckpt["weights"]  # already have normalization folded in
    biases = ckpt["biases"]

    def predict(X):
        """X : (n, input_dim) numpy → (n,) numpy predictions."""
        h = X.copy()
        for i, (W, b) in enumerate(zip(weights, biases)):
            h = h @ W.T + b
            if i < len(weights) - 1:  # ReLU on hidden layers only
                h = np.maximum(h, 0)
        return h.squeeze(-1)

    return predict, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="neur2sp/model.pt")
    ap.add_argument("--n_test", type=int, default=500,
                    help="Number of test samples")
    ap.add_argument("--R_test", type=int, default=1000,
                    help="Scenarios per test sample (high for accurate labels)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed_chars", type=int, default=42)
    ap.add_argument("--seed_test", type=int, default=999,
                    help="Seed for test data (must differ from training seed)")
    args = ap.parse_args()

    # ── load model ──
    predict, ckpt = load_nn(args.model)
    M = int(ckpt["M"])
    n_rev = int(ckpt["n_rev"])
    theta_lb = ckpt["theta_lb"]
    theta_ub = ckpt["theta_ub"]
    print(f"Loaded model: M={M}, n_rev={n_rev}, input_dim={ckpt['input_dim']}")

    # ── reconstruct problem characteristics (same seed as training) ──
    rng_c = np.random.default_rng(args.seed_chars)
    rev_base = rng_c.uniform(0, 1.0, (n_rev, M))
    _rev1 = rev_base + rng_c.uniform(-0.1, 0.1, (n_rev, M))
    rev_chars_2 = rev_base + rng_c.uniform(-0.1, 0.1, (n_rev, M))
    _state = (rng_c.random((1000, M)) > 0.9).astype(float)
    _raw = rng_c.uniform(0, 1, (M, M))
    syn_chars = (_raw + _raw.T) / 2
    np.fill_diagonal(syn_chars, 0)

    # theta bounds from checkpoint
    theta_bounds = {
        "theta_rev": (float(theta_lb[0]), float(theta_ub[0])),
        "theta_s": (float(theta_lb[n_rev]), float(theta_ub[n_rev])),
        "theta_c": (float(theta_lb[n_rev + 1]), float(theta_ub[n_rev + 1])),
    }

    beta = ckpt.get("beta", 3.0)
    K = ckpt.get("K", M // 2)

    # ── generate test set (different seed!) ──
    print(f"\nGenerating {args.n_test} test samples  (R_test={args.R_test}, "
          f"workers={args.workers})")
    t0 = time.time()
    X_test, y_test = generate_dataset(
        rev_chars_2, syn_chars, beta, M, K, n_rev,
        theta_bounds, n_samples=args.n_test, R_train=args.R_test,
        seed=args.seed_test, workers=args.workers,
    )
    print(f"Test data generated in {time.time()-t0:.1f}s")

    # ── NN predictions ──
    y_pred = predict(X_test)

    # ── metrics ──
    residuals = y_pred - y_test
    mae = np.abs(residuals).mean()
    rmse = np.sqrt((residuals ** 2).mean())
    ss_res = (residuals ** 2).sum()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum()
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    mape = np.abs(residuals / np.where(np.abs(y_test) > 1e-6, y_test, 1e-6)).mean() * 100

    print(f"\n{'='*50}")
    print(f"OUT-OF-SAMPLE VALIDATION  (n={args.n_test})")
    print(f"{'='*50}")
    print(f"  R²    = {r2:.4f}")
    print(f"  MAE   = {mae:.4f}")
    print(f"  RMSE  = {rmse:.4f}")
    print(f"  MAPE  = {mape:.1f}%")
    print(f"  Label range   : [{y_test.min():.2f}, {y_test.max():.2f}]")
    print(f"  Pred  range   : [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    print(f"  Residual mean : {residuals.mean():.4f}")
    print(f"  Residual std  : {residuals.std():.4f}")

    # ── percentile breakdown ──
    pcts = [10, 25, 50, 75, 90]
    abs_res = np.abs(residuals)
    print(f"\n  Absolute error percentiles:")
    for p in pcts:
        print(f"    P{p:2d} = {np.percentile(abs_res, p):.4f}")

    # ── save results ──
    out_path = args.model.replace(".pt", "_validation.npz")
    np.savez(out_path, X_test=X_test, y_test=y_test, y_pred=y_pred,
             residuals=residuals, r2=r2, mae=mae, rmse=rmse)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
