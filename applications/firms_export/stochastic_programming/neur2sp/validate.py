import argparse
import time
import numpy as np
import torch
from neur2sp.generate_data import generate_dataset


def load_nn(model_path):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    weights, biases = ckpt["weights"], ckpt["biases"]

    def predict(X):
        h = X.copy()
        for i, (W, b) in enumerate(zip(weights, biases)):
            h = h @ W.T + b
            if i < len(weights) - 1:
                h = np.maximum(h, 0)
        return h.squeeze(-1)

    return predict, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="neur2sp/model.pt")
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--R_test", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed_chars", type=int, default=42)
    ap.add_argument("--seed_test", type=int, default=999)
    args = ap.parse_args()

    predict, ckpt = load_nn(args.model)
    M, n_rev = int(ckpt["M"]), int(ckpt["n_rev"])
    theta_lb, theta_ub = ckpt["theta_lb"], ckpt["theta_ub"]
    beta = ckpt.get("beta", 3.0)
    K = ckpt.get("K", M // 2)
    print(f"Model: M={M}, K={K}, n_rev={n_rev}")

    from neur2sp.generate_data import _make_chars
    _, rev_chars_2, syn_chars = _make_chars(M, n_rev, args.seed_chars)

    theta_bounds = {
        "theta_rev": (float(theta_lb[0]), float(theta_ub[0])),
        "theta_s": (float(theta_lb[n_rev]), float(theta_ub[n_rev])),
        "theta_c": (float(theta_lb[n_rev + 1]), float(theta_ub[n_rev + 1])),
    }

    print(f"Generating {args.n_test} test samples  "
          f"(R_test={args.R_test}, workers={args.workers})")
    t0 = time.time()
    X_test, y_test = generate_dataset(
        rev_chars_2, syn_chars, beta, M, K, n_rev,
        theta_bounds, n_samples=args.n_test, R_train=args.R_test,
        seed=args.seed_test, workers=args.workers,
    )
    print(f"Test data generated in {time.time()-t0:.1f}s")

    y_pred = predict(X_test)
    residuals = y_pred - y_test
    mae = np.abs(residuals).mean()
    rmse = np.sqrt((residuals ** 2).mean())
    ss_res = (residuals ** 2).sum()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum()
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    mape = np.abs(residuals / np.where(
        np.abs(y_test) > 1e-6, y_test, 1e-6)).mean() * 100

    print(f"\n{'='*50}")
    print(f"OUT-OF-SAMPLE VALIDATION  (n={args.n_test})")
    print(f"{'='*50}")
    print(f"  R²    = {r2:.4f}")
    print(f"  MAE   = {mae:.4f}")
    print(f"  RMSE  = {rmse:.4f}")
    print(f"  MAPE  = {mape:.1f}%")
    print(f"  Label range   : [{y_test.min():.2f}, {y_test.max():.2f}]")
    print(f"  Pred  range   : [{y_pred.min():.2f}, {y_pred.max():.2f}]")

    abs_res = np.abs(residuals)
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  P{p:2d} abs error = {np.percentile(abs_res, p):.4f}")

    out_path = args.model.replace(".pt", "_validation.npz")
    np.savez(out_path, X_test=X_test, y_test=y_test, y_pred=y_pred,
             residuals=residuals, r2=r2, mae=mae, rmse=rmse)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
