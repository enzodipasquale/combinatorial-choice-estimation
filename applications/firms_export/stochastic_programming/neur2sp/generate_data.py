"""Generate training data for the Neur2SP value function approximation.

For each sample:
  1. Sample random theta from a specified range.
  2. Sample a random feasible b_1 (binary, sum <= K).
  3. Solve R_train independent period-2 knapsacks with fresh eps draws.
  4. Record: input = (b_1, theta), label = beta * mean(optimal period-2 values).

Usage:
    python -m neur2sp.generate_data                          (sequential)
    python -m neur2sp.generate_data --workers 8              (local parallel)
    python -m neur2sp.generate_data --M 20 --K 10 --R_train 1000 --workers 400
"""
import argparse
import time
import numpy as np
import gurobipy as gp
from multiprocessing import Pool
from functools import partial


def sample_feasible_b1(M, K, rng):
    """Random binary vector with sum <= K."""
    k = rng.integers(0, K + 1)
    b = np.zeros(M)
    if k > 0:
        idx = rng.choice(M, size=min(k, M), replace=False)
        b[idx] = 1.0
    return b


def _solve_one_sample(sample_idx, M, K, n_rev, beta,
                      rev_chars_2, syn_chars, theta_bounds, R_train, seed):
    """Solve one training sample (self-contained for multiprocessing)."""
    rng = np.random.default_rng((seed, sample_idx))

    # sample theta
    theta_rev = rng.uniform(*theta_bounds["theta_rev"], size=n_rev)
    theta_s = rng.uniform(*theta_bounds["theta_s"])
    theta_c = rng.uniform(*theta_bounds["theta_c"])
    theta = np.concatenate([theta_rev, [theta_s, theta_c]])

    # sample feasible b_1
    b1 = sample_feasible_b1(M, K, rng)

    # reusable Gurobi model
    m = gp.Model("p2")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    b2 = m.addMVar(M, vtype=gp.GRB.BINARY, name="b2")
    m.addConstr(b2.sum() <= K, name="cap")
    m.update()

    C = syn_chars
    base_coeff = rev_chars_2.T @ theta_rev + (1 - b1) * theta_s

    total_val = 0.0
    for r in range(R_train):
        eps = rng.normal(0, 1, M)
        c = base_coeff + eps
        m.setObjective(c @ b2 + theta_c * (b2 @ C @ b2), gp.GRB.MAXIMIZE)
        m.optimize()
        if m.Status == gp.GRB.OPTIMAL:
            total_val += m.ObjVal

    x = np.concatenate([b1, theta])
    y = beta * total_val / R_train
    return sample_idx, x, y


def generate_dataset(
    rev_chars_2, syn_chars, beta, M, K, n_rev,
    theta_bounds, n_samples=5000, R_train=500, seed=123, workers=1,
):
    """Generate (input, label) pairs for the NN value-function approximation.

    Parameters
    ----------
    rev_chars_2 : (n_rev, M)
    syn_chars   : (M, M)
    beta, M, K, n_rev : problem parameters
    theta_bounds : dict with keys 'theta_rev', 'theta_s', 'theta_c'
    n_samples, R_train, seed : data generation parameters
    workers : int, number of parallel processes (1 = sequential)

    Returns
    -------
    inputs : (n_samples, M + n_rev + 2)
    labels : (n_samples,)
    """
    func = partial(
        _solve_one_sample,
        M=M, K=K, n_rev=n_rev, beta=beta,
        rev_chars_2=rev_chars_2, syn_chars=syn_chars,
        theta_bounds=theta_bounds, R_train=R_train, seed=seed,
    )

    inputs = np.empty((n_samples, M + n_rev + 2))
    labels = np.empty(n_samples)
    t0 = time.time()

    if workers <= 1:
        # sequential
        for s in range(n_samples):
            _, x, y = func(s)
            inputs[s], labels[s] = x, y
            if (s + 1) % 200 == 0:
                elapsed = time.time() - t0
                rate = (s + 1) / elapsed
                print(f"  [{s+1}/{n_samples}]  {rate:.1f} samples/s  "
                      f"ETA {(n_samples-s-1)/rate:.0f}s")
    else:
        # parallel
        done = 0
        with Pool(workers) as pool:
            for idx, x, y in pool.imap_unordered(func, range(n_samples),
                                                  chunksize=max(1, n_samples // (workers * 4))):
                inputs[idx], labels[idx] = x, y
                done += 1
                if done % 200 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    print(f"  [{done}/{n_samples}]  {rate:.1f} samples/s  "
                          f"ETA {(n_samples-done)/rate:.0f}s")

    return inputs, labels


# ── CLI ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=20)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--n_rev", type=int, default=1)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--n_samples", type=int, default=3000)
    ap.add_argument("--R_train", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of parallel processes (1 = sequential)")
    ap.add_argument("--seed_chars", type=int, default=42,
                    help="Seed for problem characteristics")
    ap.add_argument("--seed_data", type=int, default=123)
    ap.add_argument("--out", type=str, default="neur2sp/data.npz")
    args = ap.parse_args()

    # ── problem characteristics ──
    rng_c = np.random.default_rng(args.seed_chars)
    rev_base = rng_c.uniform(0, 1.0, (args.n_rev, args.M))
    rev_chars_1 = rev_base + rng_c.uniform(-0.1, 0.1, (args.n_rev, args.M))
    rev_chars_2 = rev_base + rng_c.uniform(-0.1, 0.1, (args.n_rev, args.M))
    _state = (rng_c.random((1000, args.M)) > 0.9).astype(float)
    _raw = rng_c.uniform(0, 1, (args.M, args.M))
    syn_chars = (_raw + _raw.T) / 2
    np.fill_diagonal(syn_chars, 0)

    theta_bounds = {
        "theta_rev": (-5.0, 5.0),
        "theta_s": (-10.0, 0.0),
        "theta_c": (-1.0, 2.0),
    }

    print(f"Generating {args.n_samples} samples  (M={args.M}, K={args.K}, "
          f"R_train={args.R_train}, workers={args.workers})")
    t0 = time.time()
    inputs, labels = generate_dataset(
        rev_chars_2, syn_chars, args.beta, args.M, args.K, args.n_rev,
        theta_bounds, args.n_samples, args.R_train, args.seed_data,
        workers=args.workers,
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s  —  label range [{labels.min():.2f}, "
          f"{labels.max():.2f}]")

    np.savez(
        args.out, inputs=inputs, labels=labels,
        rev_chars_2=rev_chars_2, syn_chars=syn_chars,
        M=args.M, K=args.K, n_rev=args.n_rev, beta=args.beta,
        theta_bounds_rev=theta_bounds["theta_rev"],
        theta_bounds_s=theta_bounds["theta_s"],
        theta_bounds_c=theta_bounds["theta_c"],
    )
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
