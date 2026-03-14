import argparse
import time
import numpy as np
import gurobipy as gp
from multiprocessing import Pool
from functools import partial


def sample_feasible_b1(M, K, rng):
    k = rng.integers(0, K + 1)
    b = np.zeros(M)
    if k > 0:
        idx = rng.choice(M, size=min(k, M), replace=False)
        b[idx] = 1.0
    return b


def _solve_one_sample(sample_idx, M, K, n_rev, beta,
                      rev_chars_2, syn_chars, theta_bounds, R_train, seed):
    rng = np.random.default_rng((seed, sample_idx))

    theta_rev = rng.uniform(*theta_bounds["theta_rev"], size=n_rev)
    theta_s = rng.uniform(*theta_bounds["theta_s"])
    theta_c = rng.uniform(*theta_bounds["theta_c"])
    theta = np.concatenate([theta_rev, [theta_s, theta_c]])

    b1 = sample_feasible_b1(M, K, rng)

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
        m.setObjective((base_coeff + eps) @ b2 + theta_c * (b2 @ C @ b2),
                       gp.GRB.MAXIMIZE)
        m.optimize()
        if m.Status == gp.GRB.OPTIMAL:
            total_val += m.ObjVal

    x = np.concatenate([b1, theta])
    y = beta * total_val / R_train
    return sample_idx, x, y


def generate_dataset(rev_chars_2, syn_chars, beta, M, K, n_rev,
                     theta_bounds, n_samples=5000, R_train=500,
                     seed=123, workers=1):
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
        for s in range(n_samples):
            _, x, y = func(s)
            inputs[s], labels[s] = x, y
            if (s + 1) % 200 == 0:
                elapsed = time.time() - t0
                rate = (s + 1) / elapsed
                print(f"  [{s+1}/{n_samples}]  {rate:.1f} samples/s  "
                      f"ETA {(n_samples-s-1)/rate:.0f}s")
    else:
        done = 0
        with Pool(workers) as pool:
            for idx, x, y in pool.imap_unordered(
                    func, range(n_samples),
                    chunksize=max(1, n_samples // (workers * 4))):
                inputs[idx], labels[idx] = x, y
                done += 1
                if done % 200 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    print(f"  [{done}/{n_samples}]  {rate:.1f} samples/s  "
                          f"ETA {(n_samples-done)/rate:.0f}s")

    return inputs, labels


def _make_chars(M, n_rev, seed_chars):
    rng_c = np.random.default_rng(seed_chars)
    rev_base = rng_c.uniform(0, 1.0, (n_rev, M))
    rev_chars_1 = rev_base + rng_c.uniform(-0.1, 0.1, (n_rev, M))
    rev_chars_2 = rev_base + rng_c.uniform(-0.1, 0.1, (n_rev, M))
    _ = (rng_c.random((1000, M)) > 0.9).astype(float)
    _raw = rng_c.uniform(0, 1, (M, M))
    syn_chars = (_raw + _raw.T) / 2
    np.fill_diagonal(syn_chars, 0)
    return rev_chars_1, rev_chars_2, syn_chars


THETA_BOUNDS = {
    "theta_rev": (-5.0, 5.0),
    "theta_s": (-10.0, 0.0),
    "theta_c": (-1.0, 2.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=20)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--n_rev", type=int, default=1)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--n_samples", type=int, default=10000)
    ap.add_argument("--R_train", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed_chars", type=int, default=42)
    ap.add_argument("--seed_data", type=int, default=123)
    ap.add_argument("--out", type=str, default="neur2sp/data.npz")
    ap.add_argument("--chunk", type=int, default=0,
                    help="chunk index (0-based) for distributed generation")
    ap.add_argument("--n_chunks", type=int, default=1,
                    help="total number of chunks")
    args = ap.parse_args()

    _, rev_chars_2, syn_chars = _make_chars(args.M, args.n_rev, args.seed_chars)

    chunk_size = (args.n_samples + args.n_chunks - 1) // args.n_chunks
    start = args.chunk * chunk_size
    end = min(start + chunk_size, args.n_samples)
    n_local = end - start

    func = partial(
        _solve_one_sample,
        M=args.M, K=args.K, n_rev=args.n_rev, beta=args.beta,
        rev_chars_2=rev_chars_2, syn_chars=syn_chars,
        theta_bounds=THETA_BOUNDS, R_train=args.R_train, seed=args.seed_data,
    )

    n_cov = args.n_rev + 2
    inputs = np.empty((n_local, args.M + n_cov))
    labels = np.empty(n_local)
    sample_indices = range(start, end)
    t0 = time.time()

    print(f"Chunk {args.chunk}/{args.n_chunks}: samples [{start}, {end})  "
          f"(M={args.M}, K={args.K}, R={args.R_train}, workers={args.workers})")

    if args.workers <= 1:
        for i, s in enumerate(sample_indices):
            _, x, y = func(s)
            inputs[i], labels[i] = x, y
            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  [{i+1}/{n_local}]  {rate:.1f} samples/s  "
                      f"ETA {(n_local-i-1)/rate:.0f}s")
    else:
        done = 0
        with Pool(args.workers) as pool:
            for idx, x, y in pool.imap_unordered(
                    func, sample_indices,
                    chunksize=max(1, n_local // (args.workers * 4))):
                inputs[idx - start], labels[idx - start] = x, y
                done += 1
                if done % 200 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    print(f"  [{done}/{n_local}]  {rate:.1f} samples/s  "
                          f"ETA {(n_local-done)/rate:.0f}s")

    print(f"Done in {time.time()-t0:.1f}s  —  "
          f"label range [{labels.min():.2f}, {labels.max():.2f}]")

    save_dict = dict(inputs=inputs, labels=labels,
                     rev_chars_2=rev_chars_2, syn_chars=syn_chars,
                     M=args.M, K=args.K, n_rev=args.n_rev, beta=args.beta,
                     theta_bounds_rev=THETA_BOUNDS["theta_rev"],
                     theta_bounds_s=THETA_BOUNDS["theta_s"],
                     theta_bounds_c=THETA_BOUNDS["theta_c"])

    if args.n_chunks > 1:
        out = args.out.replace(".npz", f"_chunk{args.chunk}.npz")
    else:
        out = args.out
    np.savez(out, **save_dict)
    print(f"Saved to {out}")


def merge_chunks(pattern, out_path, n_chunks):
    all_inputs, all_labels = [], []
    for c in range(n_chunks):
        path = pattern.replace(".npz", f"_chunk{c}.npz")
        d = np.load(path)
        all_inputs.append(d["inputs"])
        all_labels.append(d["labels"])
    d0 = np.load(pattern.replace(".npz", "_chunk0.npz"))
    save_dict = {k: d0[k] for k in d0.files
                 if k not in ("inputs", "labels")}
    save_dict["inputs"] = np.concatenate(all_inputs)
    save_dict["labels"] = np.concatenate(all_labels)
    np.savez(out_path, **save_dict)
    print(f"Merged {n_chunks} chunks → {out_path}  "
          f"({len(save_dict['labels'])} samples)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        merge_chunks(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        main()
