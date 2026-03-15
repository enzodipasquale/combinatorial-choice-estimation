import argparse
import time
import numpy as np
import gurobipy as gp
from multiprocessing import Pool
from functools import partial


def _solve_one_sample(sample_idx, M, entry_chars, syn_chars,
                      beta, beta_perpetual, sigma_nu_2,
                      theta_bounds, eff_rev_bounds, R_train, seed):
    rng = np.random.default_rng((seed, sample_idx))

    eff_rev = rng.uniform(eff_rev_bounds[0], eff_rev_bounds[1], size=M)
    theta_s = rng.uniform(*theta_bounds["theta_s"])
    theta_sc = rng.uniform(*theta_bounds["theta_sc"])
    theta_c = rng.uniform(*theta_bounds["theta_c"])
    b1 = rng.integers(0, 2, size=M).astype(float)

    entry_2 = beta * (theta_s + theta_sc * entry_chars)
    syn_2 = theta_c * beta_perpetual * syn_chars

    m = gp.Model("p2")
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    b2 = m.addMVar(M, vtype=gp.GRB.BINARY, name="b2")
    m.update()

    total_val = 0.0
    for r in range(R_train):
        nu2 = rng.normal(0, sigma_nu_2, M)
        c = eff_rev + beta * nu2 + (1 - b1) * entry_2
        m.setObjective(c @ b2 + b2 @ syn_2 @ b2, gp.GRB.MAXIMIZE)
        m.optimize()
        if m.Status == gp.GRB.OPTIMAL:
            total_val += m.ObjVal

    x = np.concatenate([b1, eff_rev, [theta_s, theta_sc, theta_c]])
    y = total_val / R_train
    return sample_idx, x, y


def generate_dataset(entry_chars, syn_chars, beta, beta_perpetual, sigma_nu_2,
                     M, theta_bounds, eff_rev_bounds,
                     n_samples=5000, R_train=500, seed=123, workers=1):
    func = partial(
        _solve_one_sample,
        M=M, entry_chars=entry_chars, syn_chars=syn_chars,
        beta=beta, beta_perpetual=beta_perpetual, sigma_nu_2=sigma_nu_2,
        theta_bounds=theta_bounds, eff_rev_bounds=eff_rev_bounds,
        R_train=R_train, seed=seed,
    )

    n_x = 2 * M + 3
    inputs = np.empty((n_samples, n_x))
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


THETA_BOUNDS = {
    "theta_s": (-10.0, 0.0),
    "theta_sc": (-5.0, 0.0),
    "theta_c": (-1.0, 2.0),
}

EFF_REV_BOUNDS = (-50.0, 50.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=20)
    ap.add_argument("--beta", type=float, default=0.8)
    ap.add_argument("--sigma_nu_2", type=float, default=0.5)
    ap.add_argument("--n_samples", type=int, default=10000)
    ap.add_argument("--R_train", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed_chars", type=int, default=42)
    ap.add_argument("--seed_data", type=int, default=123)
    ap.add_argument("--out", type=str, default="neur2sp/data.npz")
    ap.add_argument("--eff_rev_lb", type=float, default=EFF_REV_BOUNDS[0])
    ap.add_argument("--eff_rev_ub", type=float, default=EFF_REV_BOUNDS[1])
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--n_chunks", type=int, default=1)
    args = ap.parse_args()

    beta_perpetual = args.beta / (1 - args.beta)
    rng_c = np.random.default_rng(args.seed_chars)
    entry_chars = rng_c.uniform(0, 1, args.M)
    _raw = rng_c.uniform(0, 1, (args.M, args.M))
    syn_chars = (_raw + _raw.T) / 2
    np.fill_diagonal(syn_chars, 0)

    eff_rev_bounds = (args.eff_rev_lb, args.eff_rev_ub)

    chunk_size = (args.n_samples + args.n_chunks - 1) // args.n_chunks
    start = args.chunk * chunk_size
    end = min(start + chunk_size, args.n_samples)
    n_local = end - start

    func = partial(
        _solve_one_sample,
        M=args.M, entry_chars=entry_chars, syn_chars=syn_chars,
        beta=args.beta, beta_perpetual=beta_perpetual,
        sigma_nu_2=args.sigma_nu_2,
        theta_bounds=THETA_BOUNDS, eff_rev_bounds=eff_rev_bounds,
        R_train=args.R_train, seed=args.seed_data,
    )

    n_x = 2 * args.M + 3
    inputs = np.empty((n_local, n_x))
    labels = np.empty(n_local)
    sample_indices = range(start, end)
    t0 = time.time()

    print(f"Chunk {args.chunk}/{args.n_chunks}: samples [{start}, {end})  "
          f"(M={args.M}, R={args.R_train}, workers={args.workers})")

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

    save_dict = dict(
        inputs=inputs, labels=labels,
        entry_chars=entry_chars, syn_chars=syn_chars,
        M=args.M, beta=args.beta,
        beta_perpetual=beta_perpetual,
        sigma_nu_2=args.sigma_nu_2,
        eff_rev_bounds=np.array(eff_rev_bounds),
        theta_bounds_s=THETA_BOUNDS["theta_s"],
        theta_bounds_sc=THETA_BOUNDS["theta_sc"],
        theta_bounds_c=THETA_BOUNDS["theta_c"],
    )

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
