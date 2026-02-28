#!/usr/bin/env python3
import sys
import argparse
import numpy as np
from pathlib import Path
from mpi4py import MPI

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import combchoice as cc
from paper.numerical_experiments.large_scale.generate_data import generate_data


def bundle_stats(spec, M, alpha=0.1, lambda_val=None, N=25, seed=0, sigma=1.0):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        input_data, theta_star = generate_data(spec, N, M, alpha=alpha, lambda_val=lambda_val, rho=0.5, seed=seed)
        n_mod_agent = input_data["id_data"]["modular"].shape[-1] if "modular" in input_data["id_data"] else 0
        n_mod_item = input_data["item_data"]["modular"].shape[-1] if "modular" in input_data["item_data"] else 0
        n_quad_agent = input_data["id_data"]["quadratic"].shape[-1] if "quadratic" in input_data["id_data"] else 0
        n_quad_item = input_data["item_data"]["quadratic"].shape[-1] if "quadratic" in input_data["item_data"] else 0
        subproblem = {"gross_substitutes": "Greedy",
                      "supermodular": "QuadraticSupermodularMinCut",
                      "linear_knapsack": "LinearKnapsackGRB"}[spec]
        dim_cfg = {"n_obs": N, "n_items": M,
                   "n_covariates": n_mod_agent + n_mod_item + n_quad_agent + n_quad_item,
                   "n_simulations": 1}
    else:
        input_data = theta_star = dim_cfg = subproblem = None

    dim_cfg = comm.bcast(dim_cfg, root=0)
    subproblem = comm.bcast(subproblem, root=0)

    model = cc.Model()
    model.load_config({"dimensions": dim_cfg, "subproblem": {"name": subproblem},
                     "row_generation": {"theta_bounds": {"lb": -100, "ub": 100}}})
    model.data.load_and_distribute_input_data(input_data)
    model.features.build_quadratic_covariates_from_data()
    model.features.build_local_modular_error_oracle(seed=3 * seed + 1, sigma=sigma)
    model.subproblems.load_solver()

    theta_star = comm.bcast(theta_star, root=0)
    obs_bundles = model.subproblems.generate_obs_bundles(theta_star)

    if rank != 0:
        return None

    sizes = obs_bundles.sum(axis=1)
    item_chosen = obs_bundles.sum(axis=0)
    return {
        "mean": float(np.mean(sizes)),
        "std": float(np.std(sizes)),
        "min": int(np.min(sizes)),
        "max": int(np.max(sizes)),
        "frac_empty": float(np.mean(sizes == 0)),
        "frac_full": float(np.mean(sizes == M)),
        "n_unchosen": int(np.sum(item_chosen == 0)),
        "n_always": int(np.sum(item_chosen == N)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, choices=["gross_substitutes", "supermodular", "linear_knapsack"])
    parser.add_argument("--M", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lambdas", type=float, nargs="+", default=None)
    parser.add_argument("--N", type=int, default=25)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=1.0)
    args = parser.parse_args()

    rank = MPI.COMM_WORLD.Get_rank()

    if args.lambdas is None:
        if args.spec == "gross_substitutes":
            lambdas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        else:
            lambdas = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    else:
        lambdas = args.lambdas

    for M in args.M:
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"  {args.spec}, M={M}, sigma={args.sigma}")
            print(f"{'='*80}")
            print(f"  {'lambda':>8s}  {'mean':>6s}  {'std':>5s}  {'min':>4s}  {'max':>4s}  "
                  f"{'%empty':>6s}  {'%full':>6s}  {'unchos':>6s}  {'always':>6s}")
            print(f"  {'-'*70}")

        for lam in lambdas:
            agg = []
            for s in range(args.seeds):
                st = bundle_stats(args.spec, M, alpha=args.alpha, lambda_val=lam, N=args.N, seed=s, sigma=args.sigma)
                if st is not None:
                    agg.append(st)

            if rank == 0 and agg:
                avg = {k: np.mean([a[k] for a in agg]) for k in agg[0]}
                print(f"  {lam:8.3f}  {avg['mean']:6.1f}  {avg['std']:5.1f}  "
                      f"{avg['min']:4.0f}  {avg['max']:4.0f}  "
                      f"{avg['frac_empty']:6.1%}  {avg['frac_full']:6.1%}  "
                      f"{avg['n_unchosen']:6.1f}  {avg['n_always']:6.1f}")


if __name__ == "__main__":
    main()
