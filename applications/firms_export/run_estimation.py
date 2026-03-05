#!/bin/env python
import sys
from pathlib import Path
import combest as ce
from combest.estimation.callbacks import adaptive_gurobi_timeout

sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
from prepare_data import main as prepare_data, build_input_data
from solver import DiscountedJointQuadKnapsackSolver, discounted_covariates_oracle, \
    discount_errors, count_covariates, FEATURE_KEYS

TIMEOUT_SCHEDULE = [
    {"iters": 3, "timeout": 0.5},
    {"iters": 8, "timeout": 3.0},
    {"timeout": 10.0, "retire": True},
]

COVARIATE_NAMES = {
    "modular_3d": ["revenue"],
    "modular_1d": ["oper×dist"],
    "entry_1d":   ["entry"],
    "quadratic_2d": ["proximity", "prox×dist"],
    "consec_1d":  ["consec×dist"],
}

COVARIATE_LBS = {
    "quadratic_2d": -1.0,
    "consec_1d": 0.0,
}


def build_model(country="MEX", keep_top=50, discount_factor=0.95,
                n_simulations=5, n_sample=None):
    model = ce.Model()

    if model.is_root():
        ctx = prepare_data(country, keep_top, discount_factor, n_sample=n_sample)
        input_data = build_input_data(ctx)
        n_cov = count_covariates(input_data)

        covariate_names, lbs = {}, {}
        off = 0
        for key, src in FEATURE_KEYS:
            arr = input_data[src].get(key)
            if arr is None:
                continue
            ncols = arr.shape[-1]
            names = COVARIATE_NAMES.get(key, [f"{key}[{c}]" for c in range(ncols)])
            for c in range(ncols):
                covariate_names[off + c] = names[c] if c < len(names) else f"{key}[{c}]"
                if key in COVARIATE_LBS:
                    lbs[off + c] = COVARIATE_LBS[key]
            off += ncols

        config = {
            "dimensions": {
                "n_obs": ctx["n_obs"],
                "n_items": ctx["n_items"],
                "n_covariates": n_cov,
                "n_simulations": n_simulations,
                "covariate_names": covariate_names,
            },
            "subproblem": {
                "gurobi_params": {"TimeLimit": 10},
            },
            "row_generation": {
                "max_iters": 200,
                "tolerance": 0.01,
                "theta_bounds": {"lb": -10000, "ub": 10000, "lbs": lbs},
            },
            "standard_errors": {
                "rowgen_max_iters": 100,
                "rowgen_tol": 0.01,
                "theta_bounds": {"lbs": lbs},
            },
        }
        n_dest = ctx["n_dest"]
    else:
        input_data, config, n_dest = None, {}, None

    n_dest = model.comm_manager.bcast(n_dest)
    model.load_config(config)
    model.data.load_and_distribute_input_data(input_data)

    model.features.set_covariates_oracle(discounted_covariates_oracle)
    model.features.build_local_modular_error_oracle(seed=42)
    discount_errors(model, n_dest)

    model.subproblems.load_solver(DiscountedJointQuadKnapsackSolver)
    model.subproblems.initialize_solver()

    return model


if __name__ == "__main__":
    model = build_model(keep_top=30, n_sample=200)
    pt_cb, _ = adaptive_gurobi_timeout(TIMEOUT_SCHEDULE)
    model.row_generation.solve(iteration_callback=pt_cb, verbose=True)
