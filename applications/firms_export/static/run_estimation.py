#!/bin/env python
import sys
import yaml
from pathlib import Path
import combest as ce
from combest.estimation.callbacks import adaptive_gurobi_timeout

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "data"))
from prepare_data import main as load_data, build_input_data
from solver import (DiscountedJointQuadKnapsackSolver, discounted_covariates_oracle,
                    discount_errors, count_covariates, FEATURE_KEYS,
                    COVARIATE_NAMES, COVARIATE_LBS)

config = yaml.safe_load(open(BASE_DIR / "config.yaml"))
app = config["application"]


def build_covariate_meta(input_data):
    names, lbs = {}, {}
    off = 0
    for key, src in FEATURE_KEYS:
        arr = input_data[src].get(key)
        if arr is None:
            continue
        ncols = arr.shape[-1]
        knames = COVARIATE_NAMES.get(key, [f"{key}[{c}]" for c in range(ncols)])
        for c in range(ncols):
            names[off + c] = knames[c] if c < len(knames) else f"{key}[{c}]"
            if key in COVARIATE_LBS:
                lbs[off + c] = COVARIATE_LBS[key]
        off += ncols
    return names, lbs


def build_model(n_sample=None, **overrides):
    a = {**app, **overrides}
    model = ce.Model()

    if model.is_root():
        ctx = load_data(a["country"], a["keep_top"], a["discount"], n_sample=n_sample)
        input_data = build_input_data(ctx)
        covariate_names, lbs = build_covariate_meta(input_data)

        config["dimensions"].update(
            n_obs=ctx["n_obs"], n_items=ctx["n_items"],
            n_covariates=count_covariates(input_data),
            covariate_names=covariate_names)
        config["row_generation"]["theta_bounds"]["lbs"] = lbs
        if "standard_errors" in config:
            config["standard_errors"].setdefault("theta_bounds", {})["lbs"] = lbs
        n_dest = ctx["n_dest"]
    else:
        input_data, n_dest = None, None

    n_dest = model.comm_manager.bcast(n_dest)
    model.load_config(config)
    model.data.load_and_distribute_input_data(input_data)

    model.features.set_covariates_oracle(discounted_covariates_oracle)
    model.features.build_local_modular_error_oracle(seed=a["seed"])
    discount_errors(model, n_dest)

    model.subproblems.load_solver(DiscountedJointQuadKnapsackSolver)
    model.subproblems.initialize_solver()

    return model


if __name__ == "__main__":
    model = build_model(n_sample=app.get("n_sample"))
    pt_cb, _ = adaptive_gurobi_timeout(config["callbacks"]["row_gen"])
    model.row_generation.solve(iteration_callback=pt_cb, verbose=True)