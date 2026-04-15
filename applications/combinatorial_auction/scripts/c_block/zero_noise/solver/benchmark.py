#!/usr/bin/env python3
"""Benchmark: compare custom ZeroNoiseKnapsack against combest's solver for elig_pop case."""
import sys, yaml
import numpy as np
from pathlib import Path

ZERO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ZERO_DIR.parent.parent.parent.parent.parent))

import warnings; warnings.filterwarnings("ignore")

from applications.combinatorial_auction.data.prepare import prepare
from applications.combinatorial_auction.data.loaders import load_bta_data

# load data
raw = load_bta_data()
cfg = yaml.safe_load(open(ZERO_DIR / "baseline_small.yaml"))
app = cfg["application"]

input_data, meta = prepare(
    dataset=app["dataset"],
    modular_regressors=app.get("modular_regressors", []),
    quadratic_regressors=app.get("quadratic_regressors", []),
    quadratic_id_regressors=app.get("quadratic_id_regressors", []),
    item_modular=app.get("item_modular", "fe"),
)
meta.pop("raw", None)

cfg["dimensions"].update(
    n_obs=meta["n_obs"], n_items=meta["n_items"],
    n_covariates=meta["n_covariates"],
    covariate_names=meta["covariate_names"],
)
app.update(
    n_id_mod=meta["n_id_mod"], n_item_mod=meta["n_item_mod"],
    n_id_quad=meta["n_id_quad"], n_item_quad=meta["n_item_quad"],
)

# use a dummy theta from boot results
import json
boot = json.load(open(ZERO_DIR.parent / "results" / "boot" / "bootstrap_result.json"))
theta_full = np.array(boot["theta_hat"])

# the zero-noise config has no modular regressors, so theta is shorter
# theta order: item_mod (480 FEs) + id_quad + item_quad
n_cov = meta["n_covariates"]
theta = theta_full[:n_cov]  # just take first n_cov params as dummy
print(f"n_cov={n_cov}, theta len={len(theta)}")

# ── Method 1: combest solver with elig_pop in error oracle ──
import combest as ce

model = ce.Model()
model.load_config(cfg)
model.data.load_and_distribute_input_data(input_data)
model.features.build_quadratic_covariates_from_data()

# set elig_pop as deterministic error
cm = model.features.comm_manager
n_items = model.n_items
elig = model.data.local_data.id_data["elig"]
pop = model.data.local_data.item_data["weight"].astype(float)
pop = pop / pop.sum()
local_errors = np.zeros((cm.num_local_agent, n_items))
for i in range(cm.num_local_agent):
    local_errors[i] = elig[i] * pop
model.features.local_modular_errors = local_errors
model.features._error_oracle = lambda b, ids: (model.features.local_modular_errors[ids] * b).sum(-1)
model.features._error_oracle_takes_data = False

model.subproblems.load_solver()
solver_combest = model.subproblems.subproblem_solver
solver_combest.initialize()

bundles_combest = solver_combest.solve(theta)
print(f"combest: {bundles_combest.sum(1).mean():.2f} avg bundle size, "
      f"{(bundles_combest.sum(1) > 0).sum()} nonempty")

# ── Method 2: custom solver with fixed_linear = elig_pop ──
from knapsack import ZeroNoiseKnapsack

solver_custom = ZeroNoiseKnapsack(
    comm_manager=cm,
    data_manager=model.data,
    features_manager=model.features,
    dimensions_cfg=model.config.dimensions,
    gurobi_params=cfg.get("subproblem", {}).get("gurobi_params"),
    fixed_linear=local_errors,
    fixed_quadratic=None,
)
solver_custom.initialize()

bundles_custom = solver_custom.solve(theta)
print(f"custom:  {bundles_custom.sum(1).mean():.2f} avg bundle size, "
      f"{(bundles_custom.sum(1) > 0).sum()} nonempty")

# ── Compare ──
match = (bundles_combest == bundles_custom).all()
print(f"\nExact match: {match}")
if not match:
    diff = (bundles_combest != bundles_custom).any(axis=1)
    print(f"  Differing agents: {diff.sum()}/{len(diff)}")
    for i in np.where(diff)[0][:5]:
        print(f"  agent {i}: combest |b|={bundles_combest[i].sum()}, custom |b|={bundles_custom[i].sum()}")
