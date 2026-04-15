#!/usr/bin/env python3
"""Test: custom solver with pop_centroid_delta4 as fixed quadratic term."""
import sys, yaml
import numpy as np
from pathlib import Path

ZERO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ZERO_DIR.parent.parent.parent.parent.parent))

import warnings; warnings.filterwarnings("ignore")

from applications.combinatorial_auction.data.prepare import prepare
from applications.combinatorial_auction.data.loaders import load_bta_data, build_context
from applications.combinatorial_auction.data.registries import QUADRATIC

raw = load_bta_data()
ctx = build_context(raw)

# prepare with elig_pop as modular (will go in fixed_linear),
# adjacency as estimated quadratic, pop_centroid_delta4 excluded (goes in fixed_quadratic)
cfg = yaml.safe_load(open(ZERO_DIR / "baseline_small.yaml"))
app = cfg["application"]

input_data, meta = prepare(
    dataset="c_block",
    modular_regressors=[],
    quadratic_regressors=["adjacency"],  # only adjacency estimated
    quadratic_id_regressors=["elig_adjacency"],
    item_modular="fe",
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

n_cov = meta["n_covariates"]
theta = np.zeros(n_cov)  # dummy theta
print(f"n_cov={n_cov}, theta={theta.shape}")

import combest as ce

model = ce.Model()
model.load_config(cfg)
model.data.load_and_distribute_input_data(input_data)
model.features.build_quadratic_covariates_from_data()

cm = model.features.comm_manager
n_items = model.n_items
n_local = cm.num_local_agent

# fixed linear: elig_pop
elig = model.data.local_data.id_data["elig"]
pop = model.data.local_data.item_data["weight"].astype(float)
pop = pop / pop.sum()
fixed_linear = np.zeros((n_local, n_items))
for i in range(n_local):
    fixed_linear[i] = elig[i] * pop

# fixed quadratic: pop_centroid_delta4 (item-level, broadcast to all agents)
Q_fixed = QUADRATIC["pop_centroid_delta4"](ctx)
print(f"Q_fixed shape: {Q_fixed.shape}, symmetric: {np.allclose(Q_fixed, Q_fixed.T)}")

# need zero errors for combest path
model.features.local_modular_errors = np.zeros((n_local, n_items))
model.features._error_oracle = lambda b, ids: 0.0
model.features._error_oracle_takes_data = False

from knapsack import ZeroNoiseKnapsack

solver = ZeroNoiseKnapsack(
    comm_manager=cm,
    data_manager=model.data,
    features_manager=model.features,
    dimensions_cfg=model.config.dimensions,
    gurobi_params=cfg.get("subproblem", {}).get("gurobi_params"),
    fixed_linear=fixed_linear,
    fixed_quadratic=Q_fixed,  # broadcast: (n_items, n_items) -> added to all agents
)
solver.initialize()

bundles = solver.solve(theta)
print(f"\nWith elig_pop (linear) + pop_centroid_delta4 (quadratic) fixed at 1:")
print(f"  avg bundle size: {bundles.sum(1).mean():.1f}")
print(f"  nonempty: {(bundles.sum(1) > 0).sum()}/{n_local}")
print(f"  max bundle: {bundles.sum(1).max()}")

# verify utility decomposition for a few agents
for i in range(min(5, n_local)):
    b = bundles[i]
    if b.sum() == 0:
        continue
    u_lin = fixed_linear[i] @ b
    u_quad = b @ Q_fixed @ b
    print(f"  agent {i}: |b|={b.sum()}, u_linear={u_lin:.4f}, u_quad_fixed={u_quad:.4f}, total={u_lin+u_quad:.4f}")
