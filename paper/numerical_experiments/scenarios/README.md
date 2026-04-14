# Scenarios

Numerical experiments for the JMP, organized per-scenario. Each scenario is an independent pipeline: its own DGP, its own oracle, its own estimation run. Scenarios do not share code beyond what `combest/` already provides.

## Layout

```
scenarios/
├── README.md                 # this file
├── airline/                  # GS, custom find_best_item greedy
├── multi_stage_production/   # multi-stage MILP
└── quad_knapsack/            # structured quadratic knapsack + BLP inversion
```

## Per-scenario contents

Every scenario folder contains:

- `SPEC.md` — economics and math. Source of truth for the scenario's DGP, utility function, oracle, and deliverable.
- `config.yaml` — run parameters.
- `generate_data.py`, `oracle.py`, `run.py`, `tests/` — the code.
- `results/result.json` — the deliverable.

## Status

| Scenario | Status | Notes |
|---|---|---|
| airline | **pilot** | Simplest oracle (greedy), validates the pattern. |
| multi_stage_production | pilot | Multi-stage MILP. Linear parametrization; DC version is WIP. |
| quad_knapsack | pilot | Quadratic knapsack + BLP inversion. Targets $M \approx 500$. |

## Design principles

- **BLP only in auction.** Other scenarios estimate $\theta$ (modular + interaction params) without item fixed effects $\delta$.
- **One $M$ per scenario.** No $M \times N$ grids. The showcase $M$ per scenario reflects what its demand oracle can handle.
- **One replication for now.** Scaling to many replications and HPC is a later step.
- **Incremental builds.** Each step is verified at small scale before scaling up.
