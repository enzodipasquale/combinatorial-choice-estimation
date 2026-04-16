# Portfolio-choice scenario

Zero-noise test of combest with continuous controls.
Agents choose a binary inclusion vector and continuous portfolio weights
jointly via MIQP. The estimator recovers parameter ratios from observed
choices with no idiosyncratic errors.

## How to run

```bash
source .bundle/bin/activate
cd paper/numerical_experiments/scenarios/portfolio
mpirun -n 4 python run.py
```

## Files

- `dgp.py` -- Synthetic data: characteristics, covariance, true parameters.
- `solver.py` -- MIQP subproblem solver (Gurobi).
- `oracle.py` -- Covariates and error oracles.
- `run.py` -- End-to-end driver.

## Expected output

All estimated ratios within 0.5% of truth. Runtime ~50s on 4 MPI ranks.
