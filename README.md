# combchoice

MPI-parallel estimation for combinatorial discrete choice models via row generation.

## Install

```bash
pip install -e .
```

Requires MPI. Gurobi optional but needed for row generation.

## Quick Start

```python
import combchoice as cc
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    cfg = {'dimensions': {'n_obs': 100, 'n_items': 20, 'n_covariates': 5}}
    input_data = {
        "id_data": {"modular": covariates},  # (agents, items, covariates)
        "item_data": {"quadratic": quadratic},  # optional
    }
else:
    cfg = None
    input_data = None

model = combchoice.Model()
model.load_config(cfg)
model.data.load_and_distribute_input_data(input_data)
model.features.build_quadratic_covariates_from_data()
model.features.build_local_modular_error_oracle(seed=42)

model.subproblems.load_solver('QuadraticKnapsackGRB')
model.subproblems.generate_obs_bundles(theta_star)

result = model.row_generation.solve()
print(result.theta_hat)
```

Run: `mpirun -n 10 python script.py`

## Standard Errors

```python
model.features.build_local_modular_error_oracle(seed=27)
se_result = model.standard_errors.compute_bootstrap(num_bootstrap=50, seed=123)
```

## Tests

```bash
mpirun -n 10 pytest combchoice/tests/
```
