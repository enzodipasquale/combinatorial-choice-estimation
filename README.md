# BundleChoice

MPI-parallel estimation for combinatorial discrete choice models via row generation.

## Install

```bash
pip install -e .
```

Requires MPI. Gurobi optional but needed for row generation.

## Quick Start

```python
import bundlechoice as bc
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    cfg = {'dimensions': {'n_obs': 100, 'n_items': 20, 'n_features': 5}}
    input_data = {
        "id_data": {"modular": features},  # (agents, items, features)
        "item_data": {"quadratic": quadratic},  # optional
    }
else:
    cfg = None
    input_data = None

bc = bc.BundleChoice()
bc.load_config(cfg)
bc.data.load_and_distribute_input_data(input_data)
bc.oracles.build_quadratic_features_from_data()
bc.oracles.build_local_modular_error_oracle(seed=42)

bc.subproblems.load_subproblem('QuadraticKnapsackGRB')
bc.subproblems.generate_obs_bundles(theta_star)

result = bc.row_generation.solve()
print(result.theta_hat)
```

Run: `mpirun -n 10 python script.py`

## Custom Data

```python
bc = bc.BundleChoice()
bc.load_config(cfg)
bc.data.load_and_distribute_input_data(input_data)
bc.oracles.build_quadratic_features_from_data()
bc.oracles.build_local_modular_error_oracle(seed=42)

bc.subproblems.load_subproblem('QuadraticKnapsackGRB')
bc.subproblems.generate_obs_bundles(theta_star)

result = bc.row_generation.solve()
```

## Standard Errors

```python
bc.oracles.build_local_modular_error_oracle(seed=27)
se_result = bc.standard_errors.compute_bootstrap(num_bootstrap=50, seed=123)
```

## Tests

```bash
mpirun -n 10 pytest bundlechoice/tests/
```
