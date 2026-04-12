# DC version — convex (nonlinear) parametrization

This subfolder contains the DC algorithm wrapper for the multi-stage production
experiment with nonlinear (convex) parametrization where the revenue factor
enters nonlinearly through FE or CES parameters.

The parent folder uses a fully linear parametrization where DC is unnecessary
(converges in 1 iteration). This subfolder is for future work with the
nonlinear extension.

## Files

- `dc.py` — DC algorithm wrapper (iterates between linearization and row generation)
- `run_experiment_dc.py` — estimation script using DC (TODO: implement with nonlinear utility)

## Key differences from parent

- Solver needs `set_q_linearization(theta)` to linearize the nonlinear terms at the current iterate
- Oracles need gradients of the nonlinear terms w.r.t. theta (stored via policies side-channel)
- DC outer loop iterates until theta converges
