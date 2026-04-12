# SAR Error Robustness Pipeline

This folder re-runs the bootstrap **estimation** under spatially-autoregressive (SAR) errors at four values of the spatial dependence parameter ρ ∈ {0, 0.2, 0.4, 0.6}. The goal is to check whether the assumption of iid item-specific errors biases the structural parameter estimates and downstream welfare conclusions.

## Why not use the existing `error_correlation` path

The existing `error_correlation` config field reuses entries from the `QUADRATIC` feature registry (e.g. `pop_centroid_delta4`) as covariance matrices. Those matrices are designed as utility features — they are row-normalized and population-scaled to avoid size effects in the structural complementarity term. When symmetrized and used as covariances, the off-diagonals collapse to ~0 (empirically verified: max off-diagonal ~0.006 for adjacency, ~0.026 for pop_centroid_delta4). "Correlated" runs are thus numerically identical to iid. This pipeline builds the covariance with its own dedicated machinery that is completely separate from the QUADRATIC registry.

## SAR model

$$\nu = \rho W \nu + u, \quad u \sim \mathcal{N}(0, I)$$

$$\Rightarrow \quad \Sigma = (I - \rho W)^{-1}(I - \rho W)^{-\top}$$

rescaled so that $\text{diag}(\Sigma) = 1$. $W$ is the row-normalized binary adjacency matrix of BTAs.

## Running a single ρ

```bash
# from repo root
mpirun -n 4 python applications/combinatorial_auction/scripts/c_block/sar_robustness/run.py \
    applications/combinatorial_auction/scripts/c_block/configs/sar_rho04.yaml
```

Results are saved to `../results/sar_rho04/bootstrap_result.json`.

## Launching all four ρ values on HPC

```bash
bash applications/combinatorial_auction/scripts/c_block/sar_robustness/launch_all.sh
```

Adapt the sbatch flags in `launch_all.sh` to match your cluster's partition/account as needed.
