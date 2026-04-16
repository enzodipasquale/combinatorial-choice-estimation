# SAR Error Robustness

Re-runs bootstrap estimation under spatially-autoregressive errors at rho in {0, 0.2, 0.4, 0.6}.

The standard `error_correlation` config path reuses QUADRATIC registry features as covariance matrices, but their row-normalization makes off-diagonals collapse to ~0. This pipeline builds SAR covariance directly from the binary adjacency matrix.

## SAR model

nu = rho * W * nu + u, u ~ N(0, I)
=> Sigma = (I - rho*W)^{-1} (I - rho*W)^{-T}, rescaled to unit diagonal.

## Usage

```bash
mpirun -n 4 python .../sar_robustness/run.py .../configs/sar_rho04.yaml
```

Run all four: `mpirun -n N python .../sar_robustness/run_all.py`
