"""
SAR (spatial autoregressive) correlation matrix for C-block error robustness.

Model:  nu = rho * W * nu + u,   u ~ iid N(0, I)
     => nu ~ N(0, Sigma)   where  Sigma = (I - rho*W)^{-1} (I - rho*W)^{-T}

Sigma is then rescaled to a correlation matrix (diagonal == 1).
"""
import numpy as np


def build_sar_covariance(adjacency: np.ndarray, rho: float) -> np.ndarray:
    """
    Build a spatial-autoregressive correlation matrix from a binary adjacency matrix.

    Args:
        adjacency: (n, n) binary symmetric matrix with zero diagonal.
        rho:       spatial dependence parameter, must satisfy |rho| < 1.

    Returns:
        (n, n) PSD correlation matrix with diagonal exactly 1.

    Raises:
        ValueError on invalid inputs or non-PSD result.
    """
    adjacency = np.asarray(adjacency, dtype=float)
    n = adjacency.shape[0]

    if adjacency.ndim != 2 or adjacency.shape[1] != n:
        raise ValueError(f"adjacency must be square, got shape {adjacency.shape}")
    if not np.allclose(adjacency, adjacency.T, atol=1e-10):
        raise ValueError("adjacency must be symmetric")
    if np.any(np.diag(adjacency) != 0):
        raise ValueError("adjacency diagonal must be zero")
    if abs(rho) >= 1:
        raise ValueError(f"rho must satisfy |rho| < 1, got {rho}")

    # Row-normalize W (rows with no neighbors stay zero)
    row_sums = adjacency.sum(axis=1, keepdims=True)
    W = np.where(row_sums > 0, adjacency / row_sums, 0.0)

    # M = I - rho * W,  M_inv via solve
    M = np.eye(n) - rho * W
    M_inv = np.linalg.solve(M, np.eye(n))

    # Sigma_unnorm = M_inv @ M_inv.T
    Sigma_unnorm = M_inv @ M_inv.T

    # Rescale to correlation matrix
    d = np.sqrt(np.diag(Sigma_unnorm))
    Sigma = Sigma_unnorm / np.outer(d, d)
    np.fill_diagonal(Sigma, 1.0)  # fix any floating-point drift on diagonal

    # Verify PSD
    try:
        np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        raise ValueError(f"SAR covariance is not PSD at rho={rho}. Try a smaller rho.")

    return Sigma


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

    from applications.combinatorial_auction.data.loaders import load_bta_data, build_context

    raw = load_bta_data()
    ctx = build_context(raw)
    adj = ctx["bta_adjacency"]

    # Symmetrize and binarize the raw adjacency (entries > 0 -> 1)
    adj = ((adj + adj.T) > 0).astype(float)
    np.fill_diagonal(adj, 0)

    print(f"Adjacency matrix: {adj.shape[0]} BTAs, {int(adj.sum()//2)} edges")
    print(f"Mean neighbors per BTA: {adj.sum(1).mean():.2f}")
    print()

    for rho in [0.0, 0.2, 0.4, 0.6]:
        Sigma = build_sar_covariance(adj, rho)
        off = Sigma[np.triu_indices(adj.shape[0], k=1)]
        eigvals = np.linalg.eigvalsh(Sigma)
        print(f"rho={rho:.1f}:")
        print(f"  max off-diagonal:  {off.max():.6f}")
        print(f"  mean off-diagonal: {off.mean():.6f}")
        print(f"  min eigenvalue:    {eigvals.min():.6f}")
        print(f"  condition number:  {eigvals.max()/eigvals.min():.2f}")
        print()
