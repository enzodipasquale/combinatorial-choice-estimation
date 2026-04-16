"""Plot convergence of zero-limit estimator as sigma -> 0."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent


def main():
    with open(BASE / 'sigma_path.json') as f:
        data = json.load(f)

    theta_true = np.array(data['theta_true'])
    sigmas = data['sigmas']
    results = data['results']

    # Compute L1, L2, Linf for each sigma (mean over reps)
    l1_means, l2_means, linf_means = [], [], []
    l1_ses, l2_ses, linf_ses = [], [], []

    for sigma in sigmas:
        r = results[str(sigma)]
        bias = np.array(r['bias'])
        rmse = np.array(r['rmse'])

        # Per-rep distances would be better, but we have aggregate stats.
        # Use RMSE as proxy for E[|error|] per component.
        # L2 = sqrt(sum of RMSE^2), L1 = sum of RMSE, Linf = max of RMSE
        l1_means.append(rmse.sum())
        l2_means.append(np.sqrt((rmse ** 2).sum()))
        linf_means.append(rmse.max())

    sigmas = np.array(sigmas)
    l1_means = np.array(l1_means)
    l2_means = np.array(l2_means)
    linf_means = np.array(linf_means)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    ax.plot(sigmas, l1_means, 'o-', color='#1f77b4', lw=2, ms=7, label=r'$L_1$')
    ax.plot(sigmas, l2_means, 's-', color='#d62728', lw=2, ms=7, label=r'$L_2$')
    ax.plot(sigmas, linf_means, '^-', color='#2ca02c', lw=2, ms=7, label=r'$L_\infty$')

    ax.set_xlabel(r'$\sigma$ (DGP error std)', fontsize=13)
    ax.set_ylabel(r'$\|\hat{\theta} - \theta^*\|$', fontsize=13)
    ax.set_title('Zero-limit estimator: convergence as $\\sigma \\to 0$',
                 fontsize=13)
    ax.legend(fontsize=12, frameon=True)
    ax.set_xlim(-0.05, sigmas.max() + 0.1)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    out_path = BASE / 'convergence.png'
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == '__main__':
    main()
