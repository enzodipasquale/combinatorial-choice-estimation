"""Oracle functions for the FCC-calibrated quadratic knapsack scenario.

Utility for agent i choosing bundle b:
  V_i(b) = sum_{j in b} x_ij^T theta_mod  - sum_{j in b} delta_j
           + lambda sum_{j<j' in b} Q_{jj'} b_j b_{j'}
           + sum_{j in b} nu_ij
"""

import numpy as np
from pathlib import Path

# Reuse 2SLS from parent via spec loader
import importlib.util as _ilu
_parent_oracle = Path(__file__).resolve().parent.parent / 'oracle.py'
_spec = _ilu.spec_from_file_location('parent_oracle', _parent_oracle)
_pmod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pmod)
twosls_smoke_test = _pmod.twosls_smoke_test


def compute_utility(bundle, x_mod_i, delta, Q_dense, theta_mod, lambda_,
                    errors_i):
    """Full utility V_i(b) for one agent with 6-regressor specification."""
    b = bundle.astype(float)
    linear = (x_mod_i[bundle] @ theta_mod).sum() - delta[bundle].sum()
    if errors_i is not None:
        linear += errors_i[bundle].sum()
    quadratic = 0.5 * lambda_ * b @ Q_dense @ b
    return linear + quadratic
