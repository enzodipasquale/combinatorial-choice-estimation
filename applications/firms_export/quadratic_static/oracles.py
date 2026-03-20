"""
Error oracle for the quadratic static model.

Error structure:
  e_{itj} = eps_{ij} (permanent, firm x dest) + nu_{itj} (transitory)
"""
import numpy as np


def build_oracles(model, firm_idx, seed=42, sigma_perm=1.0, sigma_trans=1.0):
    ld = model.data.local_data
    n = model.comm_manager.num_local_agent
    M = model.config.dimensions.n_items

    local_firm_idx = ld.id_data["firm_idx"]

    unique_firms = np.unique(local_firm_idx)
    firm_eps = {}
    for f in unique_firms:
        firm_eps[f] = np.random.default_rng((seed, int(f), 0)).normal(
            0, sigma_perm, M)

    eps_perm = np.zeros((n, M))
    eps_trans = np.zeros((n, M))
    for i, gid in enumerate(model.comm_manager.agent_ids):
        fi = local_firm_idx[i]
        eps_perm[i] = firm_eps[fi]
        eps_trans[i] = np.random.default_rng((seed, gid, 1)).normal(
            0, sigma_trans, M)

    errors = eps_perm + eps_trans
    model.features.local_modular_errors = errors

    def error_oracle(bundles, ids, data):
        b = bundles.astype(float)
        return (errors[ids] * b).sum(-1)

    return error_oracle
