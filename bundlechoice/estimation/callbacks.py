import numpy as np


def adaptive_gurobi_timeout(schedule):
    """
    Multi-phase adaptive Gurobi timeout for row generation.

    Returns (pt_callback, dist_callback) tuple:
      - pt_callback(iteration, row_gen_manager) for point estimation
      - dist_callback(rg_round, mixin, master) for distributed bootstrap

    Each phase controls whether bootstrap samples can retire (converge):
      - retire: True  → allow retirement when tolerance is met
      - retire: False → block retirement for the duration of this phase
      - omitted       → block retirement (default False)

    The last entry in the schedule has no 'iters' and serves as the
    final (open-ended) phase. All preceding entries must have 'iters'.

    Args:
        schedule: list of dicts, each with:
            'timeout': float — Gurobi TimeLimit
            'iters': int (required for all but last) — iterations in this phase
            'retire': bool (optional, default False) — allow convergence

    Example:
        pt_cb, dist_cb = adaptive_gurobi_timeout([
            {'iters': 2, 'timeout': 1.0},
            {'iters': 3, 'timeout': 1.5, 'retire': True},
            {'timeout': 5.0, 'retire': True},
        ])
    """
    phases = schedule[:-1]
    final = schedule[-1]
    boundaries = np.cumsum([p['iters'] for p in phases])

    def _get_settings(iteration):
        idx = np.searchsorted(boundaries, iteration, side='right')
        if idx < len(phases):
            timeout = phases[idx]['timeout']
        else:
            timeout = final['timeout']
        settings = {'TimeLimit': timeout,
                    'MIPFocus': 1 if iteration == 0 else 0}
        return settings, idx

    def pt_callback(iteration, row_gen_manager):
        settings, _ = _get_settings(iteration)
        row_gen_manager.subproblem_manager.update_gurobi_settings(settings)

    def dist_callback(rg_round, mixin, master):
        settings, phase_idx = _get_settings(rg_round)
        mixin.subproblem_manager.update_gurobi_settings(settings)
        if phase_idx < len(phases):
            allow = phases[phase_idx].get('retire', False)
            mixin.config.standard_errors.rowgen_min_iters = 0 if allow else int(boundaries[phase_idx])
        else:
            allow = final.get('retire', False)
            mixin.config.standard_errors.rowgen_min_iters = 0 if allow else float('inf')

    return pt_callback, dist_callback
