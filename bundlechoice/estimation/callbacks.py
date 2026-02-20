import numpy as np


def adaptive_gurobi_timeout(schedule, final_timeout):
    """
    Multi-phase adaptive Gurobi timeout for row generation.

    Returns (pt_callback, dist_callback) tuple:
      - pt_callback(iteration, row_gen_manager) for point estimation
      - dist_callback(rg_round, mixin, master) for distributed bootstrap

    Each phase can set 'min_iters' to prevent early convergence during
    warmup phases. The callback sets mixin.config.standard_errors.rowgen_min_iters
    accordingly.

    Args:
        schedule: list of dicts, each with:
            'iters': int — number of iterations in this phase
            'timeout': float — Gurobi TimeLimit
            'min_iters': int (optional) — set rowgen_min_iters for this phase
        final_timeout: float — TimeLimit after all phases complete

    Example:
        pt_cb, dist_cb = adaptive_gurobi_timeout(
            schedule=[
                {'iters': 5, 'timeout': 1.5, 'min_iters': 5},
                {'iters': 10, 'timeout': 5.0},
            ],
            final_timeout=5.0,
        )
    """
    boundaries = np.cumsum([p['iters'] for p in schedule])

    def _get_settings(iteration):
        idx = np.searchsorted(boundaries, iteration, side='right')
        if idx < len(schedule):
            timeout = schedule[idx]['timeout']
        else:
            timeout = final_timeout
        settings = {'TimeLimit': timeout,
                    'MIPFocus': 1 if iteration == 0 else 0}
        return settings, idx

    def pt_callback(iteration, row_gen_manager):
        settings, _ = _get_settings(iteration)
        row_gen_manager.subproblem_manager.update_gurobi_settings(settings)

    def dist_callback(rg_round, mixin, master):
        settings, phase_idx = _get_settings(rg_round)
        mixin.subproblem_manager.update_gurobi_settings(settings)
        if phase_idx < len(schedule) and 'min_iters' in schedule[phase_idx]:
            mixin.config.standard_errors.rowgen_min_iters = schedule[phase_idx]['min_iters']

    return pt_callback, dist_callback
