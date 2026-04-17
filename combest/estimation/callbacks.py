import numpy as np


def _schedule_lookup(schedule):
    phases, final = schedule[:-1], schedule[-1]
    boundaries = np.cumsum([p['iters'] for p in phases])

    def lookup(iteration):
        idx = np.searchsorted(boundaries, iteration, side='right')
        spec = phases[idx] if idx < len(phases) else final
        settings = {'TimeLimit': spec['timeout'],
                    'MIPFocus': 1 if iteration == 0 else 0}
        min_iters = 0 if spec.get('retire', False) else (
            int(boundaries[idx]) if idx < len(phases) else float('inf'))
        return settings, min_iters
    return lookup


def point_timeout_callback(schedule):
    lookup = _schedule_lookup(schedule)
    def cb(iteration, row_gen_manager):
        settings, min_iters = lookup(iteration)
        row_gen_manager.subproblem_manager.update_gurobi_settings(settings)
        row_gen_manager.cfg.min_iters = min_iters
    return cb


def bootstrap_timeout_callback(schedule, strip=None):
    lookup = _schedule_lookup(schedule)
    def cb(rg_round, mixin, master):
        settings, min_iters = lookup(rg_round)
        mixin.subproblem_manager.update_gurobi_settings(settings)
        mixin.config.standard_errors.rowgen_min_iters = min_iters
        if strip and master is not None and rg_round == 0:
            master.strip_slack_constraints(
                percentile=strip['percentile'],
                hard_threshold=strip['hard_threshold'])
    return cb


def adaptive_gurobi_timeout(schedule):
    """Back-compat: returns (point_callback, bootstrap_callback) pair."""
    return point_timeout_callback(schedule), bootstrap_timeout_callback(schedule)
