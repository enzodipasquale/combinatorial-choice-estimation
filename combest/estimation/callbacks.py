import numpy as np


def adaptive_gurobi_timeout(schedule):
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
        settings, phase_idx = _get_settings(iteration)
        row_gen_manager.subproblem_manager.update_gurobi_settings(settings)
        if phase_idx < len(phases):
            allow = phases[phase_idx].get('retire', False)
            row_gen_manager.cfg.min_iters = 0 if allow else int(boundaries[phase_idx])
        else:
            allow = final.get('retire', False)
            row_gen_manager.cfg.min_iters = 0 if allow else float('inf')

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
