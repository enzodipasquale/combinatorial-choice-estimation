from bundlechoice.utils import get_logger
logger = get_logger(__name__)

def adaptive_gurobi_timeout(initial_timeout=1.0, final_timeout=90.0, transition_iterations=10, strategy='linear'):
    def callback(iteration, row_gen_manager):
        if iteration == 0:
            row_gen_manager.cfg.min_iters = max(row_gen_manager.cfg.min_iters, transition_iterations)

        if iteration < transition_iterations:
            if strategy == 'linear':
                progress = iteration / transition_iterations
                timeout = initial_timeout + (final_timeout - initial_timeout) * progress
            elif strategy == 'exponential':
                progress = iteration / transition_iterations
                timeout = initial_timeout * (final_timeout / initial_timeout) ** progress if initial_timeout > 0 else final_timeout
            elif strategy == 'step':
                timeout = initial_timeout
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        else:
            timeout = final_timeout
        row_gen_manager.subproblem_manager.update_gurobi_settings({'TimeLimit': timeout})


    return callback

def enforce_slack_counter():
    def callback(iteration, row_gen_manager):
        max_slack_counter = row_gen_manager.cfg.max_slack_counter
        to_remove = []
        for constr in row_gen_manager.master_model.getConstrs():
            if constr.CBasis == 0:
                row_gen_manager.slack_counter.pop(constr, None)
            else:
                row_gen_manager.slack_counter[constr] = row_gen_manager.slack_counter.get(constr, 0) + 1
                if row_gen_manager.slack_counter[constr] >= max_slack_counter:
                    to_remove.append(constr)
        for constr in to_remove:
            row_gen_manager.master_model.remove(constr)
            row_gen_manager.slack_counter.pop(constr, None)
    return callback

