from bundlechoice.utils import get_logger
logger = get_logger(__name__)

def adaptive_gurobi_timeout(initial_timeout=1.0, final_timeout=90.0, transition_iterations=10, strategy='linear', log=True):
    def callback(iteration, subproblem_manager, master_model):
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
        subproblem_manager.update_gurobi_settings({'TimeLimit': timeout})
        subproblem_manager._suboptimal_mode = timeout < final_timeout - 1e-06
    return callback

