from typing import Any, Optional
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
                if initial_timeout > 0:
                    timeout = initial_timeout * (final_timeout / initial_timeout) ** progress
                else:
                    timeout = final_timeout
            elif strategy == 'step':
                timeout = initial_timeout
            else:
                raise ValueError(f"Unknown strategy: {strategy}. Use 'linear', 'exponential', or 'step'")
        else:
            timeout = final_timeout
        subproblem_manager.update_settings({'TimeLimit': timeout})
        subproblem_manager._suboptimal_mode = timeout < final_timeout - 1e-06
        if log and master_model is not None:
            logger.info(f'[Iter {iteration}] Subproblem timeout: {timeout:.2f}s (suboptimal={subproblem_manager._suboptimal_mode})')
    return callback

def constant_timeout(timeout, log=False):

    def callback(iteration, subproblem_manager, master_model):
        subproblem_manager.update_settings({'TimeLimit': timeout})
        subproblem_manager._suboptimal_mode = False
        if log and master_model is not None:
            logger.info(f'[Iter {iteration}] Subproblem timeout: {timeout:.2f}s')
    return callback