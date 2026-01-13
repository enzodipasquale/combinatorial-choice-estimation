"""
Reusable callbacks for estimation solvers.

This module provides common callback functions for subproblem management,
such as adaptive timeout schedules and other iterative strategies.
"""

from typing import Any, Optional
from bundlechoice.utils import get_logger

logger = get_logger(__name__)


def adaptive_gurobi_timeout(
    initial_timeout: float = 1.0,
    final_timeout: float = 90.0,
    transition_iterations: int = 10,
    strategy: str = "linear",
    log: bool = True
):
    """
    Create an adaptive Gurobi timeout callback that gradually increases subproblem timeout.
    
    Early iterations use fast, suboptimal cuts (low timeout).
    Later iterations use higher timeout for more optimal cuts.
    
    Args:
        initial_timeout: Starting timeout in seconds (default: 1.0)
        final_timeout: Final timeout in seconds (default: 90.0)
        transition_iterations: Number of iterations to transition (default: 10)
        strategy: Interpolation strategy - "linear", "exponential", or "step" (default: "linear")
        log: Whether to log timeout updates on root process (default: True)
    
    Returns:
        Callback function with signature: (iteration, subproblem_manager, master_model) -> None
    
    Example:
        >>> callback = adaptive_gurobi_timeout(initial=1.0, final=90.0, transition_iterations=10)
        >>> config.row_generation.subproblem_callback = callback
    """
    def callback(iteration: int, subproblem_manager: Any, master_model: Optional[Any]) -> None:
        """Adaptive timeout callback implementation."""
        if iteration < transition_iterations:
            # During transition phase
            if strategy == "linear":
                progress = iteration / transition_iterations
                timeout = initial_timeout + (final_timeout - initial_timeout) * progress
            elif strategy == "exponential":
                progress = iteration / transition_iterations
                if initial_timeout > 0:
                    timeout = initial_timeout * ((final_timeout / initial_timeout) ** progress)
                else:
                    timeout = final_timeout
            elif strategy == "step":
                # Step function: use initial until transition, then final
                timeout = initial_timeout
            else:
                raise ValueError(f"Unknown strategy: {strategy}. Use 'linear', 'exponential', or 'step'")
        else:
            # After transition: use final timeout
            timeout = final_timeout
        
        # Update subproblem timeout
        subproblem_manager.update_settings({"TimeLimit": timeout})
        
        # Track suboptimal mode (for preventing early stopping)
        subproblem_manager._suboptimal_mode = (timeout < final_timeout - 1e-6)
        
        # Optional: log on root process
        if log and master_model is not None:
            logger.info(f"[Iter {iteration}] Subproblem timeout: {timeout:.2f}s "
                  f"(suboptimal={subproblem_manager._suboptimal_mode})")
    
    return callback


def constant_timeout(timeout: float, log: bool = False):
    """
    Create a callback that sets a constant timeout (useful for testing/debugging).
    
    Args:
        timeout: Timeout value in seconds
        log: Whether to log timeout updates (default: False)
    
    Returns:
        Callback function
    """
    def callback(iteration: int, subproblem_manager: Any, master_model: Optional[Any]) -> None:
        """Constant timeout callback implementation."""
        subproblem_manager.update_settings({"TimeLimit": timeout})
        subproblem_manager._suboptimal_mode = False
        
        if log and master_model is not None:
            logger.info(f"[Iter {iteration}] Subproblem timeout: {timeout:.2f}s")
    
    return callback

