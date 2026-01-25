from dataclasses import dataclass, field
import numpy as np

from bundlechoice.utils import get_logger, format_number

logger = get_logger(__name__)


@dataclass
class RowGenerationEstimationResult:
    theta_hat: np.ndarray
    converged: bool
    num_iterations: int
    final_objective: None
    n_constraints: int = None
    final_reduced_cost: float = None
    total_time: float = None
    final_n_violations: int = None
    u_hat: np.ndarray = None
    timing: dict = field(default_factory=dict)
    iteration_history: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def log_summary(self, parameters_to_log=None):
        idx = parameters_to_log if parameters_to_log is not None else range(len(self.theta_hat))
        if isinstance(self.timing, tuple) and len(self.timing) >= 2:
            p, m = np.array(self.timing[0]), np.array(self.timing[1])
        else:
            p, m = np.array([]), np.array([])
        n_constraints = self.n_constraints if self.n_constraints is not None else 0
        n_violations = self.final_n_violations if self.final_n_violations is not None else 0
        reduced_cost = self.final_reduced_cost or 0.0
        total_time = self.total_time or 0.0
        n_iters = self.num_iterations
        obj_val = self.final_objective
        E_eps_B = obj_val / len(self.u_hat)  

        logger.info(" ")
        logger.info(" ROW GENERATION SUMMARY")
        param_labels = ' | '.join(f'{f"θ[{i}]":>12}' for i in idx)
        params_row = f"{'Parameters':>12} | {param_labels}"
        logger.info("-" * len(params_row))
        logger.info(params_row)
        param_vals = ' | '.join(format_number(self.theta_hat[i], width=12, precision=5) for i in idx)
        logger.info(f"{'':>12} | {param_vals}")
        logger.info("-" * 90)
        logger.info(f"{'Master':>12} | {'ObjVal':>12} | {'E[ε_B]':>12} | {'#Consts':>8} | {'#Viols':>6} | {'Reduced Cost':>12} | {'#Iters':>7}")
        logger.info(f"{' ':>12} | "+
            f"{format_number(obj_val, width=12, precision=5)} | "
            f"{format_number(E_eps_B, width=12, precision=5)} | "
            f"{n_constraints:>8} | "
            f"{n_violations:>6} | "
            f"{format_number(reduced_cost, width=12, precision=6)} | "
            f"{n_iters:>7}"
        )
        logger.info("-" * 90)
        logger.info(" ")
        header_time = f"{'Time (s)':>12} | {'Aggregate':>12} | {'Average':>10}"
        logger.info(header_time)
        logger.info("-" * (len(header_time) +3))
        logger.info(
            f"{'Pricing':>12} | "
            f"{format_number(p.sum(), width=12, precision=3)} | "
            f"{format_number(p.mean(), width=12, precision=3)}"
        )
        logger.info(
            f"{'Master':>12} | "
            f"{format_number(m.sum(), width=12, precision=3)} | "
            f"{format_number(m.mean(), width=12, precision=3)}"
        )
        logger.info(
            f"{'Total':>12} | "
            f"{format_number(total_time, width=12, precision=3)} | "
            f"{format_number(total_time, width=12, precision=3)}"
        )
        logger.info(" ")
