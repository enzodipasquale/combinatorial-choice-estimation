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

        logger.info(" ")
        logger.info(" ROW GENERATION SUMMARY")
        logger.info("-" * 80)
        param_labels = ' | '.join(f'{f"θ[{i}]":>12}' for i in idx)
        logger.info(f"{'Parameters':>17} | {param_labels}")
        param_vals = ' | '.join(format_number(self.theta_hat[i], width=12, precision=5) for i in idx)
        logger.info(f"{'':>17} | {param_vals}")
        logger.info("-" * 80)
        logger.info(f"{'ObjVal':>12} | {'#Consts':>8} | {'#Viols':>6} | {'Reduced Cost':>12} | {'Time (s)':>9} | {'#Iters':>7}")
        logger.info(
            f"{format_number(obj_val, width=12, precision=5)} | "
            f"{n_constraints:>8} | "
            f"{n_violations:>6} | "
            f"{format_number(reduced_cost, width=12, precision=6)} | "
            f"{format_number(total_time, width=9, precision=3)} | "
            f"{n_iters:>7}"
        )
        logger.info("-" * 80)
        logger.info(f"{'Time':>17} | {'Total (s)':>10} | {'Avg (s)':>10} | {'Range (s)':>11}")
        logger.info(
            f"{'pricing':>17} | "
            f"{format_number(p.sum(), width=10, precision=3)} | "
            f"{format_number(p.mean(), width=10, precision=3)} | "
            f"[{format_number(p.min(), precision=3)}, {format_number(p.max(), precision=3)}]"
        )
        logger.info(
            f"{'master':>17} | "
            f"{format_number(m.sum(), width=10, precision=3)} | "
            f"{format_number(m.mean(), width=10, precision=3)} | "
            f"[{format_number(m.min(), precision=3)}, {format_number(m.max(), precision=3)}]"
        )
        logger.info(" ")
