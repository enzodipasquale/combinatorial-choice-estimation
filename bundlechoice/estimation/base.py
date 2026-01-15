import numpy as np
from typing import Optional, Tuple, Dict, List, Any, TYPE_CHECKING
from numpy.typing import NDArray
from bundlechoice.utils import get_logger, extract_theta, suppress_output
from .result import EstimationResult
if TYPE_CHECKING:
    from bundlechoice.comm_manager import CommManager
    from bundlechoice.config import DimensionsConfig, RowGenerationConfig
    from bundlechoice.data_manager import DataManager
    from bundlechoice.oracles_manager import OraclesManager
    from bundlechoice.subproblems.subproblem_manager import SubproblemManager
logger = get_logger(__name__)
THETA_DISPLAY_THRESHOLD = 10
THETA_DISPLAY_ENDS = 5

class BaseEstimationManager:
    row_generation_cfg: Optional['RowGenerationConfig'] = None
    master_model: Any = None
    slack_counter: Optional[Dict] = None
    theta_val: Optional[NDArray[np.float64]] = None
    timing_stats: Optional[Dict[str, Any]] = None

    def __init__(self, comm_manager, dimensions_cfg, data_manager, oracles_manager, subproblem_manager):
        self.comm_manager = comm_manager
        self.dimensions_cfg = dimensions_cfg
        self.data_manager = data_manager
        self.oracles_manager = oracles_manager
        self.subproblem_manager = subproblem_manager
        self.agents_obs_features = self.get_agents_obs_features()
        self.obs_features = self.agents_obs_features.sum(0) if self.agents_obs_features is not None else None

    def get_obs_features(self):
        local_bundles = self.data_manager.local_data.get('obs_bundles')
        agents_obs_features = self.oracles_manager.compute_gathered_features(local_bundles)
        return agents_obs_features.sum(0) if self.comm_manager._is_root() else None

    def get_agents_obs_features(self):
        local_bundles = self.data_manager.local_data.get('obs_bundles')
        agents_obs_features = self.oracles_manager.compute_gathered_features(local_bundles)
        return agents_obs_features if self.comm_manager._is_root() else None

    def compute_obj_and_gradient(self, theta):
        B_local = self.subproblem_manager.solve_local(theta)
        agents_features = self.oracles_manager.compute_gathered_features(B_local)
        utilities = self.oracles_manager.compute_gathered_utilities(B_local, theta)
        if self.comm_manager._is_root():
            obj_value = utilities.sum() - (self.obs_features @ theta).sum()
            gradient = (agents_features.sum(0) - self.obs_features) / self.dimensions_cfg.num_obs
            return (obj_value, gradient)
        return (None, None)

    def objective(self, theta):
        theta = extract_theta(theta)
        if theta.ndim == 0:
            raise ValueError('theta must be 1D array, got scalar or 0D array')
        B_local = self.subproblem_manager.solve_local(theta)
        utilities = self.oracles_manager.compute_gathered_utilities(B_local, theta)
        if self.comm_manager._is_root():
            return utilities.sum() - (self.obs_features @ theta).sum()
        return None

    def obj_gradient(self, theta):
        B_local = self.subproblem_manager.solve_local(theta)
        agents_features = self.oracles_manager.compute_gathered_features(B_local)
        if self._is_root():
            return (agents_features.sum(0) - self.obs_features) / self.num_obs
        return None

    def solve(self):
        raise NotImplementedError('Subclasses must implement the solve method')

    def _check_bounds_hit(self, tolerance=1e-06):
        try:
            from gurobipy import GRB
        except ImportError:
            return {'hit_lower': [], 'hit_upper': [], 'any_hit': False}
        if not self.comm_manager._is_root() or self.master_model is None:
            return {'hit_lower': [], 'hit_upper': [], 'any_hit': False}
        theta = self.master_variables[0]
        hit_lower, hit_upper = ([], [])
        for k in range(self.dimensions_cfg.num_features):
            val = theta[k].X
            lb, ub = (theta[k].LB, theta[k].UB)
            if lb > -GRB.INFINITY and abs(val - lb) < tolerance:
                hit_lower.append(k)
            if ub < GRB.INFINITY and abs(val - ub) < tolerance:
                hit_upper.append(k)
        return {'hit_lower': hit_lower, 'hit_upper': hit_upper, 'any_hit': bool(hit_lower or hit_upper)}

    def _log_bounds_warnings(self, bounds_info):
        warnings_list = []
        if self.comm_manager._is_root() and bounds_info['any_hit']:
            if bounds_info['hit_lower']:
                msg = f"Theta hit LOWER bound at indices: {bounds_info['hit_lower']}"
                logger.warning(msg)
                warnings_list.append(msg)
            if bounds_info['hit_upper']:
                msg = f"Theta hit UPPER bound at indices: {bounds_info['hit_upper']}"
                logger.warning(msg)
                warnings_list.append(msg)
        return warnings_list

    def _setup_gurobi_model(self, gurobi_settings=None):
        import gurobipy as gp
        defaults = {'Method': 0, 'LPWarmStart': 2, 'OutputFlag': 0}
        params = {**defaults, **(gurobi_settings or {})}
        with suppress_output():
            model = gp.Model()
            for param, value in params.items():
                if value is not None:
                    model.setParam(param, value)
        return model

    def _create_result(self, theta, converged, num_iterations, final_objective=None, warnings=None, metadata=None):
        if self.comm_manager._is_root():
            return EstimationResult(theta_hat=theta.copy(), converged=converged, num_iterations=num_iterations, final_objective=final_objective, timing=self.timing_stats, iteration_history=None, warnings=warnings or [], metadata=metadata or {})
        return EstimationResult(theta_hat=theta.copy(), converged=converged, num_iterations=num_iterations, final_objective=None, timing=None, iteration_history=None, warnings=[], metadata={})

    def _enforce_slack_counter(self):
        if self.row_generation_cfg is None or self.master_model is None:
            return 0
        if self.row_generation_cfg.max_slack_counter >= float('inf'):
            return 0
        if self.slack_counter is None:
            self.slack_counter = {}
        to_remove = []
        for constr in self.master_model.getConstrs():
            if constr.Slack < -1e-06:
                self.slack_counter[constr] = self.slack_counter.get(constr, 0) + 1
                if self.slack_counter[constr] >= self.row_generation_cfg.max_slack_counter:
                    to_remove.append(constr)
            if constr.Pi > 1e-06:
                self.slack_counter.pop(constr, None)
        for constr in to_remove:
            self.master_model.remove(constr)
            self.slack_counter.pop(constr, None)
            self._on_constraint_removed(constr)
        if to_remove:
            logger.info('Removed %d slack constraints', len(to_remove))
        return len(to_remove)

    def _on_constraint_removed(self, constr):
        pass

    def _empty_constraints_dict(self, num_items=None):
        n = num_items or self.dimensions_cfg.num_items
        return {'indices': np.array([], dtype=np.int64), 'bundles': np.array([], dtype=np.float64).reshape(0, n)}

    def _log_solve_header(self, method_name, max_iters, tol):
        if not self.comm_manager._is_root():
            return
        lines = ['=' * 70, f'{method_name} SOLVER', '=' * 70]
        lines.append(f'  Agents: {self.dimensions_cfg.num_obs}, Features: {self.dimensions_cfg.num_features}')
        lines.append(f'  Max iters: {max_iters}, Tolerance: {tol:.1e}')
        logger.info('\n'.join(lines))

    def _expand_bounds(self, bound, fill_value):
        arr = np.full(self.dimensions_cfg.num_features, fill_value, dtype=np.float64)
        if bound is None:
            return arr
        if np.isscalar(bound):
            arr[:] = float(bound)
            return arr
        bound_arr = np.asarray(bound, dtype=object)
        if len(bound_arr) != self.dimensions_cfg.num_features:
            raise ValueError('Length of theta bounds does not match number of features.')
        mask = bound_arr != None
        arr[mask] = np.array([float(v) for v in bound_arr[mask]])
        return arr

    def _log_timing_summary(self, timing_stats, obj_val=None, theta=None, header='ESTIMATION SUMMARY'):
        if not self.comm_manager._is_root():
            return
        total_time = timing_stats.get('total_time', 0.0)
        num_iters = timing_stats.get('num_iterations', 0)
        time_per_iter = timing_stats.get('time_per_iter', total_time / num_iters if num_iters > 0 else 0)
        pricing = timing_stats.get('pricing_time', 0.0)
        master = timing_stats.get('master_time', 0.0)
        other = timing_stats.get('other_time', total_time - pricing)
        pricing_pct = timing_stats.get('pricing_pct', 100 * pricing / total_time if total_time > 0 else 0)
        master_pct = timing_stats.get('master_pct', 100 * master / total_time if total_time > 0 else 0)
        other_pct = timing_stats.get('other_pct', 100 * other / total_time if total_time > 0 else 0)
        lines = ['=' * 70, header, '=' * 70]
        if obj_val is not None:
            lines.append(f'Objective value: {obj_val:.6f}')
        if theta is not None:
            if len(theta) <= THETA_DISPLAY_THRESHOLD:
                lines.append(f'Theta: {np.array2string(theta, precision=6, suppress_small=True)}')
            else:
                lines.append(f'Theta (dim={len(theta)}):')
                lines.append(f'  First {THETA_DISPLAY_ENDS}: {np.array2string(theta[:THETA_DISPLAY_ENDS], precision=6, suppress_small=True)}')
                lines.append(f'  Last {THETA_DISPLAY_ENDS}:  {np.array2string(theta[-THETA_DISPLAY_ENDS:], precision=6, suppress_small=True)}')
                lines.append(f'  Min: {theta.min():.6f}, Max: {theta.max():.6f}, Mean: {theta.mean():.6f}')
        lines.append(f'Iterations: {num_iters}')
        lines.append(f'Total time: {total_time:.2f}s ({time_per_iter:.2f}s/iter avg)')
        lines.append('')
        pricing_per = timing_stats.get('pricing_per_iter')
        master_per = timing_stats.get('master_per_iter')
        has_per_iter = pricing_per or master_per
        timing_header = 'Timing:                  Total           [min  /  avg  /  max  per iter]' if has_per_iter else 'Timing:'
        lines.append(timing_header)
        pricing_detail = f"  [{pricing_per['min']:.3f}s / {pricing_per['avg']:.3f}s / {pricing_per['max']:.3f}s]" if pricing_per else ''
        lines.append(f'  Pricing (subproblems): {pricing:5.2f}s ({pricing_pct:5.1f}%){pricing_detail}')
        if master > 0:
            master_detail = f"  [{master_per['min']:.3f}s / {master_per['avg']:.3f}s / {master_per['max']:.3f}s]" if master_per else ''
            lines.append(f'  Master problem:        {master:5.2f}s ({master_pct:5.1f}%){master_detail}')
        lines.append(f'  Other (sync/overhead): {other:5.2f}s ({other_pct:5.1f}%)')
        lines.append('')
        logger.info('\n'.join(lines))

    def log_parameter(self):
        if self.row_generation_cfg is None or self.theta_val is None:
            return
        feature_ids = self.row_generation_cfg.parameters_to_log
        precision = 3
        if feature_ids is not None:
            logger.info('Parameters: %s', np.round(self.theta_val[feature_ids], precision))
        else:
            logger.info('Parameters: %s', np.round(self.theta_val, precision))