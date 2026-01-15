from typing import Any, cast, Optional, Dict, List, Tuple, Union
from .subproblem_registry import SUBPROBLEM_REGISTRY
from .base import BaseSubproblem
import numpy as np
from numpy.typing import NDArray
from bundlechoice.config import DimensionsConfig, SubproblemConfig
from bundlechoice.data_manager import DataManager
from bundlechoice.oracles_manager import OraclesManager
from bundlechoice.utils import get_logger
logger = get_logger(__name__)

class SubproblemManager:

    def __init__(self, dimensions_cfg, comm_manager, data_manager, oracles_manager, subproblem_cfg):
        self.dimensions_cfg = dimensions_cfg
        self.comm_manager = comm_manager
        self.data_manager = data_manager
        self.oracles_manager = oracles_manager
        self.subproblem_cfg = subproblem_cfg
        self.subproblem_instance: Optional[BaseSubproblem] = None

    def load(self, subproblem=None):
        if subproblem is None and self.subproblem_instance is not None:
            return self.subproblem_instance
        if subproblem is None:
            subproblem = getattr(self.subproblem_cfg, 'name', None)
        if isinstance(subproblem, str):
            subproblem_cls = SUBPROBLEM_REGISTRY.get(subproblem)
            if subproblem_cls is None:
                available = ', '.join(SUBPROBLEM_REGISTRY.keys())
                raise ValueError(f"Unknown subproblem algorithm: '{subproblem}'. Available: {available}")
        elif callable(subproblem):
            subproblem_cls = subproblem
        else:
            raise ValueError('Subproblem must be a string or a callable/class.')
        subproblem_instance = subproblem_cls(data_manager=self.data_manager, oracles_manager=self.oracles_manager, subproblem_cfg=self.subproblem_cfg, dimensions_cfg=self.dimensions_cfg)
        self.subproblem_instance = cast(BaseSubproblem, subproblem_instance)
        return self.subproblem_instance

    def initialize_local(self):
        if self.subproblem_instance is None and self.subproblem_cfg and self.subproblem_cfg.name:
            self.load()
        if self.subproblem_instance is None:
            raise RuntimeError('Subproblem is not initialized.')
        self.local_subproblems = self.subproblem_instance.initialize_all()
        return self.local_subproblems

    def solve_local(self, theta):
        if self.subproblem_instance is None:
            raise RuntimeError('Subproblem is not initialized.')
        if self.data_manager is None or not hasattr(self.data_manager, 'num_local_agents'):
            raise RuntimeError('DataManager or num_local_agents is not initialized.')
        return self.subproblem_instance.solve_all(theta, self.local_subproblems)

    def init_and_solve(self, theta, return_values=False):
        if self.comm_manager is None:
            raise RuntimeError('Communication manager is not set in SubproblemManager.')
        if self.subproblem_instance is None and self.subproblem_cfg and self.subproblem_cfg.name:
            self.load()
        if self.comm_manager._is_root():
            theta = np.asarray(theta, dtype=np.float64)
        else:
            theta = np.empty(self.dimensions_cfg.num_features, dtype=np.float64)
        theta = self.comm_manager._broadcast_array(theta, root=0)
        self.initialize_local()
        local_bundles = self.solve_local(theta)
        bundles = self.comm_manager._gather_array_by_row(local_bundles, root=0)
        if self.comm_manager._is_root() and bundles is not None:
            bundle_sizes = bundles.sum(axis=1)
            item_demands = bundles.sum(axis=0)
            num_items = bundles.shape[1]
            num_obs_total = bundles.shape[0]
            max_possible = num_obs_total * num_items
            lines = ['=' * 70, 'BUNDLE GENERATION STATISTICS', '=' * 70]
            lines.append(f'  Bundle sizes: min={bundle_sizes.min()}, max={bundle_sizes.max()}, mean={bundle_sizes.mean():.2f}, std={bundle_sizes.std():.2f}')
            lines.append(f'  Aggregate demands: min={item_demands.min()}, max={item_demands.max()}, mean={item_demands.mean():.2f}')
            lines.append(f'  Total items selected: {bundles.sum()} out of {max_possible}')
            if bundle_sizes.max() == 0:
                lines.append('\n  WARNING: All agents have empty bundles (no items selected)!')
            elif bundle_sizes.min() == num_items:
                lines.append(f'\n  WARNING: All agents have full bundles (all {num_items} items selected)!')
            logger.info('\n'.join(lines))
        if return_values:
            utilities = self.oracles_manager.compute_gathered_utilities(local_bundles, theta)
            return (bundles, utilities)
        else:
            return bundles

    def brute_force(self, theta):
        bf_cls = SUBPROBLEM_REGISTRY.get('BruteForce')
        if bf_cls is None:
            raise RuntimeError('BruteForce subproblem not found in registry')
        bf_instance = bf_cls(data_manager=self.data_manager, oracles_manager=self.oracles_manager, subproblem_cfg=self.subproblem_cfg, dimensions_cfg=self.dimensions_cfg)
        if self.comm_manager._is_root():
            theta = np.asarray(theta, dtype=np.float64)
        else:
            theta = np.empty(self.dimensions_cfg.num_features, dtype=np.float64)
        theta = self.comm_manager._broadcast_array(theta, root=0)
        local_subproblems = bf_instance.initialize_all()
        local_bundles = bf_instance.solve_all(theta, local_subproblems)
        bundles = self.comm_manager._gather_array_by_row(local_bundles, root=0)
        values = self.oracles_manager.compute_gathered_utilities(local_bundles, theta)
        return (bundles, values)

    def update_settings(self, settings):
        self.subproblem_cfg.settings.update(settings)
        if hasattr(self, 'local_subproblems') and self.local_subproblems is not None:
            if isinstance(self.local_subproblems, list):
                import gurobipy as gp
                for model in self.local_subproblems:
                    if model is not None and hasattr(model, 'setParam'):
                        for param_name, value in settings.items():
                            if value is not None:
                                model.setParam(param_name, value)
                            elif param_name == 'TimeLimit':
                                model.setParam(param_name, gp.GRB.INFINITY)