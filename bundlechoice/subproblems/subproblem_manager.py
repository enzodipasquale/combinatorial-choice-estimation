"""
Subproblem manager for bundle choice estimation.

Manages initialization and solving of subproblems (batch or serial).
"""

from typing import Any, cast, Optional, Dict, List, Tuple, Union
from .subproblem_registry import SUBPROBLEM_REGISTRY
from .base import BaseSubproblem
import numpy as np
from numpy.typing import NDArray
from bundlechoice.config import DimensionsConfig, SubproblemConfig
from bundlechoice.data_manager import DataManager
from bundlechoice.feature_manager import FeatureManager
from bundlechoice.base import HasDimensions, HasData, HasComm
from bundlechoice.utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# SubproblemManager
# ============================================================================

class SubproblemManager(HasDimensions, HasComm, HasData):
    """Manages subproblem initialization and solving (batch or serial)."""
    
    def __init__(self, dimensions_cfg: DimensionsConfig, comm_manager: Any, 
                 data_manager: DataManager, feature_manager: FeatureManager, 
                 subproblem_cfg: SubproblemConfig) -> None:
        self.dimensions_cfg = dimensions_cfg
        self.comm_manager = comm_manager
        self.data_manager = data_manager
        self.feature_manager = feature_manager
        self.subproblem_cfg = subproblem_cfg
        self.subproblem_instance: Optional[BaseSubproblem] = None

    # ============================================================================
    # Subproblem Loading
    # ============================================================================

    def load(self, subproblem: Optional[Union[str, type]] = None) -> BaseSubproblem:
        """Load and instantiate subproblem from registry or custom class."""
        if subproblem is None and self.subproblem_instance is not None:
            return self.subproblem_instance
        
        if subproblem is None:
            subproblem = getattr(self.subproblem_cfg, 'name', None)
        if isinstance(subproblem, str):
            subproblem_cls = SUBPROBLEM_REGISTRY.get(subproblem)
            if subproblem_cls is None:
                available = ", ".join(SUBPROBLEM_REGISTRY.keys())
                raise ValueError(
                    f"Unknown subproblem algorithm: '{subproblem}'. "
                    f"Available: {available}"
                )
        elif callable(subproblem):
            subproblem_cls = subproblem
        else:
            raise ValueError("Subproblem must be a string or a callable/class.")
        
        subproblem_instance = subproblem_cls(
            data_manager=self.data_manager,
            feature_manager=self.feature_manager,
            subproblem_cfg=self.subproblem_cfg,
            dimensions_cfg=self.dimensions_cfg
        )
        self.subproblem_instance = cast(BaseSubproblem, subproblem_instance)
        return self.subproblem_instance

    # ============================================================================
    # Initialization & Solving
    # ============================================================================

    def initialize_local(self) -> Optional[List[Any]]:
        """Initialize local subproblems (returns None for batch, list for serial)."""
        if self.subproblem_instance is None and self.subproblem_cfg and self.subproblem_cfg.name:
            self.load()
        
        if self.subproblem_instance is None:
            raise RuntimeError("Subproblem is not initialized.")
        
        self.local_subproblems = self.subproblem_instance.initialize_all()
        return self.local_subproblems

    def solve_local(self, theta: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Solve all local subproblems for current rank. Returns bool array of bundles."""
        import sys
        if self.is_root():
            print(f"DEBUG: solve_local: started, num_local_agents={self.num_local_agents}", flush=True)
            sys.stdout.flush()
        
        if self.subproblem_instance is None:
            raise RuntimeError("Subproblem is not initialized.")
        if self.data_manager is None or not hasattr(self.data_manager, 'num_local_agents'):
            raise RuntimeError("DataManager or num_local_agents is not initialized.")
        
        if self.is_root():
            print("DEBUG: solve_local: about to call solve_all", flush=True)
            sys.stdout.flush()
        
        result = self.subproblem_instance.solve_all(theta, self.local_subproblems)
        
        if self.is_root():
            print(f"DEBUG: solve_local: solve_all returned, shape={result.shape if result is not None else None}", flush=True)
            sys.stdout.flush()
        
        return result

    def init_and_solve(self, theta: NDArray[np.float64], return_values: bool = False) -> Optional[Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]]:
        """Initialize and solve local subproblems, then gather results at rank 0."""
        if self.comm_manager is None:
            raise RuntimeError("Communication manager is not set in SubproblemManager.")
        
        if self.subproblem_instance is None and self.subproblem_cfg and self.subproblem_cfg.name:
            self.load()
        
        if self.is_root():
            theta = np.asarray(theta, dtype=np.float64)
        else:
            theta = np.empty(self.num_features, dtype=np.float64)
        theta = self.comm_manager.broadcast_array(theta, root=0)
        self.initialize_local()
        local_bundles = self.solve_local(theta)
        bundles = self.comm_manager.concatenate_array_at_root_fast(local_bundles, root=0)
        
        if self.is_root() and bundles is not None:
            bundle_sizes = bundles.sum(axis=1)
            item_demands = bundles.sum(axis=0)
            num_items = bundles.shape[1]
            num_agents_total = bundles.shape[0]
            max_possible = num_agents_total * num_items
            
            print("=" * 70)
            print("BUNDLE GENERATION STATISTICS")
            print("=" * 70)
            print(f"  Bundle sizes: min={bundle_sizes.min()}, max={bundle_sizes.max()}, mean={bundle_sizes.mean():.2f}, std={bundle_sizes.std():.2f}")
            print(f"  Aggregate demands: min={item_demands.min()}, max={item_demands.max()}, mean={item_demands.mean():.2f}")
            print(f"  Total items selected: {bundles.sum()} out of {max_possible}")
            
            if bundle_sizes.max() == 0:
                print("\n  WARNING: All agents have empty bundles (no items selected)!")
            elif bundle_sizes.min() == num_items:
                print(f"\n  WARNING: All agents have full bundles (all {num_items} items selected)!")
            
            print()
        
        if return_values:
            utilities = self.feature_manager.compute_gathered_utilities(local_bundles, theta)
            return bundles, utilities
        else:
            return bundles

    def brute_force(self, theta: NDArray[np.float64]) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
        """Find maximum bundle value for each local agent using brute force."""
        if self.is_root():
            theta = np.asarray(theta, dtype=np.float64)
        else:
            theta = np.empty(self.num_features, dtype=np.float64)
        theta = self.comm_manager.broadcast_array(theta, root=0)
        
        from itertools import product
        num_local_agents = self.num_local_agents
        num_items = self.num_items
        if num_items is None:
            raise RuntimeError("num_items is not set in dimensions_cfg.")
        max_values = np.zeros(num_local_agents)
        best_bundles = np.zeros((num_local_agents, num_items), dtype=bool)
        all_bundles = list(product([0, 1], repeat=num_items))
        for local_id in range(num_local_agents):
            max_value = float('-inf')
            best_bundle = None
            for bundle_tuple in all_bundles:
                bundle = np.array(bundle_tuple, dtype=bool)
                features = self.feature_manager.features_oracle(local_id, bundle, self.local_data)
                error = self.local_data["errors"][local_id] @ bundle
                bundle_value = features @ theta + error
                if bundle_value > max_value:
                    max_value = bundle_value
                    best_bundle = bundle.copy()
            max_values[local_id] = max_value
            best_bundles[local_id] = best_bundle
        all_max_values = self.comm_manager.concatenate_array_at_root_fast(max_values, root=0)
        all_best_bundles = self.comm_manager.concatenate_array_at_root_fast(best_bundles, root=0)
        return all_best_bundles, all_max_values
    
    # ============================================================================
    # Settings Management
    # ============================================================================
    
    def update_settings(self, settings: Dict[str, Any]) -> None:
        """Update subproblem settings and apply to initialized models."""
        self.subproblem_cfg.settings.update(settings)
        
        if hasattr(self, 'local_subproblems') and self.local_subproblems is not None:
            if isinstance(self.local_subproblems, list):
                import gurobipy as gp
                for model in self.local_subproblems:
                    if model is not None and hasattr(model, 'setParam'):
                        for param_name, value in settings.items():
                            if value is not None:
                                model.setParam(param_name, value)
                            else:
                                if param_name == "TimeLimit":
                                    model.setParam(param_name, gp.GRB.INFINITY)





