from typing import Any, cast, Optional, Dict, List, Tuple, Union
from .subproblem_registry import SUBPROBLEM_REGISTRY
from .base import BaseSubproblem
import numpy as np
from numpy.typing import NDArray
from bundlechoice.config import DimensionsConfig, SubproblemConfig
from bundlechoice.data_manager import DataManager
from bundlechoice.feature_manager import FeatureManager
from bundlechoice.base import HasDimensions, HasData, HasComm

class SubproblemManager(HasDimensions, HasComm, HasData):
    """
    Manages subproblem initialization and solving for both batch (vectorized) and serial (per-agent) solvers.
    Provides a unified interface for initializing and solving subproblems across MPI ranks.
    """
    def __init__(self, dimensions_cfg: DimensionsConfig, comm_manager, data_manager: DataManager, feature_manager: FeatureManager, subproblem_cfg: SubproblemConfig):
        self.dimensions_cfg = dimensions_cfg
        self.comm_manager = comm_manager
        self.data_manager = data_manager
        self.feature_manager = feature_manager
        self.subproblem_cfg = subproblem_cfg
        self.demand_oracle: Optional[BaseSubproblem] = None
        self._solve_times = []
        self._cache_enabled = False
        self._result_cache = {}

    def load(self, subproblem: Optional[Union[str, type]] = None) -> BaseSubproblem:
        """
        Load and instantiate the subproblem class from the registry or directly.

        Args:
            subproblem (str, class, or callable, optional): Name, class, or callable for the subproblem. If None, uses subproblem_cfg.name.
        Returns:
            BaseSubproblem: Instantiated subproblem object.
        Raises:
            ValueError: If subproblem is unknown or invalid.
        """
        if subproblem is None:
            subproblem = getattr(self.subproblem_cfg, 'name', None)
        if isinstance(subproblem, str):
            subproblem_cls = SUBPROBLEM_REGISTRY.get(subproblem)
            if subproblem_cls is None:
                available = ', '.join(SUBPROBLEM_REGISTRY.keys())
                raise ValueError(
                    f"Unknown subproblem algorithm: '{subproblem}'\n\n"
                    f"Available algorithms:\n" +
                    "\n".join(f"  - {name}" for name in SUBPROBLEM_REGISTRY.keys()) +
                    "\n\nTo use: bc.subproblems.load('{list(SUBPROBLEM_REGISTRY.keys())[0]}')\n"
                    "Or provide a custom class inheriting from BaseSubproblem."
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
        self.demand_oracle = cast(BaseSubproblem, subproblem_instance)
        return self.demand_oracle

    def initialize_local(self) -> Optional[List[Any]]:
        """
        Initialize local subproblems for all local agents (serial) or all at once (batch).

        Returns:
            None for batch subproblems, or list of local subproblems for serial.
        Raises:
            RuntimeError: If subproblem is not initialized.
        """
        if self.demand_oracle is None:
            raise RuntimeError("Subproblem is not initialized.")
        
        self.local_subproblems = self.demand_oracle.initialize_all()
        return self.local_subproblems


    def solve_local(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Solve all local subproblems for the current rank.

        Args:
            theta: Parameter vector for subproblem solving.
        Returns:
            Array of bundle solutions for local agents.
        Raises:
            RuntimeError: If subproblem or local subproblems are not initialized.
        """
        if self.demand_oracle is None:
            raise RuntimeError("Subproblem is not initialized.")
        if self.data_manager is None or not hasattr(self.data_manager, 'num_local_agents'):
            raise RuntimeError("DataManager or num_local_agents is not initialized.")
        
        # Check cache
        if self._cache_enabled:
            theta_key = theta.tobytes()
            if theta_key in self._result_cache:
                return self._result_cache[theta_key]
        
        # Solve with timing
        import time
        t0 = time.time()
        result = self.demand_oracle.solve_all(theta, self.local_subproblems)
        elapsed = time.time() - t0
        self._solve_times.append(elapsed)
        
        # Cache result
        if self._cache_enabled:
            self._result_cache[theta_key] = result
        
        return result

    def init_and_solve(self, theta: NDArray[np.float64], return_values: bool = False) -> Optional[Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]]:
        """
        Initialize and solve local subproblems, then gather results at rank 0.

        Args:
            theta: Parameters for subproblem solving.
            return_values: If True, also return utility values.
        Returns:
            At rank 0: bundles array, or (bundles, utilities) tuple if return_values=True.
            At other ranks: None.
        Raises:
            RuntimeError: If MPI communicator is not set.
        """
        if self.comm_manager is None:
            raise RuntimeError("Communication manager is not set in SubproblemManager.")
        
        # Broadcast theta using fast buffer-based method (2-7x faster)
        if self.is_root():
            theta = np.asarray(theta, dtype=np.float64)
        else:
            theta = np.empty(self.num_features, dtype=np.float64)
        theta = self.comm_manager.broadcast_array(theta, root=0)
        self.initialize_local()
        local_bundles = self.solve_local(theta)
        bundles = self.comm_manager.concatenate_array_at_root_fast(local_bundles, root=0)
        if return_values:
            utilities = self.feature_manager.compute_gathered_utilities(local_bundles, theta)
            return bundles, utilities
        else:
            return bundles

    def brute_force(self, theta: NDArray[np.float64]) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
        """
        Find the maximum bundle value for each local agent using brute force.
        Iterates over all possible bundles (2^num_items combinations) for each local agent.
        Gathers results at rank 0.

        Args:
            theta: Parameter vector for computing bundle values.
        Returns:
            At rank 0: tuple (best_bundles, max_values).
            At other ranks: (None, None).
        Raises:
            RuntimeError: If num_items is not set.
        """
        # Broadcast theta using fast buffer-based method (2-7x faster)
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
        for agent_id in range(num_local_agents):
            max_value = float('-inf')
            best_bundle = None
            for bundle_tuple in all_bundles:
                bundle = np.array(bundle_tuple, dtype=bool)
                features = self.feature_manager.features_oracle(agent_id, bundle, self.local_data)
                error = self.local_data["errors"][agent_id] @ bundle
                bundle_value = features @ theta + error
                if bundle_value > max_value:
                    max_value = bundle_value
                    best_bundle = bundle.copy()
            max_values[agent_id] = max_value
            best_bundles[agent_id] = best_bundle
        all_max_values = self.comm_manager.concatenate_array_at_root_fast(max_values, root=0)
        all_best_bundles = self.comm_manager.concatenate_array_at_root_fast(best_bundles, root=0)
        return all_best_bundles, all_max_values
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get solving statistics for profiling.
        
        Returns:
            Dictionary with statistics including num_solves, total_time, mean_time, max_time.
        """
        if not self._solve_times:
            return {
                'num_solves': 0,
                'total_time': 0.0,
                'mean_time': 0.0,
                'max_time': 0.0,
            }
        
        return {
            'num_solves': len(self._solve_times),
            'total_time': sum(self._solve_times),
            'mean_time': np.mean(self._solve_times),
            'max_time': max(self._solve_times),
        }
    
    def enable_cache(self):
        """Enable result caching for repeated solves (useful for sensitivity analysis)."""
        self._cache_enabled = True
    
    def disable_cache(self):
        """Disable result caching and clear cache."""
        self._cache_enabled = False
        self._result_cache.clear()
    
    def clear_stats(self):
        """Clear solve time statistics."""
        self._solve_times.clear()





