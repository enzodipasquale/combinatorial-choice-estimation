from typing import Protocol, Any, cast, Optional, List, Union
from .subproblem_registry import SUBPROBLEM_REGISTRY
from .base import BatchSubproblemBase, SerialSubproblemBase
import numpy as np
from bundlechoice.config import DimensionsConfig, SubproblemConfig
from bundlechoice.data_manager import DataManager
from bundlechoice.feature_manager import FeatureManager
from mpi4py import MPI
from bundlechoice.base import HasDimensions, HasData, HasComm

class BatchSubproblemProtocol(Protocol):
    solve_all_local_problems: bool
    def initialize(self) -> Any: ...
    def solve(self, theta: Any) -> Any: ...

class SerialSubproblemProtocol(Protocol):
    solve_all_local_problems: bool
    def initialize(self, local_id: int) -> Any: ...
    def solve(self, local_id: int, theta: Any, pb: Any = None) -> Any: ...

SubproblemType = Union[BatchSubproblemProtocol, SerialSubproblemProtocol]

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
        self.demand_oracle: Optional[SubproblemType] = None

    def load(self, subproblem: Optional[Union[str, Any]] = None) -> SubproblemType:
        """
        Load and instantiate the subproblem class from the registry or directly.

        Args:
            subproblem (str, class, or callable, optional): Name, class, or callable for the subproblem. If None, uses subproblem_cfg.name.
        Returns:
            SubproblemType: Instantiated subproblem object.
        Raises:
            ValueError: If subproblem is unknown or invalid.
        """
        if subproblem is None:
            subproblem = getattr(self.subproblem_cfg, 'name', None)
        if isinstance(subproblem, str):
            subproblem_cls = SUBPROBLEM_REGISTRY.get(subproblem)
            if subproblem_cls is None:
                raise ValueError(f"Unknown subproblem: {subproblem}")
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
        self.demand_oracle = cast(SubproblemType, subproblem_instance)
        return self.demand_oracle

    def initialize_local(self) -> Any:
        """
        Initialize local subproblems for all local agents (serial) or all at once (batch).

        Returns:
            None for batch subproblems, or list of local subproblems for serial.
        Raises:
            RuntimeError: If subproblem is not initialized or missing methods.
        """
        if self.demand_oracle is None:
            raise RuntimeError("Subproblem is not initialized.")
        if isinstance(self.demand_oracle, BatchSubproblemBase):
            # Batch: initialize expects no arguments
            self.demand_oracle.initialize()
            return None
        elif isinstance(self.demand_oracle, SerialSubproblemBase):
            # Serial: initialize expects local_id
            self.local_subproblems = [self.demand_oracle.initialize(id) for id in range(self.num_local_agents)]


    def solve_local(self, theta: Any) -> Any:
        """
        Solve all local subproblems for the current rank.

        Args:
            theta (Any): Parameter vector for subproblem solving.
        Returns:
            list or batch result: List of results (serial) or batch result (batch).
        Raises:
            RuntimeError: If subproblem or local subproblems are not initialized.
        """
        if self.demand_oracle is None:
            raise RuntimeError("Subproblem is not initialized.")
        if self.data_manager is None or not hasattr(self.data_manager, 'num_local_agents'):
            raise RuntimeError("DataManager or num_local_agents is not initialized.")
   
        if isinstance(self.demand_oracle, BatchSubproblemBase):
            return self.demand_oracle.solve(theta)
        elif isinstance(self.demand_oracle, SerialSubproblemBase):
            if self.local_subproblems is None:
                raise RuntimeError("local_subproblems is not initialized for serial subproblem.")
            return self.demand_oracle.solve_all(theta, self.local_subproblems)

    def init_and_solve(self, theta: Any, return_values: bool = False) -> Optional[Any]:
        """
        Initialize and solve local subproblems, then gather results at rank 0.

        Args:
            theta (Any): Parameters for subproblem solving.
        Returns:
            np.ndarray or None: Gathered results at rank 0, None at other ranks.
        Raises:
            RuntimeError: If MPI communicator is not set.
        """
        if self.comm_manager is None:
            raise RuntimeError("Communication manager is not set in SubproblemManager.")
        # Broadcast theta from rank 0 to all ranks if available
        if self.is_root() and theta is not None:
            theta = self.comm_manager.broadcast_from_root(theta, root=0)
        else:
            theta = self.comm_manager.broadcast_from_root(None, root=0)
        self.initialize_local()
        local_results = self.solve_local(theta)
        gathered = self.comm_manager.concatenate_at_root(local_results, root=0)
        
        if return_values:
            if self.is_root():
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    features = self.feature_manager.get_local_agents_features(local_results)
                    errors = (self.data_manager.local_data["errors"]* local_results).sum(1)
                    utilities = features @ theta + errors
                    utilities = self.comm_manager.concatenate_at_root(utilities, root=0)
            else:
                utilities = None
            return gathered, utilities
        return gathered



    def brute_force_at_0(self, theta: Any) -> Optional[Any]:
        """
        Find the maximum bundle value for each agent using brute force, only at rank 0.
        Uses global input data without MPI parallelism.

        Args:
            theta (Any): Parameter vector for computing bundle values.
        Returns:
            tuple or None: At rank 0, tuple (max_values, best_bundles). At other ranks, None.
        Raises:
            RuntimeError: If num_items is not set or data_manager is not available.
        """
        if not self.is_root():
            return None, None
        from itertools import product
        num_agents = self.num_agents
        num_items = self.num_items
        if num_items is None:
            raise RuntimeError("num_items is not set in dimensions_cfg.")
        if self.data_manager is None or self.data_manager.input_data is None:
            raise RuntimeError("data_manager or input_data is not available.")
            
        input_data = self.data_manager.input_data
        max_values = np.zeros(num_agents)
        best_bundles = np.zeros((num_agents, num_items), dtype=bool)
        all_bundles = list(product([0, 1], repeat=num_items))
        
        for agent_id in range(num_agents):
            max_value = float('-inf')
            best_bundle = None
            for bundle_tuple in all_bundles:
                bundle = np.array(bundle_tuple, dtype=bool)
                features = self.feature_manager.get_features(agent_id, bundle, input_data)
                error = input_data["errors"][agent_id] @ bundle
                bundle_value = features @ theta + error
                if bundle_value > max_value:
                    max_value = bundle_value
                    best_bundle = bundle.copy()
            max_values[agent_id] = max_value
            best_bundles[agent_id] = best_bundle
            
        return best_bundles, max_values

    def brute_force(self, theta: Any) -> Optional[Any]:
        """
        Find the maximum bundle value for each local agent using brute force.
        Iterates over all possible bundles (2^num_items combinations) for each local agent.
        Gathers results at rank 0.

        Args:
            theta (Any): Parameter vector for computing bundle values.
        Returns:
            tuple or None: At rank 0, tuple (max_values, best_bundles). At other ranks, (None, None).
        Raises:
            RuntimeError: If num_items is not set.
        """
        # Broadcast theta from rank 0 to all ranks if available
        if self.is_root() and theta is not None:
            theta = self.comm_manager.broadcast_from_root(theta, root=0)
        else:
            theta = self.comm_manager.broadcast_from_root(None, root=0)
        
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
                features = self.feature_manager.get_features(local_id, bundle, self.local_data)
                error = self.local_data["errors"][local_id] @ bundle
                bundle_value = features @ theta + error
                if bundle_value > max_value:
                    max_value = bundle_value
                    best_bundle = bundle.copy()
            max_values[local_id] = max_value
            best_bundles[local_id] = best_bundle
        all_max_values = self.comm_manager.concatenate_at_root(max_values, root=0)
        all_best_bundles = self.comm_manager.concatenate_at_root(best_bundles, root=0)
        return all_best_bundles, all_max_values

SubproblemProtocol = SubproblemType 