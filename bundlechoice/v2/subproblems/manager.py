from typing import Protocol, Any, cast, Optional, List, Union
from .subproblem_registry import SUBPROBLEM_REGISTRY
from .base import BatchSubproblemBase
import numpy as np

class BatchSubproblemProtocol(Protocol):
    solve_all_local_problems: bool
    def initialize(self) -> Any: ...
    def solve(self, lambda_k: Any) -> Any: ...

class SerialSubproblemProtocol(Protocol):
    solve_all_local_problems: bool
    def initialize(self, local_id: int) -> Any: ...
    def solve(self, local_id: int, lambda_k: Any, pb: Any = None) -> Any: ...

SubproblemType = Union[BatchSubproblemProtocol, SerialSubproblemProtocol]

class SubproblemManager:
    """
    Manages subproblem initialization and solving for both batch (vectorized) and serial (per-agent) solvers.
    Solvers should set the class attribute `solve_all_local_problems = True` if they implement a batch interface:
        - initialize(self): prepares all local problems at once
        - solve(self, lambda_k): solves all local problems at once
    Otherwise, the default is per-agent:
        - initialize(self, local_id): prepares a single local problem
        - solve(self, local_id, lambda_k, pb=None): solves a single local problem
    """
    def __init__(
        self,
        data_manager: Any,
        feature_manager: Any,
        subproblem_cfg: Any,
        dimensions_cfg: Optional[Any],
        comm: Any
    ) -> None:
        """
        Initialize the SubproblemManager.
        Args:
            data_manager: DataManager instance providing agent/item data.
            feature_manager: FeatureManager instance for feature extraction.
            subproblem_cfg: Configuration for the subproblem.
            dimensions_cfg: Configuration for problem dimensions (agents, items, features, simuls).
            comm: MPI communicator.
        """
        self.registry = SUBPROBLEM_REGISTRY
        self.data_manager = data_manager
        self.feature_manager = feature_manager
        self.subproblem_cfg = subproblem_cfg
        self.dimensions_cfg = dimensions_cfg if dimensions_cfg is not None else (
            getattr(data_manager, 'dimensions_cfg', None) if data_manager is not None else None)
        self.subproblem: Optional[SubproblemType] = None
        self.local_subproblems: Optional[Any] = None
        self.comm = comm
        self.rank = comm.Get_rank() if comm is not None else 0

    def load(self, subproblem: Optional[Union[str, Any]] = None) -> SubproblemType:
        """
        Load and instantiate the subproblem class from the registry or directly.
        Args:
            subproblem: Name (str), class, or callable for the subproblem. If None, uses subproblem_cfg.name.
        Returns:
            Instantiated subproblem object.
        Raises:
            ValueError: If subproblem is unknown or invalid.
        """
        if subproblem is None:
            subproblem = getattr(self.subproblem_cfg, 'name', None)
        if isinstance(subproblem, str):
            subproblem_cls = self.registry.get(subproblem)
            if subproblem_cls is None:
                raise ValueError(f"Unknown subproblem: {subproblem}")
        elif callable(subproblem):
            subproblem_cls = subproblem
        else:
            raise ValueError("Subproblem must be a string or a callable/class.")
        subproblem_instance = subproblem_cls(
            data_manager=self.data_manager,
            feature_manager=self.feature_manager,
            subproblem_cfg=self.subproblem_cfg
        )
        self.subproblem = cast(SubproblemType, subproblem_instance)
        return self.subproblem

    def init_local_subproblems(self) -> Any:
        """
        Initialize local subproblems for all local agents (serial) or all at once (batch).
        Returns:
            None for batch subproblems, or list of local subproblems for serial.
        Raises:
            RuntimeError: If subproblem is not initialized or missing methods.
        """
        if self.subproblem is None:
            raise RuntimeError("Subproblem is not initialized.")
        if not hasattr(self.subproblem, 'initialize') or not callable(getattr(self.subproblem, 'initialize')):
            raise AttributeError("Subproblem does not have an 'initialize' method.")
        if isinstance(self.subproblem, BatchSubproblemBase):
            self.local_subproblems = None
            # Batch: initialize expects no arguments
            getattr(self.subproblem, 'initialize')()
            return None
        else:
            # Serial: initialize expects local_id
            self.local_subproblems = [getattr(self.subproblem, 'initialize')(local_id) for local_id in range(self.data_manager.num_local_agents)]
            return self.local_subproblems

    def solve_local_subproblems(self, lambda_k: Any) -> Any:
        """
        Solve all local subproblems for the current rank.
        Args:
            lambda_k: Parameter vector for subproblem solving.
        Returns:
            List of results (serial) or batch result (batch).
        Raises:
            RuntimeError: If subproblem or local subproblems are not initialized.
        """
        if self.subproblem is None:
            raise RuntimeError("Subproblem is not initialized.")
        if self.data_manager is None or not hasattr(self.data_manager, 'num_local_agents') or self.data_manager.num_local_agents is None:
            raise RuntimeError("DataManager or num_local_agents is not initialized.")
        if not hasattr(self.subproblem, 'solve') or not callable(getattr(self.subproblem, 'solve')):
            raise AttributeError("Subproblem does not have a 'solve' method.")
        if isinstance(self.subproblem, BatchSubproblemBase):
            return self.subproblem.solve(lambda_k)
        else:
            if self.local_subproblems is None:
                raise RuntimeError("local_subproblems is not initialized for serial subproblem.")
            return [self.subproblem.solve(local_id, lambda_k, pb) for local_id, pb in enumerate(self.local_subproblems)]

    def init_and_solve_subproblems(self, lambda_k: Any) -> Optional[Any]:
        """
        Initialize and solve local subproblems, then gather results at rank 0.
        Args:
            lambda_k: Parameters for subproblem solving.
        Returns:
            np.ndarray of gathered results at rank 0, None at other ranks.
        Raises:
            RuntimeError: If MPI communicator is not set.
        """
        if self.comm is None:
            raise RuntimeError("MPI communicator (comm) is not set in SubproblemManager.")
        self.init_local_subproblems()
        local_results = self.solve_local_subproblems(lambda_k)
        if len(local_results) > 0:
            local_results_array = np.array(local_results)
        else:
            if self.num_items is not None:
                local_results_array = np.array([], dtype=bool).reshape(0, self.num_items)
            else:
                local_results_array = np.array([], dtype=bool)
        gathered = self.comm.gather(local_results_array, root=0)
        if self.rank == 0:
            # non_empty_results = [result for result in gathered if result.size > 0]
            # if non_empty_results:
            #     return np.concatenate(non_empty_results)
            # else:
            #     if self.num_items is not None:
            #         return np.array([], dtype=bool).reshape(0, self.num_items)
            #     else:
            #         return np.array([], dtype=bool)
            return np.concatenate(gathered)
        else:
            return None

    def find_max_bundle_bruteforce(self, lambda_k: Any) -> Optional[Any]:
        """
        Find the maximum bundle value for each local agent using brute force.
        Iterates over all possible bundles (2^num_items combinations) for each local agent.
        Gathers results at rank 0.
        Args:
            lambda_k: Parameter vector for computing bundle values
        Returns:
            At rank 0: tuple (max_values, best_bundles) where:
                - max_values: numpy array of shape (num_agents,) with maximum values
                - best_bundles: numpy array of shape (num_agents, num_items) with best bundles
            At other ranks: None
        Raises:
            RuntimeError: If num_items is not set.
        """
        from itertools import product
        num_local_agents = self.data_manager.num_local_agents
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
                features = self.feature_manager.get_features(local_id, bundle, self.data_manager.local_data)
                error = self.data_manager.local_data["errors"][local_id] @ bundle
                bundle_value = features @ lambda_k + error
                if bundle_value > max_value:
                    max_value = bundle_value
                    best_bundle = bundle.copy()
            max_values[local_id] = max_value
            best_bundles[local_id] = best_bundle
        gathered_max_values = self.comm.gather(max_values, root=0)
        gathered_best_bundles = self.comm.gather(best_bundles, root=0)
        if self.rank == 0:
            all_max_values = np.concatenate(gathered_max_values)
            all_best_bundles = np.concatenate(gathered_best_bundles)
            return all_max_values, all_best_bundles
        else:
            return None, None

    @property
    def num_agents(self) -> Optional[int]:
        """Number of agents in the problem."""
        return self.dimensions_cfg.num_agents if self.dimensions_cfg else None

    @property
    def num_items(self) -> Optional[int]:
        """Number of items in the problem."""
        return self.dimensions_cfg.num_items if self.dimensions_cfg else None

    @property
    def num_features(self) -> Optional[int]:
        """Number of features in the problem."""
        return self.dimensions_cfg.num_features if self.dimensions_cfg else None

    @property
    def num_simuls(self) -> Optional[int]:
        """Number of simulations in the problem."""
        return self.dimensions_cfg.num_simuls if self.dimensions_cfg else None 

SubproblemProtocol = SubproblemType 