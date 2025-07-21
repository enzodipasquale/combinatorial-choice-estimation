from typing import Protocol, Any, cast, Optional, List, Union
from .subproblem_registry import SUBPROBLEM_REGISTRY
from .base import BatchSubproblemBase
import numpy as np
from bundlechoice.config import DimensionsConfig, SubproblemConfig
from bundlechoice.data_manager import DataManager
from bundlechoice.feature_manager import FeatureManager
from mpi4py import MPI
from bundlechoice.base import HasDimensions, HasData, HasComm

class BatchSubproblemProtocol(Protocol):
    solve_all_local_problems: bool
    def initialize(self) -> Any: ...
    def solve(self, lambda_k: Any) -> Any: ...

class SerialSubproblemProtocol(Protocol):
    solve_all_local_problems: bool
    def initialize(self, local_id: int) -> Any: ...
    def solve(self, local_id: int, lambda_k: Any, pb: Any = None) -> Any: ...

SubproblemType = Union[BatchSubproblemProtocol, SerialSubproblemProtocol]

class SubproblemManager(HasDimensions, HasComm, HasData):
    """
    Manages subproblem initialization and solving for both batch (vectorized) and serial (per-agent) solvers.
    Provides a unified interface for initializing and solving subproblems across MPI ranks.
    """
    def __init__(self, dimensions_cfg: DimensionsConfig, comm: MPI.Comm, data_manager: DataManager, feature_manager: FeatureManager, subproblem_cfg: SubproblemConfig):
        self.dimensions_cfg = dimensions_cfg
        self.comm = comm
        self.data_manager = data_manager
        self.feature_manager = feature_manager
        self.subproblem_cfg = subproblem_cfg
        self.subproblem: Optional[SubproblemType] = None
        self.local_subproblems: Optional[Any] = None
        # self.rank and self.comm_size now provided by HasComm

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
            subproblem_cfg=self.subproblem_cfg
        )
        self.subproblem = cast(SubproblemType, subproblem_instance)
        return self.subproblem

    def initialize_local(self) -> Any:
        """
        Initialize local subproblems for all local agents (serial) or all at once (batch).

        Returns:
            None for batch subproblems, or list of local subproblems for serial.
        Raises:
            RuntimeError: If subproblem is not initialized or missing methods.
        """
        if self.subproblem is None:
            raise RuntimeError("Subproblem is not initialized.")
        if isinstance(self.subproblem, BatchSubproblemBase):
            self.local_subproblems = None
            # Batch: initialize expects no arguments
            self.subproblem.initialize()
            return None
        else:
            # Serial: initialize expects local_id
            self.local_subproblems = [self.subproblem.initialize(local_id) for local_id in range(self.num_local_agents)]
            return self.local_subproblems

    def solve_local(self, lambda_k: Any) -> Any:
        """
        Solve all local subproblems for the current rank.

        Args:
            lambda_k (Any): Parameter vector for subproblem solving.
        Returns:
            list or batch result: List of results (serial) or batch result (batch).
        Raises:
            RuntimeError: If subproblem or local subproblems are not initialized.
        """
        if self.subproblem is None:
            raise RuntimeError("Subproblem is not initialized.")
        if self.data_manager is None or not hasattr(self.data_manager, 'num_local_agents'):
            raise RuntimeError("DataManager or num_local_agents is not initialized.")
   
        if isinstance(self.subproblem, BatchSubproblemBase):
            return self.subproblem.solve(lambda_k)
        else:
            if self.local_subproblems is None:
                raise RuntimeError("local_subproblems is not initialized for serial subproblem.")
            return [self.subproblem.solve(local_id, lambda_k, pb) for local_id, pb in enumerate(self.local_subproblems)]

    def init_and_solve(self, lambda_k: Any) -> Optional[Any]:
        """
        Initialize and solve local subproblems, then gather results at rank 0.

        Args:
            lambda_k (Any): Parameters for subproblem solving.
        Returns:
            np.ndarray or None: Gathered results at rank 0, None at other ranks.
        Raises:
            RuntimeError: If MPI communicator is not set.
        """
        if self.comm is None:
            raise RuntimeError("MPI communicator (comm) is not set in SubproblemManager.")
        self.initialize_local()
        local_results = self.solve_local(lambda_k)
        gathered = self.comm.gather(local_results, root=0)
        if self.rank == 0:
            return np.concatenate(gathered)
        else:
            return None

    def find_max_bundle_bruteforce(self, lambda_k: Any) -> Optional[Any]:
        """
        Find the maximum bundle value for each local agent using brute force.
        Iterates over all possible bundles (2^num_items combinations) for each local agent.
        Gathers results at rank 0.

        Args:
            lambda_k (Any): Parameter vector for computing bundle values.
        Returns:
            tuple or None: At rank 0, tuple (max_values, best_bundles). At other ranks, (None, None).
        Raises:
            RuntimeError: If num_items is not set.
        """
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

SubproblemProtocol = SubproblemType 