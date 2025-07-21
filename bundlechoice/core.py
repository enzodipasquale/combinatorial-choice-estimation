from .config import DimensionsConfig, RowGenConfig, SubproblemConfig, load_config
from .data_manager import DataManager
from .feature_manager import FeatureManager
from .subproblems import SUBPROBLEM_REGISTRY
from .subproblems.subproblem_manager import SubproblemManager, SubproblemProtocol
from mpi4py import MPI
from typing import Optional, Callable
from bundlechoice.utils import get_logger
from bundlechoice.compute_estimator.row_generation import RowGenerationSolver
logger = get_logger(__name__)

class BundleChoice:
    """
    Main orchestrator for modular bundle choice estimation.

    Provides a clean API for distributed bundle choice estimation with MPI.
    Users must explicitly load configuration, data, and features in sequence before
    solving subproblems. All distributed computation is handled transparently.

    Typical usage:
        bc = BundleChoice()
        bc.load_config(cfg)
        bc.load_data(data, scatter=True)
        bc.build_feature_oracle_from_data()
        results = bc.init_and_solve_subproblems(lambda_k)
    """
    def __init__(self):
        """
        Initialize an empty BundleChoice instance.
        All configuration, data, and features must be loaded explicitly via the
        provided methods. MPI communicator is automatically initialized.
        """
        self.dimensions_cfg = None
        self.subproblem_cfg = None
        self.get_features = None
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.comm_size = self.comm.Get_size()
        self.data_manager = None
        self.feature_manager = None
        self.subproblem: SubproblemProtocol | None = None
        self.subproblem_manager = None

    # --- Configuration ---
    def load_config(self, cfg: dict):
        """
        Load configuration dictionaries for dimensions, row generation, and subproblem.

        Args:
            cfg: Dictionary with keys 'dimensions', 'rowgen', and 'subproblem'.
                - 'dimensions': Specifies num_agents, num_items, num_features, num_simuls
                - 'subproblem': Specifies subproblem type and solver settings
                - 'rowgen': (Optional) Row generation settings
        Returns:
            self for method chaining
        """
        load_config(self, cfg)
        return self

    # --- Data Management ---
    def load_data(self, data: dict, scatter: bool = False):
        """
        Load input data and optionally scatter it across MPI ranks.

        Args:
            data: Dictionary containing agent_data, item_data, errors, etc.
            scatter: If True, immediately scatter data using MPI for distributed processing
        Returns:
            self for method chaining
        """
        self._try_init_data_manager()
        if self.data_manager is None:
            raise RuntimeError("DataManager is not initialized. Call load_config first.")
        self.data_manager.load_input_data(data)
        if scatter:
            self.scatter_data()
        return self

    def scatter_data(self):
        """
        Distribute input data across MPI ranks using the DataManager.
        This method must be called before solving subproblems to ensure
        each rank has access to its local portion of the data.
        Raises:
            RuntimeError: If DataManager is not initialized.
        """
        if self.data_manager is None:
            raise RuntimeError("DataManager is not initialized. Call load_config and load_data first.")
        self.data_manager.scatter_data()

    # --- Feature Management ---
    def load_features_oracle(self, features_oracle):
        """
        Load a user-supplied feature extraction function.

        Args:
            features_oracle: Callable (i_id, B_j, data) -> np.ndarray
                Function that extracts features for agent i_id with bundle B_j
        Returns:
            self for method chaining
        """
        self._try_init_feature_manager()
        self.feature_manager.features_oracle = features_oracle
        return self

    def build_feature_oracle_from_data(self):
        """
        Dynamically build and load a feature extraction function based on input_data structure.
        This method analyzes the input_data to automatically construct a feature
        extraction function that works with the provided data format.
        Returns:
            self for method chaining
        Raises:
            RuntimeError: If input_data or feature_manager is not initialized.
        """
        if self.input_data is None:
            raise RuntimeError("input_data must be set before calling build_feature_oracle_from_data().")
        self._try_init_feature_manager()
        self.feature_manager.build_feature_oracle_from_data()
        return self

    # --- Subproblem Solving ---
    def init_and_solve_subproblems(self, lambda_k):
        """
        Solve the subproblem offline using the current parameters.
        Handles the complete distributed workflow:
            1. Initializes subproblem manager if needed
            2. Initializes local subproblems on each rank
            3. Solves subproblems with the given parameters
            4. Gathers results at rank 0
        Args:
            lambda_k: Parameter vector for the subproblem (numpy array)
        Returns:
            At rank 0: numpy array of shape (num_agents * num_simuls, num_items)
            At other ranks: None
        Raises:
            RuntimeError: If subproblem_manager cannot be initialized.
        """
        if self.subproblem_manager is None:
            self._try_init_subproblem_manager()
        return self.subproblem_manager.init_and_solve_subproblems(lambda_k)

    # --- Properties: Data Access ---
    @property
    def input_data(self):
        """
        Return the input_data dictionary from the DataManager (or None if not initialized).
        """
        if self.data_manager is not None:
            return self.data_manager.input_data
        return None

    @property
    def local_data(self):
        """
        Return the local_data dictionary from the DataManager (or None if not initialized).
        """
        if self.data_manager is not None:
            return self.data_manager.local_data
        return None

    # --- Properties: Dimensions ---
    @property
    def num_agents(self):
        return self.dimensions_cfg.num_agents if self.dimensions_cfg else None

    @property
    def num_items(self):
        return self.dimensions_cfg.num_items if self.dimensions_cfg else None

    @property
    def num_features(self):
        return self.dimensions_cfg.num_features if self.dimensions_cfg else None

    @property
    def num_simuls(self):
        return self.dimensions_cfg.num_simuls if self.dimensions_cfg else None

    # --- Private Helpers ---
    def _try_init_data_manager(self):
        """
        Initialize the DataManager if both dimensions_cfg is set and not already initialized.
        Sets self.data_manager or None.
        """
        if self.dimensions_cfg is not None:
            self.data_manager = DataManager(
                dimensions_cfg=self.dimensions_cfg,
                comm=self.comm,
                input_data=None
            )
        else:
            self.data_manager = None

    def _try_init_feature_manager(self):
        """
        Initialize the FeatureManager if dimensions_cfg is set.
        Sets self.feature_manager or None.
        """
        if self.dimensions_cfg is not None:
            self.feature_manager = FeatureManager(
                dimensions_cfg=self.dimensions_cfg,
                comm=self.comm,
                data_manager=self.data_manager
            )
        else:
            self.feature_manager = None

    def _try_init_subproblem_manager(self):
        """
        Initialize the subproblem manager if subproblem_cfg is set.
        Sets self.subproblem_manager and self.subproblem or None.
        """
        if self.data_manager is None or self.feature_manager is None or self.subproblem_cfg is None:
            raise RuntimeError("DataManager, FeatureManager, and SubproblemConfig must be set before initializing subproblem manager.")
            return
        self.subproblem_manager = SubproblemManager(
            dimensions_cfg=self.dimensions_cfg,
            comm=self.comm,
            data_manager=self.data_manager,
            feature_manager=self.feature_manager,
            subproblem_cfg=self.subproblem_cfg
        )
        self.subproblem = self.subproblem_manager.load()

    def _try_init_rowgeneration_manager(self):
        """
        Initialize the RowGenerationSolver if not already present.
        Returns a RowGenerationSolver instance using explicit dependencies.
        """

        self.rowgeneration_manager = RowGenerationSolver(
            comm=self.comm,
            dimensions_cfg=self.dimensions_cfg,
            rowgen_cfg=self.rowgen_cfg,
            data_manager=self.data_manager,
            feature_manager=self.feature_manager,
            subproblem_manager=self.subproblem_manager
        )
        return self.rowgeneration_manager

    def compute_estimator_row_gen(self):
        """
        Compute the estimator using the RowGenerationSolver.
        """
        self._try_init_subproblem_manager()
        self._try_init_rowgeneration_manager()
        return self.rowgeneration_manager.compute_estimator_row_gen()   

