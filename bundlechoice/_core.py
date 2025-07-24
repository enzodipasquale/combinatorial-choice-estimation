from .config import DimensionsConfig, RowGenConfig, SubproblemConfig, load_config
from .data_manager import DataManager
from .feature_manager import FeatureManager
from .subproblems import SUBPROBLEM_REGISTRY
from .subproblems.subproblem_manager import SubproblemManager, SubproblemProtocol
from mpi4py import MPI
from typing import Optional, Callable
from bundlechoice.utils import get_logger
from bundlechoice.compute_estimator.row_generation import RowGenerationSolver
from bundlechoice.base import HasDimensions, HasData, HasComm
logger = get_logger(__name__)

class BundleChoice(HasDimensions, HasComm, HasData):
    """
    Main orchestrator for modular bundle choice estimation.

    Provides a clean API for distributed bundle choice estimation with MPI.
    Users must explicitly load configuration, data, and features in sequence before
    solving subproblems. All distributed computation is handled transparently.

    Typical usage::
        bc = BundleChoice()
        bc.load_config(cfg)
        bc.load_data(data, scatter=True)
        bc.build_feature_oracle_from_data()
        results = bc.init_and_solve_subproblems(lambda_k)
    """
    dimensions_cfg: Optional[DimensionsConfig]
    subproblem_cfg: Optional[SubproblemConfig]
    data_manager: Optional[DataManager]
    feature_manager: Optional[FeatureManager]
    subproblem_manager: Optional[SubproblemManager]
    row_generation_manager: Optional[RowGenerationSolver]
    comm: MPI.Comm

    def __init__(self):
        """
        Initialize an empty BundleChoice instance.

        All configuration, data, and features must be loaded explicitly via the
        provided methods. MPI communicator is automatically initialized.
        """
        self.dimensions_cfg = None
        self.subproblem_cfg = None
        self.comm = MPI.COMM_WORLD
        self.data_manager = None
        self.feature_manager = None
        self.subproblem_manager = None
        self.row_generation_manager = None


    def compute_estimator_row_gen(self):
        """
        Compute the estimator using the RowGenerationSolver.

        Returns:
            tuple: (lambda_k_iter, p_j_iter) from RowGenerationSolver.compute_estimator_row_gen().
        """
        self._try_init_subproblem_manager()
        self._try_init_row_generation_manager()
        return self.row_generation_manager.solve()   



    # --- Configuration ---
    def load_config(self, cfg: dict):
        """
        Load configuration dictionaries for dimensions, row generation, and subproblem.

        Args:
            cfg (dict): Dictionary with keys 'dimensions', 'rowgen', and 'subproblem'.
                - 'dimensions': Specifies num_agents, num_items, num_features, num_simuls
                - 'subproblem': Specifies subproblem type and solver settings
                - 'rowgen': (Optional) Row generation settings
        Returns:
            BundleChoice: self for method chaining.
        """
        load_config(self, cfg)
        return self

    # --- Data Management ---
    def load_data(self, data: dict, scatter: bool = False):
        """
        Load input data and optionally scatter it across MPI ranks.

        Args:
            data (dict): Dictionary containing agent_data, item_data, errors, etc.
            scatter (bool): If True, immediately scatter data using MPI for distributed processing.
        Returns:
            BundleChoice: self for method chaining.
        """
        self._try_init_data_manager()
        self.data_manager.load(data)
        if scatter:
            self.scatter_data()
        return self

    def scatter_data(self):
        """
        Distribute input data across MPI ranks using the DataManager.

        Raises:
            RuntimeError: If DataManager is not initialized.
        """
        if self.data_manager is None:
            raise RuntimeError("DataManager is not initialized.")
        self.data_manager.scatter_data()

    # --- Feature Management ---
    def load_features_oracle(self, features_oracle):
        """
        Load a user-supplied feature extraction function.

        Args:
            features_oracle (Callable): Function (i_id, B_j, data) -> np.ndarray
                that extracts features for agent i_id with bundle B_j.
        Returns:
            BundleChoice: self for method chaining.
        """
        self._try_init_feature_manager()
        self.feature_manager.load(features_oracle)
        return self

    def build_feature_oracle_from_data(self):
        """
        Dynamically build and load a feature extraction function based on input_data structure.
        This method analyzes the input_data to automatically construct a feature
        extraction function that works with the provided data format.

        Returns:
            BundleChoice: self for method chaining.
        Raises:
            RuntimeError: If input_data or feature_manager is not initialized.
        """
        if self.input_data is None:
            raise RuntimeError("input_data must be set before calling build_feature_oracle_from_data().")
        self._try_init_feature_manager()
        self.feature_manager.build_from_data()
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
            lambda_k (np.ndarray): Parameter vector for the subproblem.
        Returns:
            np.ndarray or None: At rank 0, numpy array of shape (num_agents * num_simuls, num_items). At other ranks, None.
        Raises:
            RuntimeError: If subproblem_manager cannot be initialized.
        """
        self._try_init_subproblem_manager()
        return self.subproblem_manager.init_and_solve_subproblems(lambda_k)

    # --- Initialization ---
    def _try_init_data_manager(self):
        """
        Initialize the DataManager if dimensions_cfg is set and not already initialized.
        Sets self.data_manager or None.

        Returns:
            DataManager or None: The initialized DataManager instance, or None if not possible.
        """
        if self.dimensions_cfg is not None:
            self.data_manager = DataManager(
                dimensions_cfg=self.dimensions_cfg,
                comm=self.comm
            )
        else:
            logger.error("dimensions_cfg must be set before initializing data manager.")
            self.data_manager = None
        return self.data_manager
    def _try_init_feature_manager(self):
        """
        Initialize the FeatureManager if dimensions_cfg is set.
        Sets self.feature_manager or None.

        Returns:
            FeatureManager or None: The initialized FeatureManager instance, or None if not possible.
        """
        if self.dimensions_cfg is not None:
            self.feature_manager = FeatureManager(
                dimensions_cfg=self.dimensions_cfg,
                comm=self.comm,
                data_manager=self.data_manager
            )
        else:
            logger.error("dimensions_cfg must be set before initializing feature manager.")
            self.feature_manager = None
        return self.feature_manager
    def _try_init_subproblem_manager(self):
        """
        Initialize the subproblem manager if subproblem_cfg is set.
        Sets self.subproblem_manager.

        Returns:
            SubproblemManager: The initialized SubproblemManager instance.
        Raises:
            RuntimeError: If required managers or configs are not set.
        """
        if self.data_manager is None or self.feature_manager is None or self.subproblem_cfg is None:
            raise RuntimeError("DataManager, FeatureManager, and SubproblemConfig must be set before initializing subproblem manager.")
            return None

        self.subproblem_manager = SubproblemManager(
            dimensions_cfg=self.dimensions_cfg,
            comm=self.comm,
            data_manager=self.data_manager,
            feature_manager=self.feature_manager,
            subproblem_cfg=self.subproblem_cfg
        )
        self.subproblem_manager.load()
        return self.subproblem_manager
    
    def _try_init_row_generation_manager(self):
        """
        Initialize the RowGenerationSolver if not already present.
        Returns a RowGenerationSolver instance using explicit dependencies.

        Returns:
            RowGenerationSolver: The initialized RowGenerationSolver instance.
        Raises:
            RuntimeError: If required managers are not set.
        """
        if self.data_manager is None or self.feature_manager is None or self.subproblem_manager is None:
            raise RuntimeError("DataManager, FeatureManager, and SubproblemManager must be set before initializing row generation manager.")
            return None

        self.row_generation_manager = RowGenerationSolver(
            comm=self.comm,
            dimensions_cfg=self.dimensions_cfg,
            rowgen_cfg=self.rowgen_cfg,
            data_manager=self.data_manager,
            feature_manager=self.feature_manager,
            subproblem_manager=self.subproblem_manager
        )
        return self.row_generation_manager

    # Properties
    @property
    def data(self):
        if self.data_manager is None:
            self._try_init_data_manager()
        return self.data_manager

    @property
    def features(self):
        if self.feature_manager is None:
            self._try_init_feature_manager()
        return self.feature_manager

    @property
    def subproblems(self):
        if self.subproblem_manager is None:
            self._try_init_subproblem_manager()
        return self.subproblem_manager

    @property
    def row_generation(self):
        if self.row_generation_manager is None:
            self._try_init_row_generation_manager()
        return self.row_generation_manager
