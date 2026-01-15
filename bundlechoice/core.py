from mpi4py import MPI
from bundlechoice.config import BundleChoiceConfig
from bundlechoice.comm_manager import CommManager
from bundlechoice.data_manager import DataManager
from bundlechoice.oracles_manager import OraclesManager
from bundlechoice.subproblems.subproblem_manager import SubproblemManager
from bundlechoice.estimation import RowGenerationManager, StandardErrorsManager, ColumnGenerationManager
from bundlechoice.estimation.ellipsoid import EllipsoidManager
from bundlechoice.utils import get_logger

logger = get_logger(__name__)

class BundleChoice:

    def __init__(self) -> None:
        self.config = cfg = BundleChoiceConfig()
        self.comm_manager = comm = CommManager(MPI.COMM_WORLD)
        self.data_manager = data = DataManager(cfg.dimensions, comm)
        self.oracles_manager = oracles = OraclesManager(cfg.dimensions, comm, data)
        
        base = (comm, cfg, data, oracles)
        self.subproblem_manager = subpb = SubproblemManager(*base)
        base_est = (*base, subpb)
        self.row_generation_manager = RowGenerationManager(*base_est)
        self.column_generation_manager = ColumnGenerationManager(*base_est)
        self.ellipsoid_manager = EllipsoidManager(*base_est)
        self.standard_errors_manager = StandardErrorsManager(*base_est)

    @property
    def data(self) -> 'DataManager':
        return self.data_manager

    @property
    def oracles(self) -> 'OraclesManager':
        return self.oracles_manager

    @property
    def subproblems(self) -> 'SubproblemManager':
        return self.subproblem_manager

    @property
    def row_generation(self) -> 'RowGenerationManager':
        return self.row_generation_manager

    @property
    def ellipsoid(self) -> 'EllipsoidManager':
        return self.ellipsoid_manager

    @property
    def column_generation(self) -> 'ColumnGenerationManager':
        return self.column_generation_manager

    @property
    def standard_errors(self) -> 'StandardErrorsManager':
        return self.standard_errors_manager

    @property
    def num_obs(self) -> int:
        return self.config.dimensions.num_obs

    @property
    def num_items(self) -> int:
        return self.config.dimensions.num_items

    @property
    def num_features(self) -> int:
        return self.config.dimensions.num_features

    @property
    def num_simulations(self) -> int:
        return self.config.dimensions.num_simulations

    @property
    def rank(self) -> int:
        return self.comm_manager.rank

    def is_root(self) -> bool:
        return self.comm_manager._is_root()


    def load_config(self, cfg):
        self.config.update_in_place(cfg)
    


   