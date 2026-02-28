from mpi4py import MPI
from pathlib import Path
from combchoice.config import ModelConfig
from combchoice.comm_manager import CommManager
from combchoice.data_manager import DataManager
from combchoice.features_manager import FeaturesManager
from combchoice.subproblems.subproblem_manager import SubproblemManager
from combchoice.estimation import RowGenerationManager
from combchoice.estimation.standard_errors import StandardErrorsManager
from combchoice.utils import get_logger

logger = get_logger(__name__)

class Model:

    def __init__(self) -> None:
        self.config = cfg = ModelConfig()
        self.comm_manager = comm = CommManager(MPI.COMM_WORLD)
        self.data_manager = data = DataManager(cfg.dimensions, comm)
        self.features_manager = features = FeaturesManager(cfg.dimensions, comm, data)

        base = (comm, cfg, data, features)
        self.subproblem_manager = subpb = SubproblemManager(*base)
        base_est = (*base, subpb)
        self.row_generation_manager = row_gen = RowGenerationManager(*base_est)
        self.standard_errors_manager = StandardErrorsManager(*base_est, row_gen)

    @property
    def data(self) -> 'DataManager':
        return self.data_manager

    @property
    def features(self) -> 'FeaturesManager':
        return self.features_manager

    @property
    def subproblems(self) -> 'SubproblemManager':
        return self.subproblem_manager

    @property
    def row_generation(self) -> 'RowGenerationManager':
        return self.row_generation_manager

    @property
    def standard_errors(self) -> 'StandardErrorsManager':
        return self.standard_errors_manager

    @property
    def n_obs(self) -> int:
        return self.config.dimensions.n_obs

    @property
    def n_items(self) -> int:
        return self.config.dimensions.n_items

    @property
    def n_covariates(self) -> int:
        return self.config.dimensions.n_covariates

    @property
    def n_simulations(self) -> int:
        return self.config.dimensions.n_simulations

    @property
    def rank(self) -> int:
        return self.comm_manager.rank

    def is_root(self) -> bool:
        return self.comm_manager.is_root()

    def load_config(self, cfg):
        if isinstance(cfg, (str, Path)):
            cfg = ModelConfig.from_yaml(cfg)
        elif isinstance(cfg, dict):
            cfg = ModelConfig.from_dict(cfg)
        cfg = self.comm_manager.bcast(cfg)
        self.config.update_in_place(cfg)
        self.comm_manager.init_assignment(self.config.dimensions)
