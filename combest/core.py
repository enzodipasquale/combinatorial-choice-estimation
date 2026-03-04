from mpi4py import MPI
from pathlib import Path
from combest.config import ModelConfig
from combest.comm_manager import CommManager
from combest.data_manager import DataManager
from combest.features_manager import FeaturesManager
from combest.subproblems.subproblem_manager import SubproblemManager
from combest.estimation import PointEstimationManager
from combest.estimation.standard_errors import StandardErrorsManager
from combest.utils import get_logger

logger = get_logger(__name__)


class Model:

    def __init__(self) -> None:
        self.config = cfg = ModelConfig()
        self.comm_manager = comm = CommManager(MPI.COMM_WORLD)
        self.data_manager = data = DataManager(cfg.dimensions, comm)
        self.features_manager = features = FeaturesManager(cfg.dimensions, comm, data)

        base = (comm, cfg, data, features)
        self.subproblem_manager = subpb = SubproblemManager(*base)
        self.point_estimation_manager = pt_est = PointEstimationManager(*base, subpb)
        self.standard_errors_manager = StandardErrorsManager(pt_est)

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
    def point_estimation(self):
        return self.point_estimation_manager

    @property
    def row_generation(self):
        return self.point_estimation_manager.n_slack

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
