from .resampling import ResamplingMixin

class StandardErrorsManager(ResamplingMixin):

    def __init__(self, comm_manager, config, data_manager, oracles_manager, subproblem_manager, row_generation_manager):
        self.comm_manager = comm_manager
        self.config = config
        self.data_manager = data_manager
        self.oracles_manager = oracles_manager
        self.subproblem_manager = subproblem_manager
        self.row_generation_manager = row_generation_manager

        self.se_cfg = config.standard_errors
        self.dim = config.dimensions
