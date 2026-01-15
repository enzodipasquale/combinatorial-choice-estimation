import numpy as np
import pandas as pd
from pathlib import Path
from bundlechoice.utils import get_logger

logger = get_logger(__name__)

class DataManager:

    def __init__(self, dimensions_cfg, comm_manager):
        self.dimensions_cfg = dimensions_cfg
        self.comm_manager = comm_manager
        self.input_data = {'agent_data': {}, 'item_data': {}}
        self.local_data = None
        self.local_ids = np.arange(self.comm_manager.rank, self.dimensions_cfg.num_simulations * self.dimensions_cfg.num_obs, self.comm_manager.comm_size)
        self.local_obs_ids = self.local_ids % self.dimensions_cfg.num_obs
        self.agent_counts = [len(v) for v in np.array_split(np.arange(self.dimensions_cfg.num_agents), self.comm_manager.comm_size)]

    def load_input_data(self, input_data):
        local_agent_data = self.comm_manager._scatter_dict(input_data['agent_data'], agent_counts=self.agent_counts)
        item_data = self.comm_manager._broadcast_dict(input_data['item_data'])
        self.local_data = {'agent_data': local_agent_data, 'item_data': item_data}

    def quadratic_features_flags(self):
        has_modular_agent = 'modular' in self.local_data['agent_data']
        has_quadratic_agent = 'quadratic' in self.local_data['agent_data']
        has_modular_item = 'modular' in self.local_data['item_data']
        has_quadratic_item = 'quadratic' in self.local_data['item_data']
        return has_modular_agent, has_quadratic_agent, has_modular_item, has_quadratic_item

    def load_from_directory(self, path, agent_files=None, item_files=None, auto_detect=False):
        if not self.comm_manager._is_root():
            return
        path = Path(path)
        if auto_detect:
            available = {f.stem.lower(): f for f in path.iterdir() if f.suffix in ('.csv', '.npy')}
            agent_files = {k: available[v] for k, v in [('modular', 'modular_agent'), ('quadratic', 'quadratic_agent')] if v in available}
            item_files = {k: available[v] for k, v in [('modular', 'modular_item'), ('quadratic', 'quadratic_item')] if v in available}
        for k, f in (agent_files or {}).items():
            self.input_data['agent_data'][k] = self._load(f)
        for k, f in (item_files or {}).items():
            self.input_data['item_data'][k] = self._load(f)

    def _load(self, f):
        f = Path(f)
        return pd.read_csv(f).values if f.suffix == '.csv' else np.load(f)


