import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
from bundlechoice.utils import get_logger

logger = get_logger(__name__)

@dataclass
class QuadraticDataInfo:
    modular_agent: int = 0
    modular_item: int = 0
    quadratic_agent: int = 0
    quadratic_item: int = 0
    constraint_mask: np.ndarray = None
    
    def __post_init__(self):
        offset, self.slices = 0, {}
        for key in ['modular_agent', 'modular_item', 'quadratic_agent', 'quadratic_item']:
            dim = getattr(self, key)
            if dim:
                self.slices[key] = slice(offset, offset + dim)
                offset += dim

class DataManager:

    def __init__(self, dimensions_cfg, comm_manager):
        self.dimensions_cfg = dimensions_cfg
        self.comm_manager = comm_manager


        self.input_data = {'agent_data': {}, 'item_data': {}}
        self.local_data = {'agent_data': {}, 'item_data': {}}
        self.input_data_dictionary_metadata = {'agent_data': {}, 'item_data': {}}
        
    @property
    def global_ids(self):
        return self._global_ids(self.dimensions_cfg.num_obs, self.dimensions_cfg.num_simulations)

    @lru_cache(maxsize=1)
    def _global_ids(self, num_obs, num_simulations):
        return np.arange(self.comm_manager.rank, num_simulations * num_obs, self.comm_manager.comm_size)

    @property
    def num_local_agent(self):
        return len(self.global_ids)

    @property
    def obs_ids(self):
        return self.global_ids % self.dimensions_cfg.num_obs

    @property
    def agent_counts(self):
        return self._agent_counts(self.dimensions_cfg.num_agents)

    @lru_cache(maxsize=1)
    def _agent_counts(self, num_agents):
        return np.array([len(v) for v in np.array_split(np.arange(num_agents), self.comm_manager.comm_size)], dtype=np.int64)


    def load_input_data(self, input_data, preserve_global_data=False):
        if "obs_bundles" not in input_data["obs_data"]:
            raise ValueError("obs_bundles not found in input_data")

        local_agent_data, agent_data_metadata = self.comm_manager.scatter_dict(input_data["obs_data"], agent_counts=self.agent_counts, return_metadata=True)
        item_data, item_data_metadata = self.comm_manager.bcast_dict(input_data["item_data"], return_metadata=True)
        
        self.local_data["obs_data"].update(local_agent_data)
        self.local_data["item_data"].update(item_data)
        self.input_data_dictionary_metadata["obs_data"].update(agent_data_metadata)
        self.input_data_dictionary_metadata["item_data"].update(item_data_metadata)
        
        self._local_data_version = getattr(self, "_local_data_version", 0) + 1
        self.local_obs_bundles = self.local_data["obs_data"]["obs_bundles"]
        if preserve_global_data:
            self.input_data.update(input_data)
        else:
            self.input_data = {}

    def erase_input_data(self):
        self.input_data = {}
        self.input_data_dictionary_metadata = None
        self.local_data = None
        self._local_data_version = 0
        self.local_obs_bundles = None
                
    @property
    def quadratic_data_info(self):
        return self._quadratic_data_info(self._local_data_version)

    @lru_cache(maxsize=1)
    def _quadratic_data_info(self, _version):
        ad, id = self.local_data["obs_data"], self.local_data["item_data"]
        dim = lambda d, k: d[k].shape[-1] if k in d else 0
        return QuadraticDataInfo(dim(ad, "modular"), dim(id, "modular"), dim(ad, "quadratic"), dim(id, "quadratic"), ad.get("constraint_mask"))

    def load_from_directory(self, path, agent_files=None, item_files=None, auto_detect_quadratic_features=False):
        if not self.comm_manager._is_root():
            return
        path = Path(path)
        if auto_detect_quadratic_features:
            available = {f.stem.lower(): f for f in path.iterdir() if f.suffix in (".csv", ".npy")}
            agent_files, item_files = {}, {}
            for feat in ["modular", "quadratic"]:
                if f"{feat}_agent" in available:
                    agent_files[feat] = available[f"{feat}_agent"]
                if f"{feat}_item" in available:
                    item_files[feat] = available[f"{feat}_item"]
        for k, f in (agent_files or {}).items():
            self.input_data["obs_data"][k] = self._load(f)
        for k, f in (item_files or {}).items():
            self.input_data["item_data"][k] = self._load(f)

    def _load(self, f):
        f = Path(f)
        return pd.read_csv(f).values if f.suffix == ".csv" else np.load(f)
    



