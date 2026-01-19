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
    modular_agent_names: str = None
    modular_item_names: str = None
    quadratic_agent_names: str = None
    quadratic_item_names: str = None

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


        self.input_data = {"id_data": {}, "item_data": {}}
        self.local_data = {"id_data": {}, "item_data": {}}
        self.input_data_dictionary_metadata = {"id_data": {}, "item_data": {}}
        
    @property
    def agent_ids(self):
        return self._agent_ids(self.dimensions_cfg.n_obs, self.dimensions_cfg.n_simulations)

    @lru_cache(maxsize=1)
    def _agent_ids(self, n_obs, n_simulations):
        splits = np.array_split(np.arange(n_simulations * n_obs), self.comm_manager.comm_size)
        return splits[self.comm_manager.rank]

    @lru_cache(maxsize=1)
    def _agent_counts(self, num_agents, comm_size):
        return np.array([len(v) for v in np.array_split(np.arange(num_agents), comm_size)], dtype=np.int64)

    @property
    def agent_counts(self):
        return self._agent_counts(self.dimensions_cfg.num_agents, self.comm_manager.comm_size)

    @property
    def num_local_agent(self):
        return len(self.agent_ids)

    @property
    def obs_ids(self):
        return self.agent_ids % self.dimensions_cfg.n_obs

    @property
    def local_obs_bundles(self):
        return np.asarray(self.local_data["id_data"]["obs_bundles"], dtype=bool)


    def load_input_data(self, input_data, preserve_global_data=False):
        self.input_data = input_data if self.comm_manager._is_root() else self.input_data
            
        local_agent_data, agent_data_metadata = self.comm_manager.scatter_dict(self.input_data["id_data"], 
                                                                                agent_counts=self.agent_counts, 
                                                                                return_metadata=True)
        item_data, item_data_metadata = self.comm_manager.bcast_dict(self.input_data["item_data"], 
                                                                        return_metadata=True)
        
        self.local_data["id_data"].update(local_agent_data)
        self.local_data["item_data"].update(item_data)
        self.input_data_dictionary_metadata["id_data"].update(agent_data_metadata)
        self.input_data_dictionary_metadata["item_data"].update(item_data_metadata)
        
        self._local_data_version = getattr(self, "_local_data_version", 0) + 1
        
        



        if preserve_global_data:
            self.input_data.update(input_data)
        else:
            self.input_data = {"id_data": {}, "item_data": {}} if self.comm_manager._is_root() else self.input_data

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
        ad, id = self.local_data["id_data"], self.local_data["item_data"]
        dim = lambda d, k: d[k].shape[-1] if k in d else 0
        return QuadraticDataInfo(dim(ad, "modular"), dim(id, "modular"), 
                                    dim(ad, "quadratic"), dim(id, "quadratic"), 
                                    ad.get("constraint_mask"))

    def _detect_quadratic_features(self, path):
        path = Path(path)
        available = {f.stem.lower(): f for f in path.iterdir() if f.suffix in (".csv", ".npy")}
        agent_files, item_files = {}, {}
        for feat in ["modular", "quadratic"]:
            if f"{feat}_agent" in available:
                agent_files[feat] = available[f"{feat}_agent"]
                self.quadratic_data_info.modular_agent_names = pd.read_csv(agent_files[feat]).columns.tolist()
            if f"{feat}_item" in available:
                item_files[feat] = available[f"{feat}_item"]
                self.quadratic_data_info.modular_item_names = pd.read_csv(item_files[feat]).columns.tolist()

    def load_from_directory(self, path, agent_files=None, item_files=None, auto_detect_quadratic_features=False):
        if not self.comm_manager._is_root():
            return
        path = Path(path)
        if auto_detect_quadratic_features:
            agent_files, item_files = self._detect_quadratic_features(path)
        for k, f in (agent_files or {}).items():
            # load csv as pandas dataframe and turn into numpy array    
            self.input_data["id_data"][k] = pd.read_csv(f).values
        for k, f in (item_files or {}).items():
            self.input_data["item_data"][k] = pd.read_csv(f).values

    def _load(self, f):
        f = Path(f)
        return pd.read_csv(f).values if f.suffix == ".csv" else np.load(f)
    



