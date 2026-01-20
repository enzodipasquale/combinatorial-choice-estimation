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


    def load_and_distribute_input_data(self, input_data, preserve_global_data=False):
        if self.comm_manager._is_root():
            self.input_data = input_data    
        
        self.distribute_data()
        self._local_data_version = getattr(self, "_local_data_version", 0) + 1

        if preserve_global_data:
            self.input_data.update(input_data)
        else:
            self.input_data = {"id_data": {}, "item_data": {}} if self.comm_manager._is_root() else self.input_data


    def distribute_data(self):
        local_agent_data, agent_data_metadata = self.comm_manager.scatter_dict(self.input_data["id_data"], 
                                                                                agent_counts=self.agent_counts, 
                                                                                return_metadata=True)
        item_data, item_data_metadata = self.comm_manager.bcast_dict(self.input_data["item_data"], 
                                                                        return_metadata=True)
        
        self.local_data["id_data"].update(local_agent_data)
        self.local_data["item_data"].update(item_data)
        self.input_data_dictionary_metadata["id_data"].update(agent_data_metadata)
        self.input_data_dictionary_metadata["item_data"].update(item_data_metadata)

    def erase_input_data(self):
        self.input_data = {"id_data": {}, "item_data": {}}
        self.input_data_dictionary_metadata = {"id_data": {}, "item_data": {}}
        self.local_data = {"id_data": {}, "item_data": {}}
        self._local_data_version = 0
                












    @property
    def quadratic_data_info(self):
        return self._quadratic_data_info(self._local_data_version)

    @lru_cache(maxsize=1)
    def _quadratic_data_info(self, _version):
        agent_data, item_data = self.local_data["id_data"], self.local_data["item_data"]
        dim = lambda d, k: d[k].shape[-1] if k in d else 0
        return QuadraticDataInfo(
                                    modular_agent=dim(agent_data, "modular"),
                                    modular_item=dim(item_data, "modular"),
                                    quadratic_agent=dim(agent_data, "quadratic"),
                                    quadratic_item=dim(item_data, "quadratic"),
                                    constraint_mask=agent_data.get("constraint_mask"),
                                )

    def _load(self, f):
        f = Path(f)
        return pd.read_csv(f).values if f.suffix == ".csv" else np.load(f)
    
    def _load_folder_features(self, folder_path):
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            return None
        arrays = [self._load(f) for f in sorted(folder.glob("*.csv")) + sorted(folder.glob("*.npy"))]
        if not arrays:
            return None
        return np.concatenate(arrays, axis=-1) if len(arrays) > 1 else arrays[0]

    def load_quadratic_data_from_directory(self, path, additional_agent_data= None, 
                                        additional_item_data= None, error_seed= None):

        if not self.comm_manager._is_root():
            return None
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data directory not found: {path}")
        
        input_data = {"id_data": {}, "item_data": {}}
        features_base = path / "features" if (path / "features").exists() else path
        

        for folder_name, data_type, key in [
            ('modular_agent', 'id_data', 'modular'),
            ('modular_item', 'item_data', 'modular'),
            ('quadratic_agent', 'id_data', 'quadratic'),
            ('quadratic_item', 'item_data', 'quadratic'),
        ]:
            arr = self._load_folder_features(features_base / folder_name)
            if arr is not None:
                input_data[data_type][key] = arr
  
        for key, file_path in (additional_agent_data or {}).items():
            input_data["id_data"][key] = self._load(Path(file_path))
        
        for key, file_path in (additional_item_data or {}).items():
            input_data["item_data"][key] = self._load(Path(file_path))
        
        if error_seed is not None:
            input_data["_error_seed"] = error_seed
        
        return input_data


