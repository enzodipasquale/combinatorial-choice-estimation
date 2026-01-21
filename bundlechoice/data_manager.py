import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
from bundlechoice.utils import get_logger

logger = get_logger(__name__)

def update_dict_recursive(target, source):
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            update_dict_recursive(target[key], value)
        else:
            target[key] = value

@dataclass
class QuadraticDataInfo:
    modular_agent: int = 0
    modular_item: int = 0
    quadratic_agent: int = 0
    quadratic_item: int = 0
    constraint_mask: np.ndarray = None
    modular_agent_names: list = None
    modular_item_names: list = None
    quadratic_agent_names: list = None
    quadratic_item_names: list = None

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
            if preserve_global_data:
                update_dict_recursive(self.input_data, input_data)
            else:
                self.input_data = input_data    
        
        tiled_id_data_for_simulations = self._tile_arrays_in_dict(self.input_data["id_data"])
        self.distribute_data(tiled_id_data_for_simulations, self.input_data["item_data"])
        self._local_data_version = getattr(self, "_local_data_version", 0) + 1

        if not preserve_global_data and self.comm_manager._is_root():
            self.input_data = {"id_data": {}, "item_data": {}} 


    def _tile_id_array_for_simulations(self, id_array):
        reps = (self.dimensions_cfg.n_simulations,) + (1,) * (id_array.ndim - 1)
        return np.tile(id_array, reps)

    def _tile_arrays_in_dict(self, dict):
        if not self.comm_manager._is_root():
            return None
        dict_of_tiled_arrays = {}
        for k, v in dict.items():
            if isinstance(v, np.ndarray):
                dict_of_tiled_arrays[k] = self._tile_id_array_for_simulations(v)
            elif isinstance(v, dict):
                dict_of_tiled_arrays[k] = self._tile_arrays_in_dict(v)
        return dict_of_tiled_arrays

    def distribute_data(self, to_scatter, to_broadcast):
        local_agent_data, agent_data_metadata = self.comm_manager.scatter_dict(to_scatter, 
                                                                                agent_counts=self.agent_counts, 
                                                                                return_metadata=True)
        item_data, item_data_metadata = self.comm_manager.bcast_dict(to_broadcast, 
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
                    modular_agent_names=agent_data.get("modular_names"),
                    modular_item_names=item_data.get("modular_names"),
                    quadratic_agent_names=agent_data.get("quadratic_names"),
                    quadratic_item_names=item_data.get("quadratic_names"),
                )

    def _load_file_to_array(self, f):
        f = Path(f)
        if f.suffix == ".csv":
            df = pd.read_csv(f)
            arr = df.values
            # Squeeze single-column CSVs to 1D
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr.squeeze(axis=1)
            return arr, df.columns
        elif f.suffix == ".npy":
            return np.load(f), None
     
    
    def _load_folder_features_data(self, folder_path):
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            return None, None
        
        files = sorted(folder.glob("*.csv"))
        if not files:
            return None, None
        
        folder_str = str(folder_path)
        is_id_quadratic = 'id_data/quadratic' in folder_str
        is_item_modular = 'item_data/modular' in folder_str
        needs_stacking = not is_item_modular
        
        arrays = []
        names = []
        
        for f in files:
            arr, col_names = self._load_file_to_array(f)
            if is_id_quadratic:
                arr = arr.reshape(self.dimensions_cfg.n_obs, self.dimensions_cfg.n_items, self.dimensions_cfg.n_items)
            
            arrays.append(arr)
            if is_item_modular:
                names.extend([str(name) for name in col_names])
            else:
                names.append(f.stem)
        
        if needs_stacking:
            combined = np.stack(arrays, axis=-1) if len(arrays) > 1 else np.expand_dims(arrays[0], axis=-1)
        else:
            combined = arrays[0]
        
        return combined, names

    def load_quadratic_data_from_directory(self, path):
        if not self.comm_manager._is_root():
            return None    
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data directory not found: {path}")
        
        input_data = {"id_data": {}, "item_data": {}}
        
        # Load features_data
        features_base = path / "features_data"
        if features_base.exists():
            for data_type, key in [('id_data', 'modular'), ('id_data', 'quadratic'),
                                ('item_data', 'modular'), ('item_data', 'quadratic')]:
                arr, names = self._load_folder_features_data(features_base / data_type / key)
                if arr is not None:
                    input_data[data_type][key] = arr
                    input_data[data_type][f"{key}_names"] = names
        
        # Load other_data
        other_base = path / "other_data"
        if other_base.exists():
            for data_type in ['id_data', 'item_data']:
                folder = other_base / data_type
                if folder.exists():
                    for f in sorted(folder.glob("*.csv")) + sorted(folder.glob("*.npy")):
                        input_data[data_type][f.stem], _ = self._load_file_to_array(f)
        
        # Load obs_bundles
        obs_path = path / "obs_bundles.csv"
        if obs_path.exists():
            input_data["id_data"]["obs_bundles"], _= self._load_file_to_array(obs_path)
        
        return input_data


    