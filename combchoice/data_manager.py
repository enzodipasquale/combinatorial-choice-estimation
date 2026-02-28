import numpy as np
from dataclasses import dataclass
from functools import lru_cache


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

    @lru_cache(maxsize=1)
    def _get_local_obs_bundles(self, _version):
        return np.asarray(self.local_data["id_data"]["obs_bundles"], dtype=bool)

    @property
    def local_obs_bundles(self):
        return self._get_local_obs_bundles(self._local_data_version)

    def load_and_distribute_input_data(self, input_data, preserve_global_data=False):
        if self.comm_manager.is_root():
            if preserve_global_data:
                update_dict_recursive(self.input_data, input_data)
            else:
                self.input_data = input_data

        root_data = self.input_data if self.comm_manager.is_root() else {"id_data": {}, "item_data": {}}
        id_data_full, id_meta = self.comm_manager.bcast_dict(root_data["id_data"], return_metadata=True)
        obs_ids = self.comm_manager.obs_ids
        local_id = {k: v[obs_ids] if isinstance(v, np.ndarray) else v for k, v in id_data_full.items()}
        del id_data_full

        item_data, item_meta = self.comm_manager.bcast_dict(root_data["item_data"], return_metadata=True)

        self.local_data["id_data"].update(local_id)
        self.local_data["item_data"].update(item_data)
        self.input_data_dictionary_metadata["id_data"].update(id_meta)
        self.input_data_dictionary_metadata["item_data"].update(item_meta)
        self._local_data_version = getattr(self, "_local_data_version", 0) + 1

        if not preserve_global_data and self.comm_manager.is_root():
            self.input_data = {"id_data": {}, "item_data": {}}

    def erase_input_data(self):
        self.input_data = {"id_data": {}, "item_data": {}}
        self.input_data_dictionary_metadata = {"id_data": {}, "item_data": {}}
        self.local_data = {"id_data": {}, "item_data": {}}
        self._local_data_version = 0

    @property
    def local_obs_quantity(self):
        q = self.local_data["id_data"].get("obs_quantity")
        if q is not None:
            return np.asarray(q, dtype=np.float64)
        return np.ones(self.comm_manager.num_local_agent, dtype=np.float64)

    @property
    def quadratic_data_info(self):
        return self._quadratic_data_info(self._local_data_version)

    def _validate_quadratic_data_dimensions(self):
        agent_data, item_data = self.local_data["id_data"], self.local_data["item_data"]
        dim = lambda d, k: d[k].shape[-1] if k in d else 0
        modular_agent_dim = dim(agent_data, "modular")
        modular_item_dim = dim(item_data, "modular")
        quadratic_agent_dim = dim(agent_data, "quadratic")
        quadratic_item_dim = dim(item_data, "quadratic")
        n_items = self.dimensions_cfg.n_items

        if modular_agent_dim:
            assert agent_data['modular'].shape == (self.comm_manager.num_local_agent, n_items, modular_agent_dim)
        if modular_item_dim:
            assert item_data['modular'].shape == (n_items, modular_item_dim)
        if quadratic_agent_dim:
            assert agent_data['quadratic'].shape == (self.comm_manager.num_local_agent, n_items, n_items, quadratic_agent_dim)
        if quadratic_item_dim:
            assert item_data['quadratic'].shape == (n_items, n_items, quadratic_item_dim)

        total_features = modular_agent_dim + modular_item_dim + quadratic_agent_dim + quadratic_item_dim
        assert total_features == self.dimensions_cfg.n_covariates

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
