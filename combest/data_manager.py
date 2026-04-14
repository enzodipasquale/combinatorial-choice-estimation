import numpy as np
from dataclasses import dataclass, field
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

    def __post_init__(self):
        offset, self.slices = 0, {}
        for key in ['modular_agent', 'modular_item', 'quadratic_agent', 'quadratic_item']:
            dim = getattr(self, key)
            if dim:
                self.slices[key] = slice(offset, offset + dim)
                offset += dim


@dataclass
class LocalData:
    id_data: dict = field(default_factory=dict)
    item_data: dict = field(default_factory=dict)
    id_metadata: dict = field(default_factory=dict)
    item_metadata: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)

class DataManager:

    def __init__(self, dimensions_cfg, comm_manager):
        self.dimensions_cfg = dimensions_cfg
        self.comm_manager = comm_manager
        self.local_data = LocalData()

    @property
    def local_obs_bundles(self):
        return self.local_data.id_data["obs_bundles"]

    def load_and_distribute_input_data(self, input_data):
        root_data = input_data if self.comm_manager.is_root() else {"id_data": {}, "item_data": {}}
        id_data_full, id_meta = self.comm_manager.bcast_dict(root_data["id_data"], return_metadata=True)
        obs_ids = self.comm_manager.obs_ids
        sim_ids = self.comm_manager.agent_ids // self.dimensions_cfg.n_obs
        local_id = {}
        for k, v in id_data_full.items():
            if not isinstance(v, np.ndarray):
                local_id[k] = v
            elif v.ndim == 2 and v.shape == (self.dimensions_cfg.n_obs, self.dimensions_cfg.n_simulations):
                local_id[k] = v[obs_ids, sim_ids]
            else:
                local_id[k] = v[obs_ids]
        del id_data_full

        item_data, item_meta = self.comm_manager.bcast_dict(root_data["item_data"], return_metadata=True)

        self.local_data.id_data.update(local_id)
        self.local_data.item_data.update(item_data)
        self.local_data.id_metadata.update(id_meta)
        self.local_data.item_metadata.update(item_meta)



    def erase_input_data(self):
        self.local_data = LocalData()

    @property
    def local_obs_quantity(self):
        q = self.local_data.id_data.get("obs_quantity")
        if q is not None:
            return np.asarray(q, dtype=np.float64)
        return np.ones(self.comm_manager.num_local_agent, dtype=np.float64)


    def get_quadratic_data_info(self):
        agent_data, item_data = self.local_data.id_data, self.local_data.item_data
        dim = lambda d, k: d[k].shape[-1] if k in d else 0
        return QuadraticDataInfo(
                    modular_agent=dim(agent_data, "modular"),
                    modular_item=dim(item_data, "modular"),
                    quadratic_agent=dim(agent_data, "quadratic"),
                    quadratic_item=dim(item_data, "quadratic"),
                )

    def _validate_quadratic_data_dimensions(self):
        qinfo = self.get_quadratic_data_info()
        agent_data, item_data = self.local_data.id_data, self.local_data.item_data
        n_items = self.dimensions_cfg.n_items
        n_local = self.comm_manager.num_local_agent

        expected_shapes = {
            ('id_data', 'modular', qinfo.modular_agent): (n_local, n_items, qinfo.modular_agent),
            ('item_data', 'modular', qinfo.modular_item): (n_items, qinfo.modular_item),
            ('id_data', 'quadratic', qinfo.quadratic_agent): (n_local, n_items, n_items, qinfo.quadratic_agent),
            ('item_data', 'quadratic', qinfo.quadratic_item): (n_items, n_items, qinfo.quadratic_item),
        }
        for (src, key, dim), expected in expected_shapes.items():
            if not dim:
                continue
            data = agent_data if src == 'id_data' else item_data
            if data[key].shape != expected:
                raise ValueError(f"{src}['{key}'] shape {data[key].shape} != expected {expected}")

        total_covariates = qinfo.modular_agent + qinfo.modular_item + qinfo.quadratic_agent + qinfo.quadratic_item
        if total_covariates != self.dimensions_cfg.n_covariates:
            raise ValueError(
                f"Total covariates from data ({total_covariates}) != "
                f"n_covariates in config ({self.dimensions_cfg.n_covariates})"
            )
