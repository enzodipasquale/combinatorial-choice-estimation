import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.config import DimensionsConfig
from bundlechoice.comm_manager import CommManager
from bundlechoice.data_manager import DataManager, QuadraticDataInfo

def test_data_manager_init():
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    dc = DimensionsConfig(num_obs=10, num_items=5, num_features=3, num_simulations=1)
    dm = DataManager(dc, cm)
    assert dm.dimensions_cfg == dc
    assert dm.comm_manager == cm
    assert dm.input_data == {'agent_data': {}, 'item_data': {}}

def test_data_manager_local_id():
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    dc = DimensionsConfig(num_obs=10, num_items=5, num_features=3, num_simulations=2)
    dm = DataManager(dc, cm)
    local_id = dm.local_id
    assert len(local_id) > 0
    assert all(local_id % cm.comm_size == cm.rank)

def test_data_manager_num_local_agent():
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    dc = DimensionsConfig(num_obs=10, num_items=5, num_features=3, num_simulations=2)
    dm = DataManager(dc, cm)
    assert dm.num_local_agent == len(dm.local_id)

def test_data_manager_local_obs_id():
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    dc = DimensionsConfig(num_obs=10, num_items=5, num_features=3, num_simulations=2)
    dm = DataManager(dc, cm)
    local_obs_id = dm.local_obs_id
    assert all(local_obs_id < dc.num_obs)

def test_data_manager_agent_counts():
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    dc = DimensionsConfig(num_obs=10, num_items=5, num_features=3, num_simulations=2)
    dm = DataManager(dc, cm)
    counts = dm.agent_counts
    assert counts.sum() == dc.num_agents
    assert len(counts) == cm.comm_size

def test_data_manager_load_input_data():
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    dc = DimensionsConfig(num_obs=10, num_items=5, num_features=3, num_simulations=1)
    dm = DataManager(dc, cm)
    num_obs = 10
    num_items = 5
    if cm._is_root():
        input_data = {
            'agent_data': {
                'obs_bundles': np.random.randn(num_obs, num_items) > 0.5,
                'modular': np.random.randn(num_obs, num_items, 2)
            },
            'item_data': {'modular': np.random.randn(num_items, 1)}
        }
    else:
        input_data = {'agent_data': {}, 'item_data': {}}
    dm.load_input_data(input_data)
    assert dm.local_data is not None
    assert 'obs_bundles' in dm.local_data['agent_data']
    assert 'modular' in dm.local_data['agent_data']

def test_data_manager_load_input_data_missing_obs_bundles():
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    dc = DimensionsConfig(num_obs=10, num_items=5, num_features=3, num_simulations=1)
    dm = DataManager(dc, cm)
    if cm._is_root():
        input_data = {
            'agent_data': {'modular': np.random.randn(10, 5, 2)},
            'item_data': {}
        }
    else:
        input_data = {'agent_data': {}, 'item_data': {}}
    with pytest.raises(ValueError, match="obs_bundles not found"):
        dm.load_input_data(input_data)

def test_data_manager_quadratic_data_info():
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    dc = DimensionsConfig(num_obs=10, num_items=5, num_features=3, num_simulations=1)
    dm = DataManager(dc, cm)
    num_obs = 10
    num_items = 5
    if cm._is_root():
        input_data = {
            'agent_data': {
                'obs_bundles': np.random.randn(num_obs, num_items) > 0.5,
                'modular': np.random.randn(num_obs, num_items, 2),
                'quadratic': np.random.randn(num_obs, num_items, num_items, 1)
            },
            'item_data': {
                'modular': np.random.randn(num_items, 1),
                'quadratic': np.random.randn(num_items, num_items, 1)
            }
        }
    else:
        input_data = {'agent_data': {}, 'item_data': {}}
    dm.load_input_data(input_data)
    qinfo = dm.quadratic_data_info
    assert qinfo.modular_agent == 2
    assert qinfo.modular_item == 1
    assert qinfo.quadratic_agent == 1
    assert qinfo.quadratic_item == 1

def test_data_manager_quadratic_data_info_slices():
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    dc = DimensionsConfig(num_obs=10, num_items=5, num_features=3, num_simulations=1)
    dm = DataManager(dc, cm)
    num_obs = 10
    num_items = 5
    if cm._is_root():
        input_data = {
            'agent_data': {
                'obs_bundles': np.random.randn(num_obs, num_items) > 0.5,
                'modular': np.random.randn(num_obs, num_items, 2),
                'quadratic': np.random.randn(num_obs, num_items, num_items, 1)
            },
            'item_data': {
                'modular': np.random.randn(num_items, 1),
                'quadratic': np.random.randn(num_items, num_items, 1)
            }
        }
    else:
        input_data = {'agent_data': {}, 'item_data': {}}
    dm.load_input_data(input_data)
    qinfo = dm.quadratic_data_info
    assert 'modular_agent' in qinfo.slices
    assert 'modular_item' in qinfo.slices
    assert 'quadratic_agent' in qinfo.slices
    assert 'quadratic_item' in qinfo.slices

def test_data_manager_quadratic_data_info_empty():
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    dc = DimensionsConfig(num_obs=10, num_items=5, num_features=3, num_simulations=1)
    dm = DataManager(dc, cm)
    num_obs = 10
    num_items = 5
    if cm._is_root():
        input_data = {
            'agent_data': {
                'obs_bundles': np.random.randn(num_obs, num_items) > 0.5
            },
            'item_data': {}
        }
    else:
        input_data = {'agent_data': {}, 'item_data': {}}
    dm.load_input_data(input_data)
    qinfo = dm.quadratic_data_info
    assert qinfo.modular_agent == 0
    assert qinfo.modular_item == 0
    assert qinfo.quadratic_agent == 0
    assert qinfo.quadratic_item == 0
