import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.config import DimensionsConfig
from bundlechoice.comm_manager import CommManager
from bundlechoice.data_manager import DataManager
from bundlechoice.oracles_manager import OraclesManager

def test_oracles_manager_init():
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    dc = DimensionsConfig(num_obs=10, num_items=5, num_features=3, num_simulations=1)
    dm = DataManager(dc, cm)
    om = OraclesManager(dc, cm, dm)
    assert om.dimensions_cfg == dc
    assert om.comm_manager == cm
    assert om.data_manager == dm
    assert om._features_oracle is None
    assert om._error_oracle is None

def test_oracles_manager_set_features_oracle():
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
    om = OraclesManager(dc, cm, dm)
    def features_oracle(bundles, local_id, data):
        return np.ones((len(local_id), dc.num_features))
    om.set_features_oracle(features_oracle)
    assert om._features_oracle is not None

def test_oracles_manager_set_error_oracle():
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
    om = OraclesManager(dc, cm, dm)
    def error_oracle(bundles, local_id):
        return np.ones((len(local_id),))
    om.set_error_oracle(error_oracle)
    assert om._error_oracle is not None

def test_oracles_manager_build_local_modular_error_oracle():
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
    om = OraclesManager(dc, cm, dm)
    oracle = om.build_local_modular_error_oracle(seed=42)
    assert oracle is not None
    assert om._error_oracle_vectorized == True
    assert om._error_oracle_takes_data == False

def test_oracles_manager_build_quadratic_features_from_data():
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
    om = OraclesManager(dc, cm, dm)
    oracle = om.build_quadratic_features_from_data()
    assert oracle is not None
    assert om._features_oracle_vectorized == True
    assert om._features_oracle_takes_data == True

def test_oracles_manager_features_oracle():
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
    om = OraclesManager(dc, cm, dm)
    def features_oracle(bundles, local_id, data):
        return np.ones((len(local_id), dc.num_features))
    om.set_features_oracle(features_oracle)
    bundles = np.random.randn(dm.num_local_agent, num_items) > 0.5
    features = om.features_oracle(bundles)
    assert features.shape == (dm.num_local_agent, dc.num_features)

def test_oracles_manager_error_oracle():
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
    om = OraclesManager(dc, cm, dm)
    om.build_local_modular_error_oracle(seed=42)
    bundles = np.random.randn(dm.num_local_agent, num_items) > 0.5
    errors = om.error_oracle(bundles)
    assert errors.shape == (dm.num_local_agent,)

def test_oracles_manager_utility_oracle():
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
    om = OraclesManager(dc, cm, dm)
    def features_oracle(bundles, local_id, data):
        return np.ones((len(local_id), dc.num_features))
    om.set_features_oracle(features_oracle)
    om.build_local_modular_error_oracle(seed=42)
    bundles = np.random.randn(dm.num_local_agent, num_items) > 0.5
    theta = np.ones(dc.num_features)
    utility = om.utility_oracle(bundles, theta)
    assert utility.shape == (dm.num_local_agent,)

def test_oracles_manager_features_oracle_individual():
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
    om = OraclesManager(dc, cm, dm)
    def features_oracle(bundles, local_id, data):
        return np.ones((len(local_id), dc.num_features))
    om.set_features_oracle(features_oracle)
    bundle = np.random.randn(num_items) > 0.5
    local_id = 0
    features = om.features_oracle_individual(bundle, local_id)
    assert features.shape == (dc.num_features,)

def test_oracles_manager_error_oracle_individual():
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
    om = OraclesManager(dc, cm, dm)
    om.build_local_modular_error_oracle(seed=42)
    bundle = np.random.randn(num_items) > 0.5
    local_id = 0
    error = om.error_oracle_individual(bundle, local_id)
    assert isinstance(error, (float, np.floating))

def test_oracles_manager_utility_oracle_individual():
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
    om = OraclesManager(dc, cm, dm)
    def features_oracle(bundles, local_id, data):
        return np.ones((len(local_id), dc.num_features))
    om.set_features_oracle(features_oracle)
    om.build_local_modular_error_oracle(seed=42)
    bundle = np.random.randn(num_items) > 0.5
    local_id = 0
    theta = np.ones(dc.num_features)
    utility = om.utility_oracle_individual(bundle, theta, local_id)
    assert isinstance(utility, (float, np.floating))

def test_oracles_manager_compute_features_at_obs_bundles_at_root_at_root():
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
    om = OraclesManager(dc, cm, dm)
    om.build_quadratic_features_from_data()
    features = om._features_at_obs_bundles_at_root
    if cm._is_root():
        assert features.shape == (dc.num_obs, dc.num_features)
