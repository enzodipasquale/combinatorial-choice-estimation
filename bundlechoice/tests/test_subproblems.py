import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.config import BundleChoiceConfig, DimensionsConfig
from bundlechoice.comm_manager import CommManager
from bundlechoice.data_manager import DataManager
from bundlechoice.oracles_manager import OraclesManager
from bundlechoice.subproblems.subproblem_manager import SubproblemManager
from bundlechoice.subproblems.subproblem_registry import SUBPROBLEM_REGISTRY

def test_subproblem_registry():
    """Test subproblem registry"""
    assert 'Greedy' in SUBPROBLEM_REGISTRY
    assert 'LinearKnapsack' in SUBPROBLEM_REGISTRY
    assert 'PlainSingleItem' in SUBPROBLEM_REGISTRY
    assert 'BruteForce' in SUBPROBLEM_REGISTRY
    assert 'QuadKnapsack' in SUBPROBLEM_REGISTRY

def test_subproblem_registry_keys():
    """Test registry keys method"""
    keys = list(SUBPROBLEM_REGISTRY.keys())
    assert 'Greedy' in keys
    assert 'LinearKnapsack' in keys
    assert len(keys) > 0

def test_subproblem_registry_get():
    """Test registry get method"""
    cls = SUBPROBLEM_REGISTRY.get('Greedy')
    assert cls is not None

def test_subproblem_registry_get_nonexistent():
    """Test registry get with nonexistent subproblem"""
    cls = SUBPROBLEM_REGISTRY.get('NonExistent')
    assert cls is None

def test_subproblem_registry_contains():
    """Test registry contains method"""
    assert 'Greedy' in SUBPROBLEM_REGISTRY
    assert 'NonExistent' not in SUBPROBLEM_REGISTRY

def test_subproblem_manager_init():
    """Test SubproblemManager initialization"""
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    dc = DimensionsConfig(num_obs=10, num_items=5, num_features=3, num_simulations=1)
    dm = DataManager(dc, cm)
    om = OraclesManager(dc, cm, dm)
    cfg = BundleChoiceConfig()
    cfg.dimensions = dc
    sm = SubproblemManager(cm, cfg, dm, om)
    assert sm.config == cfg
    assert sm.comm_manager == cm
    assert sm.data_manager == dm
    assert sm.oracles_manager == om
    assert sm.subproblem is None

def test_subproblem_manager_load():
    """Test SubproblemManager load method"""
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    dc = DimensionsConfig(num_obs=10, num_items=5, num_features=3, num_simulations=1)
    dm = DataManager(dc, cm)
    om = OraclesManager(dc, cm, dm)
    cfg = BundleChoiceConfig()
    cfg.dimensions = dc
    cfg.subproblem.name = 'Greedy'
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
    om.build_quadratic_features_from_data()
    sm = SubproblemManager(cm, cfg, dm, om)
    sub = sm.load('Greedy')
    assert sub is not None
    assert sm.subproblem is not None

def test_subproblem_manager_load_from_config():
    """Test SubproblemManager load from config"""
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    dc = DimensionsConfig(num_obs=10, num_items=5, num_features=3, num_simulations=1)
    dm = DataManager(dc, cm)
    om = OraclesManager(dc, cm, dm)
    cfg = BundleChoiceConfig()
    cfg.dimensions = dc
    cfg.subproblem.name = 'Greedy'
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
    om.build_quadratic_features_from_data()
    sm = SubproblemManager(cm, cfg, dm, om)
    sub = sm.load()
    assert sub is not None
    assert sm.subproblem is not None

def test_subproblem_manager_load_unknown():
    """Test SubproblemManager load with unknown subproblem"""
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    dc = DimensionsConfig(num_obs=10, num_items=5, num_features=3, num_simulations=1)
    dm = DataManager(dc, cm)
    om = OraclesManager(dc, cm, dm)
    cfg = BundleChoiceConfig()
    sm = SubproblemManager(cm, cfg, dm, om)
    with pytest.raises(ValueError, match="Unknown subproblem"):
        sm.load('NonExistent')

def test_subproblem_manager_initialize_subproblems():
    """Test SubproblemManager initialize_subproblems"""
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    dc = DimensionsConfig(num_obs=10, num_items=5, num_features=3, num_simulations=1)
    dm = DataManager(dc, cm)
    om = OraclesManager(dc, cm, dm)
    cfg = BundleChoiceConfig()
    cfg.dimensions = dc
    cfg.subproblem.name = 'Greedy'
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
    om.build_quadratic_features_from_data()
    sm = SubproblemManager(cm, cfg, dm, om)
    sm.initialize_subproblems()
    assert sm.subproblem is not None

def test_subproblem_manager_solve_subproblems():
    """Test SubproblemManager solve_subproblems"""
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    dc = DimensionsConfig(num_obs=10, num_items=5, num_features=3, num_simulations=1)
    dm = DataManager(dc, cm)
    om = OraclesManager(dc, cm, dm)
    cfg = BundleChoiceConfig()
    cfg.dimensions = dc
    cfg.subproblem.name = 'Greedy'
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
    om.build_quadratic_features_from_data()
    sm = SubproblemManager(cm, cfg, dm, om)
    sm.initialize_subproblems()
    theta = np.ones(dc.num_features)
    result = sm.solve_subproblems(theta)
    assert result is not None
    assert result.shape[0] == dm.num_local_agent
    assert result.shape[1] == dc.num_items
    assert result.dtype == bool

def test_subproblem_manager_initialize_and_solve_subproblems():
    """Test SubproblemManager initialize_and_solve_subproblems"""
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    dc = DimensionsConfig(num_obs=10, num_items=5, num_features=3, num_simulations=1)
    dm = DataManager(dc, cm)
    om = OraclesManager(dc, cm, dm)
    cfg = BundleChoiceConfig()
    cfg.dimensions = dc
    cfg.subproblem.name = 'Greedy'
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
    om.build_quadratic_features_from_data()
    sm = SubproblemManager(cm, cfg, dm, om)
    if cm._is_root():
        theta = np.ones(dc.num_features)
    else:
        theta = np.empty(dc.num_features)
    result = sm.initialize_and_solve_subproblems(theta)
    assert result is not None
    assert result.shape[0] == dm.num_local_agent
    assert result.shape[1] == dc.num_items

def test_subproblem_manager_multiple_subproblems():
    """Test loading different subproblem types"""
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    dc = DimensionsConfig(num_obs=10, num_items=5, num_features=3, num_simulations=1)
    dm = DataManager(dc, cm)
    om = OraclesManager(dc, cm, dm)
    cfg = BundleChoiceConfig()
    cfg.dimensions = dc
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
    om.build_quadratic_features_from_data()
    
    # Test Greedy
    cfg.subproblem.name = 'Greedy'
    sm1 = SubproblemManager(cm, cfg, dm, om)
    sub1 = sm1.load('Greedy')
    assert sub1 is not None
    
    # Test PlainSingleItem
    cfg.subproblem.name = 'PlainSingleItem'
    sm2 = SubproblemManager(cm, cfg, dm, om)
    sub2 = sm2.load('PlainSingleItem')
    assert sub2 is not None
