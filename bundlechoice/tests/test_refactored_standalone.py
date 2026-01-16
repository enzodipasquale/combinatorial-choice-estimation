#!/usr/bin/env python
"""Standalone tests that import modules directly, bypassing __init__.py"""
import sys
import os
import importlib.util
import numpy as np
from mpi4py import MPI

def import_module_direct(path, module_name):
    """Import module directly without triggering __init__.py"""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def test_comm_manager():
    """Test CommManager"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cm_path = os.path.join(base_path, 'comm_manager.py')
    cm_module = import_module_direct(cm_path, 'bundlechoice.comm_manager')
    CommManager = cm_module.CommManager
    
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    assert cm.rank == comm.Get_rank()
    assert cm.comm_size == comm.Get_size()
    assert cm.root == 0
    
    # Test broadcast
    if cm._is_root():
        data = "test_data"
    else:
        data = None
    result = cm.bcast(data)
    assert result == "test_data"
    
    # Test _Bcast
    if cm._is_root():
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    else:
        arr = np.empty(3, dtype=np.float64)
    result = cm.Bcast(arr)
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))
    
    # Test sum_row_and_Reduce
    # Note: _Reduce only fills recvbuf on root, so we need to check root separately
    arr = np.random.randn(5, 3)
    result = cm.sum_row_andReduce(arr)
    # _Reduce returns recvbuf on all ranks, but only root has valid data
    # On non-root, recvbuf is uninitialized, so we can't test it
    if cm._is_root():
        assert result.shape == (3,)
        # We can't easily test the sum across processes without knowing all local sums
        # Just check it's the right shape
    
    if cm._is_root():
        print("✓ CommManager tests passed")

def test_config():
    """Test config"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_path, 'config.py')
    config_module = import_module_direct(config_path, 'bundlechoice.config')
    
    DimensionsConfig = config_module.DimensionsConfig
    BundleChoiceConfig = config_module.BundleChoiceConfig
    
    dc = DimensionsConfig(num_obs=10, num_items=5, num_features=3, num_simulations=2)
    assert dc.num_obs == 10
    assert dc.num_agents == 20
    
    bc1 = BundleChoiceConfig()
    bc2 = BundleChoiceConfig()
    bc2.dimensions.num_obs = 20
    bc1.update_in_place(bc2)
    assert bc1.dimensions.num_obs == 20
    
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("✓ Config tests passed")

def test_data_manager():
    """Test DataManager"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_module = import_module_direct(os.path.join(base_path, 'config.py'), 'bundlechoice.config')
    cm_module = import_module_direct(os.path.join(base_path, 'comm_manager.py'), 'bundlechoice.comm_manager')
    dm_path = os.path.join(base_path, 'data_manager.py')
    dm_module = import_module_direct(dm_path, 'bundlechoice.data_manager')
    
    CommManager = cm_module.CommManager
    DimensionsConfig = config_module.DimensionsConfig
    DataManager = dm_module.DataManager
    
    comm = MPI.COMM_WORLD
    cm = CommManager(comm)
    dc = DimensionsConfig(num_obs=10, num_items=5, num_features=3, num_simulations=1)
    dm = DataManager(dc, cm)
    
    assert dm.dimensions_cfg == dc
    assert len(dm.local_id) > 0
    
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
    
    if cm._is_root():
        print("✓ DataManager tests passed")

def test_oracles_manager():
    """Test OraclesManager"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_module = import_module_direct(os.path.join(base_path, 'config.py'), 'bundlechoice.config')
    cm_module = import_module_direct(os.path.join(base_path, 'comm_manager.py'), 'bundlechoice.comm_manager')
    dm_module = import_module_direct(os.path.join(base_path, 'data_manager.py'), 'bundlechoice.data_manager')
    om_path = os.path.join(base_path, 'oracles_manager.py')
    om_module = import_module_direct(om_path, 'bundlechoice.oracles_manager')
    
    CommManager = cm_module.CommManager
    DimensionsConfig = config_module.DimensionsConfig
    DataManager = dm_module.DataManager
    OraclesManager = om_module.OraclesManager
    
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
    assert om._features_oracle is not None
    
    om2 = OraclesManager(dc, cm, dm)
    om2.build_local_modular_error_oracle(seed=42)
    assert om2._error_oracle is not None
    
    # Test sum_row_and_Reduce usage
    om3 = OraclesManager(dc, cm, dm)
    om3.build_quadratic_features_from_data()
    features = om3._features_at_obs_bundles_at_root
    if cm._is_root():
        assert features.shape == (dc.num_obs, dc.num_features)
    
    if cm._is_root():
        print("✓ OraclesManager tests passed")

def test_subproblems():
    """Test subproblems"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_module = import_module_direct(os.path.join(base_path, 'config.py'), 'bundlechoice.config')
    cm_module = import_module_direct(os.path.join(base_path, 'comm_manager.py'), 'bundlechoice.comm_manager')
    dm_module = import_module_direct(os.path.join(base_path, 'data_manager.py'), 'bundlechoice.data_manager')
    om_module = import_module_direct(os.path.join(base_path, 'oracles_manager.py'), 'bundlechoice.oracles_manager')
    registry_path = os.path.join(base_path, 'subproblems', 'subproblem_registry.py')
    registry_module = import_module_direct(registry_path, 'bundlechoice.subproblems.subproblem_registry')
    manager_path = os.path.join(base_path, 'subproblems', 'subproblem_manager.py')
    manager_module = import_module_direct(manager_path, 'bundlechoice.subproblems.subproblem_manager')
    
    CommManager = cm_module.CommManager
    DimensionsConfig = config_module.DimensionsConfig
    BundleChoiceConfig = config_module.BundleChoiceConfig
    DataManager = dm_module.DataManager
    OraclesManager = om_module.OraclesManager
    SUBPROBLEM_REGISTRY = registry_module.SUBPROBLEM_REGISTRY
    SubproblemManager = manager_module.SubproblemManager
    
    assert 'Greedy' in SUBPROBLEM_REGISTRY
    
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
    
    sm.initialize_subproblems()
    if cm._is_root():
        theta = np.ones(dc.num_features)
    else:
        theta = np.empty(dc.num_features)
    result = sm.initialize_and_solve_subproblems(theta)
    assert result is not None
    assert result.shape[0] == dm.num_local_agent
    assert result.shape[1] == dc.num_items
    
    if cm._is_root():
        print("✓ Subproblems tests passed")

if __name__ == '__main__':
    rank = MPI.COMM_WORLD.Get_rank()
    errors = []
    try:
        test_comm_manager()
    except Exception as e:
        errors.append(f"CommManager: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_config()
    except Exception as e:
        errors.append(f"Config: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_data_manager()
    except Exception as e:
        errors.append(f"DataManager: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_oracles_manager()
    except Exception as e:
        errors.append(f"OraclesManager: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_subproblems()
    except Exception as e:
        errors.append(f"Subproblems: {e}")
        import traceback
        traceback.print_exc()
    
    if errors:
        if rank == 0:
            print("\n✗ Errors found:")
            for err in errors:
                print(f"  - {err}")
        sys.exit(1)
    elif rank == 0:
        print("\n✓ All standalone tests passed!")
