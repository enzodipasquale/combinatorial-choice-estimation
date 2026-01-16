#!/usr/bin/env python
"""Test row generation with subproblem scenarios - manual setup to avoid broken imports"""
import sys
import os
import types

# Setup path
_base = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _base)

# Create a fake bundlechoice module to prevent __init__.py from loading
fake_bc = types.ModuleType('bundlechoice')
fake_bc.__path__ = [os.path.join(_base, 'bundlechoice')]
sys.modules['bundlechoice'] = fake_bc

# Also fake bundlechoice.estimation to skip standard_errors import
fake_estimation = types.ModuleType('bundlechoice.estimation')
fake_estimation.__path__ = [os.path.join(_base, 'bundlechoice', 'estimation')]
sys.modules['bundlechoice.estimation'] = fake_estimation

import numpy as np
from mpi4py import MPI

# Now import the actual modules we need
from bundlechoice.config import BundleChoiceConfig
from bundlechoice.comm_manager import CommManager
from bundlechoice.data_manager import DataManager
from bundlechoice.oracles_manager import OraclesManager
from bundlechoice.subproblems.subproblem_manager import SubproblemManager
from bundlechoice.estimation.row_generation import RowGenerationManager

def test_greedy():
    return _test_subproblem('Greedy', num_agents=10, num_items=8, num_features=4)

def test_plain_single_item():
    return _test_subproblem('PlainSingleItem', num_agents=10, num_items=8, num_features=4)

def test_linear_knapsack():
    return _test_subproblem_knapsack('LinearKnapsack', num_agents=10, num_items=8, num_features=4)

def test_quadratic_knapsack():
    return _test_subproblem_knapsack('QuadKnapsack', num_agents=10, num_items=8, num_features=4, quadratic=True)

def test_supermodular():
    return _test_subproblem_quadratic('QuadSupermodularNetwork', num_agents=10, num_items=8, num_features=4)

def _get_managers(comm, name, num_agents, num_items, num_features, settings=None):
    cfg = BundleChoiceConfig()
    cfg.dimensions.num_obs = num_agents
    cfg.dimensions.num_items = num_items
    cfg.dimensions.num_features = num_features
    cfg.dimensions.num_simulations = 1
    cfg.subproblem.name = name
    cfg.subproblem.settings = settings or {}
    cfg.row_generation.max_iters = 3
    cfg.row_generation.min_iters = 1
    
    cm = CommManager(comm)
    dm = DataManager(cfg.dimensions, cm)
    om = OraclesManager(cfg.dimensions, cm, dm)
    sm = SubproblemManager(cm, cfg, dm, om)
    
    return cfg, cm, dm, om, sm

def _test_subproblem(name, num_agents, num_items, num_features):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    try:
        cfg, cm, dm, om, sm = _get_managers(comm, name, num_agents, num_items, num_features)
        
        np.random.seed(42)
        if rank == 0:
            obs = (np.random.rand(num_agents, num_items) > 0.5).astype(float)
            data = {
                'agent_data': {'modular': np.random.randn(num_agents, num_items, num_features - 1), 'obs_bundles': obs},
                'item_data': {'modular': np.random.randn(num_items, 1)},
                'errors': np.random.randn(1, num_agents, num_items),
            }
        else:
            data = None
        
        dm.load_input_data(data)
        om.build_quadratic_features_from_data()
        om.build_local_modular_error_oracle(seed=42)
        sm.load()
        
        rgm = RowGenerationManager(cm, cfg, dm, om, sm)
        result = rgm.solve(init_master=True, init_subproblems=True)
        
        if rank == 0:
            print(f"  {name}: PASS (iters={result.num_iterations})")
        return True
    except Exception as e:
        if rank == 0:
            print(f"  {name}: FAIL - {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        return False

def _test_subproblem_knapsack(name, num_agents, num_items, num_features, quadratic=False):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    try:
        cfg, cm, dm, om, sm = _get_managers(comm, name, num_agents, num_items, num_features, {'TimeLimit': 5})
        
        np.random.seed(42)
        if rank == 0:
            weights = np.random.randint(1, 10, num_items)
            capacity = np.full(num_agents, int(0.5 * weights.sum()))
            obs = (np.random.rand(num_agents, num_items) > 0.5).astype(float)
            data = {
                'agent_data': {'modular': np.abs(np.random.randn(num_agents, num_items, 2)), 'capacity': capacity, 'obs_bundles': obs},
                'item_data': {'modular': np.abs(np.random.randn(num_items, 2)), 'weights': weights},
                'errors': np.random.randn(1, num_agents, num_items),
            }
            if quadratic:
                data['agent_data']['quadratic'] = np.random.rand(num_agents, num_items, num_items, 1) * 0.1
                data['item_data']['quadratic'] = np.random.rand(num_items, num_items, 1) * 0.1
        else:
            data = None
        
        dm.load_input_data(data)
        om.build_quadratic_features_from_data()
        om.build_local_modular_error_oracle(seed=42)
        sm.load()
        
        rgm = RowGenerationManager(cm, cfg, dm, om, sm)
        result = rgm.solve(init_master=True, init_subproblems=True)
        
        if rank == 0:
            print(f"  {name}: PASS (iters={result.num_iterations})")
        return True
    except Exception as e:
        if rank == 0:
            print(f"  {name}: FAIL - {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        return False

def _test_subproblem_quadratic(name, num_agents, num_items, num_features):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    try:
        cfg, cm, dm, om, sm = _get_managers(comm, name, num_agents, num_items, num_features)
        
        np.random.seed(42)
        if rank == 0:
            quad = np.random.rand(num_items, num_items, 2) * 0.1
            np.fill_diagonal(quad[:, :, 0], 0)
            np.fill_diagonal(quad[:, :, 1], 0)
            obs = (np.random.rand(num_agents, num_items) > 0.5).astype(float)
            data = {
                'agent_data': {'modular': -np.abs(np.random.randn(num_agents, num_items, 2)), 'obs_bundles': obs},
                'item_data': {'modular': -np.abs(np.random.randn(num_items, 2)), 'quadratic': quad},
                'errors': np.random.randn(1, num_agents, num_items) * 5,
            }
        else:
            data = None
        
        dm.load_input_data(data)
        om.build_quadratic_features_from_data()
        om.build_local_modular_error_oracle(seed=42)
        sm.load()
        
        rgm = RowGenerationManager(cm, cfg, dm, om, sm)
        result = rgm.solve(init_master=True, init_subproblems=True)
        
        if rank == 0:
            print(f"  {name}: PASS (iters={result.num_iterations})")
        return True
    except Exception as e:
        if rank == 0:
            print(f"  {name}: FAIL - {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        return False

if __name__ == '__main__':
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        print("Testing subproblem scenarios with row generation:")
    
    tests = [
        ('Greedy', test_greedy),
        ('PlainSingleItem', test_plain_single_item),
        ('LinearKnapsack', test_linear_knapsack),
        ('QuadKnapsack', test_quadratic_knapsack),
        ('QuadSupermodular', test_supermodular),
    ]
    
    results = {}
    for name, test_fn in tests:
        results[name] = test_fn()
    
    if rank == 0:
        passed = sum(results.values())
        total = len(results)
        print(f"\nResult: {passed}/{total} passed")
