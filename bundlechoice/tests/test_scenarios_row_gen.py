#!/usr/bin/env python
import sys, os, types
_base = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _base)

fake_bc = types.ModuleType('bundlechoice')
fake_bc.__path__ = [os.path.join(_base, 'bundlechoice')]
sys.modules['bundlechoice'] = fake_bc
fake_estimation = types.ModuleType('bundlechoice.estimation')
fake_estimation.__path__ = [os.path.join(_base, 'bundlechoice', 'estimation')]
sys.modules['bundlechoice.estimation'] = fake_estimation

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from bundlechoice.config import BundleChoiceConfig
from bundlechoice.comm_manager import CommManager
from bundlechoice.data_manager import DataManager
from bundlechoice.oracles_manager import OraclesManager
from bundlechoice.subproblems.subproblem_manager import SubproblemManager
from bundlechoice.estimation.row_generation import RowGenerationManager

def _test(name, n, m, k, settings, make_data):
    try:
        cfg = BundleChoiceConfig()
        cfg.dimensions.num_obs = n
        cfg.dimensions.num_items = m
        cfg.dimensions.num_features = k
        cfg.dimensions.num_simulations = 1
        cfg.subproblem.name = name
        cfg.subproblem.settings = settings or {}
        cfg.row_generation.max_iters = 2
        cfg.row_generation.min_iters = 1
        
        cm = CommManager(comm)
        dm = DataManager(cfg.dimensions, cm)
        om = OraclesManager(cfg.dimensions, cm, dm)
        sm = SubproblemManager(cm, cfg, dm, om)
        
        np.random.seed(42)
        # Root creates full data, non-root provides minimal placeholder
        if rank == 0:
            data = make_data(n, m, k)
        else:
            data = {'agent_data': {'obs_bundles': None}, 'item_data': {}}
        
        dm.load_input_data(data)
        om.build_quadratic_features_from_data()
        om.build_local_modular_error_oracle(seed=42)
        sm.load()
        
        rgm = RowGenerationManager(cm, cfg, dm, om, sm)
        result = rgm.solve(init_master=True, init_subproblems=True)
        
        if rank == 0:
            print(f"  {name}: PASS (iters={result.num_iterations})", flush=True)
        return True
    except Exception as e:
        if rank == 0:
            print(f"  {name}: FAIL - {type(e).__name__}: {e}", flush=True)
        return False

def _greedy_data(n, m, k):
    obs = (np.random.rand(n, m) > 0.5).astype(float)
    return {'agent_data': {'modular': np.random.randn(n, m, k), 'obs_bundles': obs}, 'item_data': {}, 'errors': np.random.randn(1, n, m)}

def _plain_data(n, m, k):
    obs = (np.random.rand(n, m) > 0.5).astype(float)
    return {'agent_data': {'modular': np.random.randn(n, m, k - 1), 'obs_bundles': obs}, 'item_data': {'modular': np.random.randn(m, 1)}, 'errors': np.random.randn(1, n, m)}

def _knapsack_data(n, m, k, quad=False):
    obs = (np.random.rand(n, m) > 0.5).astype(float)
    weights = np.random.randint(1, 10, m)
    data = {'agent_data': {'modular': np.abs(np.random.randn(n, m, k // 2)), 'capacity': np.full(n, int(0.5 * weights.sum())), 'obs_bundles': obs}, 'item_data': {'modular': np.abs(np.random.randn(m, k // 2)), 'weights': weights}, 'errors': np.random.randn(1, n, m)}
    if quad:
        data['agent_data']['quadratic'] = np.random.rand(n, m, m, 1) * 0.1
        data['item_data']['quadratic'] = np.random.rand(m, m, 1) * 0.1
    return data

def _supermod_data(n, m, k):
    obs = (np.random.rand(n, m) > 0.5).astype(float)
    quad = np.random.rand(m, m, k // 2) * 0.1
    np.fill_diagonal(quad[:, :, 0], 0)
    return {'agent_data': {'modular': -np.abs(np.random.randn(n, m, k // 2)), 'obs_bundles': obs}, 'item_data': {'modular': -np.abs(np.random.randn(m, k // 2)), 'quadratic': quad}, 'errors': np.random.randn(1, n, m) * 5}

if __name__ == '__main__':
    if rank == 0:
        print("Testing subproblem scenarios:", flush=True)
    
    tests = [
        ('Greedy', None, _greedy_data),
        ('PlainSingleItem', None, _plain_data),
        ('LinearKnapsack', {'TimeLimit': 1}, lambda n, m, k: _knapsack_data(n, m, k)),
        ('QuadKnapsack', {'TimeLimit': 1}, lambda n, m, k: _knapsack_data(n, m, k, True)),
        ('QuadSupermodularNetwork', None, _supermod_data),
    ]
    
    passed = 0
    for name, settings, data_fn in tests:
        if _test(name, 6, 5, 4, settings, data_fn):
            passed += 1
    
    if rank == 0:
        print(f"\nResult: {passed}/{len(tests)} passed", flush=True)
