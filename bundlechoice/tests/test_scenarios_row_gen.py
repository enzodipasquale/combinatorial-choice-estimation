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

def log(msg):
    print(f"[{rank}] {msg}", flush=True)

def _greedy_data(n, m, k):
    obs = (np.random.rand(n, m) > 0.5).astype(float)
    return {'agent_data': {'modular': np.random.randn(n, m, k), 'obs_bundles': obs}, 'item_data': {}, 'errors': np.random.randn(1, n, m)}

if __name__ == '__main__':
    n, m, k = 6, 5, 4
    
    cfg = BundleChoiceConfig()
    cfg.dimensions.num_obs = n
    cfg.dimensions.num_items = m
    cfg.dimensions.num_features = k
    cfg.dimensions.num_simulations = 1
    cfg.subproblem.name = 'Greedy'
    cfg.row_generation.max_iters = 2
    cfg.row_generation.min_iters = 1
    
    cm = CommManager(comm)
    dm = DataManager(cfg.dimensions, cm)
    om = OraclesManager(cfg.dimensions, cm, dm)
    sm = SubproblemManager(cm, cfg, dm, om)
    
    np.random.seed(42)
    data = _greedy_data(n, m, k) if rank == 0 else {'agent_data': {'obs_bundles': None}, 'item_data': {}}
    
    log("load_input_data")
    dm.load_input_data(data)
    
    log("build_quadratic_features_from_data")
    om.build_quadratic_features_from_data()
    
    log("build_local_modular_error_oracle")
    om.build_local_modular_error_oracle(seed=42)
    
    log("sm.load")
    sm.load()
    
    log("RowGenerationManager init")
    rgm = RowGenerationManager(cm, cfg, dm, om, sm)
    
    log("rgm.solve")
    result = rgm.solve(init_master=True, init_subproblems=True, agent_weights=np.ones(n))
    
    log(f"DONE iters={result.num_iterations}")
