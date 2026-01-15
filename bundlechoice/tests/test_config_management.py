import pytest
import numpy as np
from bundlechoice.config import BundleChoiceConfig, DimensionsConfig, SubproblemConfig, EllipsoidConfig
from bundlechoice.core import BundleChoice

def test_config_properties():
    cfg = BundleChoiceConfig(dimensions=DimensionsConfig(num_obs=10, num_items=5, num_features=3), subproblem=SubproblemConfig(name='Greedy'), ellipsoid=EllipsoidConfig(max_iterations=25))
    assert cfg.subproblem is not None
    assert cfg.subproblem.name == 'Greedy'
    assert cfg.row_generation is not None
    assert cfg.row_generation.tolerance_optimality == 1e-06
    assert cfg.ellipsoid is not None
    assert cfg.ellipsoid.max_iterations == 25
    assert cfg.dimensions.num_obs == 10
    assert cfg.dimensions.num_items == 5
    assert cfg.dimensions.num_features == 3

def test_bundlechoice_config_integration():
    num_obs = 10
    num_items = 5
    num_features = 3
    cfg_dict = {'dimensions': {'num_obs': num_obs, 'num_items': num_items, 'num_features': num_features}, 'subproblem': {'name': 'Greedy'}, 'ellipsoid': {'max_iterations': 25}}
    bc = BundleChoice()
    bc.load_config(cfg_dict)
    assert bc.config is not None
    assert bc.config.dimensions is not None
    assert bc.config.dimensions.num_obs == num_obs
    assert bc.config.dimensions.num_items == num_items
    assert bc.config.dimensions.num_features == num_features
    assert bc.config.subproblem is not None
    assert bc.config.subproblem.name == 'Greedy'
    assert bc.config.ellipsoid is not None
    assert bc.config.ellipsoid.max_iterations == 25
    assert bc.data_manager is None
    bc._try_init_data_manager()
    assert bc.data_manager is not None