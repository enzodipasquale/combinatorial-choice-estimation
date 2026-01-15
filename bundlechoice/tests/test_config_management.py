import pytest
import numpy as np
from bundlechoice.config import BundleChoiceConfig, DimensionsConfig, SubproblemConfig, EllipsoidConfig

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

def test_config_update_in_place():
    cfg = BundleChoiceConfig()
    cfg_dict = BundleChoiceConfig.from_dict({'dimensions': {'num_obs': 10, 'num_items': 5, 'num_features': 3}, 'subproblem': {'name': 'Greedy'}, 'ellipsoid': {'max_iterations': 25}})
    cfg.update_in_place(cfg_dict)
    assert cfg.dimensions.num_obs == 10
    assert cfg.dimensions.num_items == 5
    assert cfg.dimensions.num_features == 3
    assert cfg.subproblem.name == 'Greedy'
    assert cfg.ellipsoid.max_iterations == 25
