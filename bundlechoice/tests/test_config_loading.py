import pytest
import tempfile
import os
import yaml
from bundlechoice.config import DimensionsConfig, RowGenerationConfig, SubproblemConfig
from typing import Dict, Any

def load_configs_from_dict(config_dict: Dict[str, Any]):
    from bundlechoice.config import BundleChoiceConfig
    full_config = BundleChoiceConfig.from_dict(config_dict)
    return (full_config.dimensions, full_config.row_generation, full_config.subproblem)

def load_configs_from_yaml(yaml_path: str):
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return load_configs_from_dict(config_dict)

def test_load_configs_from_dict_partial():
    config_dict = {'dimensions': {'num_obs': 5, 'num_items': 10, 'num_features': 3, 'num_simulations': 2}, 'subproblem': {'name': 'test_subproblem', 'settings': {'foo': 'bar'}}}
    dimensions_cfg, row_generation_cfg, subproblem_cfg = load_configs_from_dict(config_dict)
    assert row_generation_cfg.tolerance_optimality == 1e-06
    assert dimensions_cfg.num_obs == 5
    assert subproblem_cfg.name == 'test_subproblem'

def test_load_configs_from_dict_full():
    config_dict = {'dimensions': {'num_obs': 5, 'num_items': 10, 'num_features': 3, 'num_simulations': 2}, 'row_generation': {'tolerance_optimality': 0.05, 'max_iters': 50}, 'subproblem': {'name': 'test_subproblem', 'settings': {'foo': 'bar'}}}
    dimensions_cfg, row_generation_cfg, subproblem_cfg = load_configs_from_dict(config_dict)
    assert row_generation_cfg.tolerance_optimality == 0.05
    assert row_generation_cfg.max_iters == 50
    assert dimensions_cfg.num_obs == 5
    assert subproblem_cfg.name == 'test_subproblem'

def test_load_configs_from_yaml():
    yaml_content = '\ndimensions:\n  num_obs: 7\n  num_items: 12\n  num_features: 4\n  num_simulations: 3\nrow_generation:\n  tolerance_optimality: 0.02\n  max_iters: 100\nsubproblem:\n  name: greedy\n  settings:\n    alpha: 0.1\n    beta: 0.2\n'
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(yaml_content)
        tmp_path = tmp.name
    try:
        dimensions_cfg, row_generation_cfg, subproblem_cfg = load_configs_from_yaml(tmp_path)
        assert dimensions_cfg.num_obs == 7
        assert dimensions_cfg.num_items == 12
        assert dimensions_cfg.num_features == 4
        assert dimensions_cfg.num_simulations == 3
        assert row_generation_cfg.tolerance_optimality == 0.02
        assert row_generation_cfg.max_iters == 100
        assert subproblem_cfg.name == 'greedy'
        assert subproblem_cfg.settings == {'alpha': 0.1, 'beta': 0.2}
    finally:
        os.remove(tmp_path)

def load_dimensions_cfg(cfg: Dict[str, Any]):
    return DimensionsConfig(**cfg.get('dimensions', {}))

def load_row_generation_config(cfg: Dict[str, Any]):
    return RowGenerationConfig(**cfg.get('row_generation', {}))

def load_subproblem_config(cfg: Dict[str, Any]):
    return SubproblemConfig(**cfg.get('subproblem', {}))

def test_automatic_update_in_place():
    from bundlechoice.config import BundleChoiceConfig, EllipsoidConfig
    initial_config = BundleChoiceConfig(dimensions=DimensionsConfig(num_obs=10, num_items=5, num_features=3), subproblem=SubproblemConfig(name='greedy', settings={'alpha': 0.1}), row_generation=RowGenerationConfig(tolerance_optimality=1e-06, max_iters=100, theta_ubs=50, gurobi_settings={'Method': 0}), ellipsoid=EllipsoidConfig(max_iterations=1000))
    update_config = BundleChoiceConfig(dimensions=DimensionsConfig(num_obs=20, num_features=6), subproblem=SubproblemConfig(settings={'beta': 0.2}), row_generation=RowGenerationConfig(max_iters=200, theta_lbs=[-100] * 5, parameters_to_log=[0, 1, 2], gurobi_settings={'OutputFlag': 1}), ellipsoid=EllipsoidConfig(verbose=False))
    initial_config.update_in_place(update_config)
    assert initial_config.dimensions.num_obs == 20
    assert initial_config.dimensions.num_items == 5
    assert initial_config.dimensions.num_features == 6
    assert initial_config.dimensions.num_simulations == 1
    assert initial_config.subproblem.name == 'greedy'
    assert initial_config.subproblem.settings == {'alpha': 0.1, 'beta': 0.2}
    assert initial_config.row_generation.tolerance_optimality == 1e-06
    assert initial_config.row_generation.max_iters == 200
    assert initial_config.row_generation.theta_ubs == 1000
    assert initial_config.row_generation.theta_lbs == [-100] * 5
    assert initial_config.row_generation.parameters_to_log == [0, 1, 2]
    assert initial_config.row_generation.gurobi_settings == {'Method': 0, 'OutputFlag': 1}
    assert initial_config.ellipsoid.max_iterations == 1000
    assert initial_config.ellipsoid.verbose == False

def test_nested_config_updates():
    from bundlechoice.config import BundleChoiceConfig
    config1 = BundleChoiceConfig(row_generation=RowGenerationConfig(gurobi_settings={'Method': 0, 'Threads': 4}), subproblem=SubproblemConfig(settings={'algorithm': 'greedy', 'tolerance': 1e-06}))
    config2 = BundleChoiceConfig(row_generation=RowGenerationConfig(gurobi_settings={'OutputFlag': 1, 'TimeLimit': 3600}), subproblem=SubproblemConfig(settings={'max_iterations': 1000, 'algorithm': 'modified_greedy'}))
    config1.update_in_place(config2)
    expected_gurobi = {'Method': 0, 'Threads': 4, 'OutputFlag': 1, 'TimeLimit': 3600}
    assert config1.row_generation.gurobi_settings == expected_gurobi
    expected_settings = {'algorithm': 'modified_greedy', 'tolerance': 1e-06, 'max_iterations': 1000}
    assert config1.subproblem.settings == expected_settings

def test_new_field_automatically_handled():
    from bundlechoice.config import BundleChoiceConfig
    config = BundleChoiceConfig(row_generation=RowGenerationConfig(tolerance_optimality=1e-06, max_iters=100))
    update = BundleChoiceConfig(row_generation=RowGenerationConfig(max_iters=200, theta_lbs=[-100, -100, -100]))
    config.update_in_place(update)
    assert config.row_generation.max_iters == 200
    assert config.row_generation.theta_lbs == [-100, -100, -100]
    assert config.row_generation.tolerance_optimality == 1e-06

def test_none_values_preserved():
    from bundlechoice.config import BundleChoiceConfig
    config = BundleChoiceConfig(dimensions=DimensionsConfig(num_obs=10, num_items=5), row_generation=RowGenerationConfig(theta_lbs=[-50] * 5))
    update = BundleChoiceConfig(dimensions=DimensionsConfig(num_obs=None, num_items=None), row_generation=RowGenerationConfig(theta_lbs=None))
    config.update_in_place(update)
    assert config.dimensions.num_obs == 10
    assert config.dimensions.num_items == 5
    assert config.row_generation.theta_lbs == [-50] * 5