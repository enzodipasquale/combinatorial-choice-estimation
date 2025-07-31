import pytest
import tempfile
import os
import yaml
from bundlechoice.config import (
    DimensionsConfig, RowGenConfig, SubproblemConfig
)
from typing import Dict, Any

def load_configs_from_dict(config_dict: Dict[str, Any]):
    """Helper function to load configs from dict without requiring a bundle_choice instance."""
    dimensions_cfg = DimensionsConfig(**config_dict.get("dimensions", {}))
    rowgen_cfg = RowGenConfig(**config_dict.get("rowgen", {}))
    subproblem_cfg = SubproblemConfig(**config_dict.get("subproblem", {}))
    return dimensions_cfg, rowgen_cfg, subproblem_cfg

def load_configs_from_yaml(yaml_path: str):
    """Helper function to load configs from YAML file without requiring a bundle_choice instance."""
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return load_configs_from_dict(config_dict)

def test_load_configs_from_dict_partial():
    config_dict = {
        'dimensions': {
            'num_agents': 5,
            'num_items': 10,
            'num_features': 3,
            'num_simuls': 2,
        },
        'subproblem': {
            'name': 'test_subproblem',
            'settings': {'foo': 'bar'},
        }
    }
    dimensions_cfg, rowgen_cfg, subproblem_cfg = load_configs_from_dict(config_dict)
    assert rowgen_cfg.item_fixed_effects is False  # default

def test_load_configs_from_dict_full():
    config_dict = {
        'dimensions': {
            'num_agents': 5,
            'num_items': 10,
            'num_features': 3,
            'num_simuls': 2,
        },
        'rowgen': {
            'item_fixed_effects': True,
            'tol_certificate': 0.05,
            'max_iters': 50,
        },
        'subproblem': {
            'name': 'test_subproblem',
            'settings': {'foo': 'bar'},
        }
    }
    dimensions_cfg, rowgen_cfg, subproblem_cfg = load_configs_from_dict(config_dict)
    assert rowgen_cfg.item_fixed_effects is True

def test_load_configs_from_yaml():
    yaml_content = '''
dimensions:
  num_agents: 7
  num_items: 12
  num_features: 4
  num_simuls: 3
rowgen:
  item_fixed_effects: false
  tol_certificate: 0.02
  max_iters: 100
subproblem:
  name: greedy
  settings:
    alpha: 0.1
    beta: 0.2
'''
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(yaml_content)
        tmp_path = tmp.name
    try:
        dimensions_cfg, rowgen_cfg, subproblem_cfg = load_configs_from_yaml(tmp_path)
        assert dimensions_cfg.num_agents == 7
        assert dimensions_cfg.num_items == 12
        assert dimensions_cfg.num_features == 4
        assert dimensions_cfg.num_simuls == 3
        assert rowgen_cfg.item_fixed_effects is False
        assert rowgen_cfg.tol_certificate == 0.02
        assert rowgen_cfg.max_iters == 100
        assert subproblem_cfg.name == 'greedy'
        assert subproblem_cfg.settings == {'alpha': 0.1, 'beta': 0.2}
    finally:
        os.remove(tmp_path)

def load_dimensions_cfg(cfg: Dict[str, Any]) -> DimensionsConfig:
    return DimensionsConfig(**cfg.get("dimensions", {}))

def load_rowgen_config(cfg: Dict[str, Any]) -> RowGenConfig:
    return RowGenConfig(**cfg.get("rowgen", {}))

def load_subproblem_config(cfg: Dict[str, Any]) -> SubproblemConfig:
    return SubproblemConfig(**cfg.get("subproblem", {}))
