import pytest
import tempfile
import os
import yaml
from bundlechoice.config import (
    DimensionsConfig, row_generationerationConfig, SubproblemConfig
)
from typing import Dict, Any


def load_configs_from_dict(config_dict: Dict[str, Any]):
    """Load configs from dictionary without requiring a bundle_choice instance."""
    dimensions_cfg = DimensionsConfig(**config_dict.get("dimensions", {}))
    row_generation_cfg = row_generationerationConfig(**config_dict.get("row_generation", {}))
    subproblem_cfg = SubproblemConfig(**config_dict.get("subproblem", {}))
    return dimensions_cfg, row_generation_cfg, subproblem_cfg


def load_configs_from_yaml(yaml_path: str):
    """Load configs from YAML file without requiring a bundle_choice instance."""
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return load_configs_from_dict(config_dict)


def test_load_configs_from_dict_partial():
    """Test loading configs with partial configuration (missing row_generation)."""
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
    dimensions_cfg, row_generation_cfg, subproblem_cfg = load_configs_from_dict(config_dict)
    
    # Test default values are applied
    assert row_generation_cfg.tolerance_optimality == 0.01
    assert dimensions_cfg.num_agents == 5
    assert subproblem_cfg.name == 'test_subproblem'


def test_load_configs_from_dict_full():
    """Test loading configs with complete configuration."""
    config_dict = {
        'dimensions': {
            'num_agents': 5,
            'num_items': 10,
            'num_features': 3,
            'num_simuls': 2,
        },
        'row_generation': {
            'tolerance_optimality': 0.05,
            'max_iters': 50,
        },
        'subproblem': {
            'name': 'test_subproblem',
            'settings': {'foo': 'bar'},
        }
    }
    dimensions_cfg, row_generation_cfg, subproblem_cfg = load_configs_from_dict(config_dict)
    
    # Test all values are correctly loaded
    assert row_generation_cfg.tolerance_optimality == 0.05
    assert row_generation_cfg.max_iters == 50
    assert dimensions_cfg.num_agents == 5
    assert subproblem_cfg.name == 'test_subproblem'


def test_load_configs_from_yaml():
    """Test loading configs from YAML file."""
    yaml_content = '''
dimensions:
  num_agents: 7
  num_items: 12
  num_features: 4
  num_simuls: 3
row_generation:
  tolerance_optimality: 0.02
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
        dimensions_cfg, row_generation_cfg, subproblem_cfg = load_configs_from_yaml(tmp_path)
        
        # Test dimensions
        assert dimensions_cfg.num_agents == 7
        assert dimensions_cfg.num_items == 12
        assert dimensions_cfg.num_features == 4
        assert dimensions_cfg.num_simuls == 3
        
        # Test row generation
        assert row_generation_cfg.tolerance_optimality == 0.02
        assert row_generation_cfg.max_iters == 100
        
        # Test subproblem
        assert subproblem_cfg.name == 'greedy'
        assert subproblem_cfg.settings == {'alpha': 0.1, 'beta': 0.2}
    finally:
        os.remove(tmp_path)


# Helper functions for specific config loading
def load_dimensions_cfg(cfg: Dict[str, Any]) -> DimensionsConfig:
    """Load dimensions configuration from dictionary."""
    return DimensionsConfig(**cfg.get("dimensions", {}))


def load_row_generation_config(cfg: Dict[str, Any]) -> row_generationerationConfig:
    """Load row generation configuration from dictionary."""
    return row_generationerationConfig(**cfg.get("row_generation", {}))


def load_subproblem_config(cfg: Dict[str, Any]) -> SubproblemConfig:
    """Load subproblem configuration from dictionary."""
    return SubproblemConfig(**cfg.get("subproblem", {}))
