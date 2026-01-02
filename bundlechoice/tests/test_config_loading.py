import pytest
import tempfile
import os
import yaml
from bundlechoice.config import (
    DimensionsConfig, RowGenerationConfig, SubproblemConfig
)
from typing import Dict, Any


def load_configs_from_dict(config_dict: Dict[str, Any]):
    """Load configs from dictionary without requiring a bundle_choice instance."""
    from bundlechoice.config import BundleChoiceConfig
    # Use BundleChoiceConfig.from_dict to handle num_simulations → num_simulations conversion
    full_config = BundleChoiceConfig.from_dict(config_dict)
    return full_config.dimensions, full_config.row_generation, full_config.subproblem


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
            'num_simulations': 2,
        },
        'subproblem': {
            'name': 'test_subproblem',
            'settings': {'foo': 'bar'},
        }
    }
    dimensions_cfg, row_generation_cfg, subproblem_cfg = load_configs_from_dict(config_dict)
    
    # Test default values are applied
    assert row_generation_cfg.tolerance_optimality == 1e-6  # Updated default value
    assert dimensions_cfg.num_agents == 5
    assert subproblem_cfg.name == 'test_subproblem'


def test_load_configs_from_dict_full():
    """Test loading configs with complete configuration."""
    config_dict = {
        'dimensions': {
            'num_agents': 5,
            'num_items': 10,
            'num_features': 3,
            'num_simulations': 2,
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
  num_simulations: 3
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
        assert dimensions_cfg.num_simulations == 3
        
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


def load_row_generation_config(cfg: Dict[str, Any]) -> RowGenerationConfig:
    """Load row generation configuration from dictionary."""
    return RowGenerationConfig(**cfg.get("row_generation", {}))


def load_subproblem_config(cfg: Dict[str, Any]) -> SubproblemConfig:
    """Load subproblem configuration from dictionary."""
    return SubproblemConfig(**cfg.get("subproblem", {}))


def test_automatic_update_in_place():
    """Test that the automatic update_in_place method works correctly for all config types."""
    from bundlechoice.config import BundleChoiceConfig, EllipsoidConfig
    
    # Create initial config
    initial_config = BundleChoiceConfig(
        dimensions=DimensionsConfig(num_agents=10, num_items=5, num_features=3),
        subproblem=SubproblemConfig(name="greedy", settings={"alpha": 0.1}),
        row_generation=RowGenerationConfig(
            tolerance_optimality=1e-6,
            max_iters=100,
            theta_ubs=50,
            gurobi_settings={"Method": 0}
        ),
        ellipsoid=EllipsoidConfig(max_iterations=1000, tolerance=1e-6)
    )
    
    # Create update config with some new fields and some existing fields
    update_config = BundleChoiceConfig(
        dimensions=DimensionsConfig(num_agents=20, num_features=6),
        subproblem=SubproblemConfig(settings={"beta": 0.2}),
        row_generation=RowGenerationConfig(
            max_iters=200,
            theta_lbs=[-100] * 5,
            parameters_to_log=[0, 1, 2],
            gurobi_settings={"OutputFlag": 1}
        ),
        ellipsoid=EllipsoidConfig(verbose=False, decay_factor=0.9)
    )
    
    # Update the initial config
    initial_config.update_in_place(update_config)
    
    # Test that dimensions were updated correctly
    assert initial_config.dimensions.num_agents == 20  # Updated
    assert initial_config.dimensions.num_items == 5    # Unchanged
    assert initial_config.dimensions.num_features == 6 # Updated
    assert initial_config.dimensions.num_simulations == 1   # Default unchanged
    
    # Test that subproblem was updated correctly
    assert initial_config.subproblem.name == "greedy"  # Unchanged
    assert initial_config.subproblem.settings == {"alpha": 0.1, "beta": 0.2}  # Merged
    
    # Test that row generation was updated correctly
    assert initial_config.row_generation.tolerance_optimality == 1e-6  # Unchanged
    assert initial_config.row_generation.max_iters == 200  # Updated
    assert initial_config.row_generation.theta_ubs == 1000  # Default value (not specified in update)
    assert initial_config.row_generation.theta_lbs == [-100] * 5  # New field added
    assert initial_config.row_generation.parameters_to_log == [0, 1, 2]  # New field added
    assert initial_config.row_generation.gurobi_settings == {"Method": 0, "OutputFlag": 1}  # Merged
    
    # Test that ellipsoid was updated correctly
    assert initial_config.ellipsoid.max_iterations == 1000  # Unchanged
    assert initial_config.ellipsoid.tolerance == 1e-6       # Unchanged
    assert initial_config.ellipsoid.verbose == False        # Updated
    assert initial_config.ellipsoid.decay_factor == 0.9    # Updated


def test_nested_config_updates():
    """Test that nested config updates work correctly."""
    from bundlechoice.config import BundleChoiceConfig
    
    # Create config with nested settings
    config1 = BundleChoiceConfig(
        row_generation=RowGenerationConfig(
            gurobi_settings={"Method": 0, "Threads": 4}
        ),
        subproblem=SubproblemConfig(
            settings={"algorithm": "greedy", "tolerance": 1e-6}
        )
    )
    
    config2 = BundleChoiceConfig(
        row_generation=RowGenerationConfig(
            gurobi_settings={"OutputFlag": 1, "TimeLimit": 3600}
        ),
        subproblem=SubproblemConfig(
            settings={"max_iterations": 1000, "algorithm": "modified_greedy"}
        )
    )
    
    # Update config1 with config2
    config1.update_in_place(config2)
    
    # Test that dictionaries were merged correctly
    expected_gurobi = {"Method": 0, "Threads": 4, "OutputFlag": 1, "TimeLimit": 3600}
    assert config1.row_generation.gurobi_settings == expected_gurobi
    
    expected_settings = {"algorithm": "modified_greedy", "tolerance": 1e-6, "max_iterations": 1000}
    assert config1.subproblem.settings == expected_settings


def test_new_field_automatically_handled():
    """Test that adding new fields automatically works without updating update_in_place method."""
    from bundlechoice.config import BundleChoiceConfig
    
    # Create a config with existing fields
    config = BundleChoiceConfig(
        row_generation=RowGenerationConfig(
            tolerance_optimality=1e-6,
            max_iters=100
        )
    )
    
    # Create an update with existing fields (this should work automatically)
    update = BundleChoiceConfig(
        row_generation=RowGenerationConfig(
            max_iters=200,
            theta_lbs=[-100, -100, -100]  # Using existing field
        )
    )
    
    # The update should work automatically
    config.update_in_place(update)
    
    # Verify the update worked
    assert config.row_generation.max_iters == 200
    assert config.row_generation.theta_lbs == [-100, -100, -100]
    assert config.row_generation.tolerance_optimality == 1e-6  # Unchanged


def test_none_values_preserved():
    """Test that None values in the update config don't overwrite existing values."""
    from bundlechoice.config import BundleChoiceConfig
    
    config = BundleChoiceConfig(
        dimensions=DimensionsConfig(num_agents=10, num_items=5),
        row_generation=RowGenerationConfig(theta_lbs=[-50] * 5)
    )
    
    update = BundleChoiceConfig(
        dimensions=DimensionsConfig(num_agents=None, num_items=None),  # None values
        row_generation=RowGenerationConfig(theta_lbs=None)  # None value
    )
    
    config.update_in_place(update)
    
    # None values should not overwrite existing values
    assert config.dimensions.num_agents == 10
    assert config.dimensions.num_items == 5
    assert config.row_generation.theta_lbs == [-50] * 5
