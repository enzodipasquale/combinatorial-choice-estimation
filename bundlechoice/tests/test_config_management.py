import pytest
import numpy as np
from bundlechoice.base import HasConfig, HasDimensions
from bundlechoice.config import BundleChoiceConfig, DimensionsConfig, SubproblemConfig, EllipsoidConfig
from bundlechoice.core import BundleChoice


class ConfigurableClass(HasConfig, HasDimensions):
    """Test class that uses both HasConfig and HasDimensions mixins."""
    
    def __init__(self, config: BundleChoiceConfig):
        self.config = config
        self.dimensions_cfg = config.dimensions
        super().__init__()


def test_has_config_mixin():
    """Test that HasConfig mixin provides correct config access."""
    # Create test configuration
    cfg = BundleChoiceConfig(
        dimensions=DimensionsConfig(num_agents=10, num_items=5, num_features=3),
        subproblem=SubproblemConfig(name="Greedy"),
        ellipsoid=EllipsoidConfig(max_iterations=25)
    )
    
    # Create test object
    test_obj = ConfigurableClass(cfg)
    
    # Test config access through mixin properties
    assert test_obj.subproblem_cfg is not None
    assert test_obj.subproblem_cfg.name == "Greedy"
    
    assert test_obj.row_generation_cfg is not None
    assert test_obj.row_generation_cfg.tol_certificate == 0.01
    
    assert test_obj.ellipsoid_cfg is not None
    assert test_obj.ellipsoid_cfg.max_iterations == 25
    
    # Test dimension access through HasDimensions
    assert test_obj.num_agents == 10
    assert test_obj.num_items == 5
    assert test_obj.num_features == 3


def test_bundlechoice_config_integration():
    """Test that BundleChoice correctly integrates with the simplified config system."""
    # Create configuration
    num_agents = 10
    num_items = 5
    num_features = 3
    
    cfg_dict = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features
        },
        "subproblem": {
            "name": "Greedy"
        },
        "ellipsoid": {
            "max_iterations": 25
        }
    }
    
    # Create BundleChoice instance
    bc = BundleChoice()
    bc.load_config(cfg_dict)
    
    # Test config access through mixin
    assert bc.config is not None
    assert bc.config.dimensions is not None
    assert bc.config.dimensions.num_agents == num_agents
    assert bc.config.dimensions.num_items == num_items
    assert bc.config.dimensions.num_features == num_features
    
    assert bc.subproblem_cfg is not None
    assert bc.subproblem_cfg.name == "Greedy"
    
    assert bc.ellipsoid_cfg is not None
    assert bc.ellipsoid_cfg.max_iterations == 25
    
    # Test manager initialization with explicit config injection
    assert bc.data_manager is None
    bc._try_init_data_manager()
    assert bc.data_manager is not None 