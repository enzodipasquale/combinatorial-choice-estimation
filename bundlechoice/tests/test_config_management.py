import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.base import HasConfig, HasDimensions
from bundlechoice.config import BundleChoiceConfig, DimensionsConfig, SubproblemConfig, EllipsoidConfig

class TestConfigurableClass(HasConfig, HasDimensions):
    """Test class that uses both HasConfig and HasDimensions mixins."""
    
    def __init__(self):
        super().__init__()
        self.config = None
        self.dimensions_cfg = None
    
    def test_config_access(self):
        """Test that config access works correctly."""
        # Test that properties work when config is None
        assert self.dimensions_cfg is None
        assert self.subproblem_cfg is None
        assert self.ellipsoid_cfg is None
        assert self.num_agents is None
        
        # Test with actual config
        cfg = {
            "dimensions": {
                "num_agents": 100,
                "num_items": 20,
                "num_features": 6,
                "num_simuls": 1,
            },
            "subproblem": {
                "name": "Greedy",
            },
            "ellipsoid": {
                "max_iterations": 50,
                "tolerance": 1e-6,
            }
        }
        
        self.config = BundleChoiceConfig.from_dict(cfg)
        self.dimensions_cfg = self.config.dimensions
        
        # Test that properties now return correct values
        assert self.dimensions_cfg is not None
        assert self.dimensions_cfg.num_agents == 100
        assert self.dimensions_cfg.num_items == 20
        assert self.num_agents == 100
        assert self.num_items == 20
        assert self.num_features == 6
        
        assert self.subproblem_cfg is not None
        assert self.subproblem_cfg.name == "Greedy"
        
        assert self.ellipsoid_cfg is not None
        assert self.ellipsoid_cfg.max_iterations == 50
        assert self.ellipsoid_cfg.tolerance == 1e-6

def test_has_config_mixin():
    """Test the HasConfig mixin functionality."""
    test_obj = TestConfigurableClass()
    test_obj.test_config_access()

def test_bundlechoice_config_integration():
    """Test that BundleChoice works correctly with the new config system."""
    num_agents = 50
    num_items = 10
    num_features = 4
    
    cfg = {
        "dimensions": {
            "num_agents": num_agents,
            "num_items": num_items,
            "num_features": num_features,
            "num_simuls": 1,
        },
        "subproblem": {
            "name": "Greedy",
        },
        "ellipsoid": {
            "max_iterations": 25,
            "tolerance": 1e-5,
            "initial_radius": 1.0,
            "decay_factor": 0.9,
            "min_volume": 1e-10,
            "verbose": False
        }
    }
    
    # Test BundleChoice with new config system
    bc = BundleChoice()
    bc.load_config(cfg)
    
    # Test that config access works through the mixin
    assert bc.config is not None
    assert bc.config.dimensions is not None
    assert bc.config.dimensions.num_agents == num_agents
    assert bc.config.dimensions.num_items == num_items
    assert bc.config.dimensions.num_features == num_features
    
    assert bc.subproblem_cfg is not None
    assert bc.subproblem_cfg.name == "Greedy"
    
    assert bc.ellipsoid_cfg is not None
    assert bc.ellipsoid_cfg.max_iterations == 25
    assert bc.ellipsoid_cfg.tolerance == 1e-5
    assert bc.ellipsoid_cfg.initial_radius == 1.0
    assert bc.ellipsoid_cfg.decay_factor == 0.9
    assert bc.ellipsoid_cfg.min_volume == 1e-10
    assert bc.ellipsoid_cfg.verbose == False

def test_config_registry():
    """Test the ConfigRegistry functionality."""
    from bundlechoice.config import ConfigRegistry, DimensionsConfig, SubproblemConfig
    
    registry = ConfigRegistry()
    
    # Register configurations
    dim_cfg = DimensionsConfig(num_agents=100, num_items=20)
    sub_cfg = SubproblemConfig(name="Greedy")
    
    registry.register("dimensions", dim_cfg)
    registry.register("subproblem", sub_cfg)
    
    # Test retrieval
    retrieved_dim = registry.get("dimensions")
    assert retrieved_dim is not None
    assert retrieved_dim.num_agents == 100
    
    retrieved_sub = registry.get("subproblem")
    assert retrieved_sub is not None
    assert retrieved_sub.name == "Greedy"
    
    # Test typed retrieval
    typed_dim = registry.get_typed("dimensions", DimensionsConfig)
    assert typed_dim is not None
    assert isinstance(typed_dim, DimensionsConfig)
    
    # Test attribute access
    assert registry.dimensions is not None
    assert registry.subproblem is not None
    
    # Test missing config
    assert registry.get("nonexistent") is None
    with pytest.raises(AttributeError):
        _ = registry.nonexistent

def test_bundlechoice_config_registry():
    """Test that BundleChoiceConfig registry works correctly."""
    cfg = {
        "dimensions": {
            "num_agents": 75,
            "num_items": 15,
            "num_features": 5,
        },
        "subproblem": {
            "name": "LinearKnapsack",
        },
        "ellipsoid": {
            "max_iterations": 30,
        }
    }
    
    config = BundleChoiceConfig.from_dict(cfg)
    
    # Test registry access
    dim_cfg = config.get_config("dimensions")
    assert dim_cfg is not None
    assert dim_cfg.num_agents == 75
    
    sub_cfg = config.get_config("subproblem")
    assert sub_cfg is not None
    assert sub_cfg.name == "LinearKnapsack"
    
    ellipsoid_cfg = config.get_config("ellipsoid")
    assert ellipsoid_cfg is not None
    assert ellipsoid_cfg.max_iterations == 30
    
    # Test typed access
    typed_dim = config.get_typed_config("dimensions", DimensionsConfig)
    assert typed_dim is not None
    assert isinstance(typed_dim, DimensionsConfig)
    
    # Test validation
    validation_results = config.validate_configs()
    assert validation_results["dimensions"] is True
    assert validation_results["subproblem"] is True
    assert validation_results["ellipsoid"] is True 