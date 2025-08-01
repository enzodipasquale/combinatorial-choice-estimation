from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Type, TypeVar
import numpy as np
import yaml

T = TypeVar('T')

class ConfigRegistry:
    """
    Registry for configuration components that provides dynamic access and validation.
    
    This class allows for flexible configuration management where components
    can register their configuration types and access them dynamically.
    """
    
    def __init__(self):
        self._configs: Dict[str, Any] = {}
        self._validators: Dict[str, callable] = {}
    
    def register(self, name: str, config: Any, validator: Optional[callable] = None):
        """Register a configuration component."""
        self._configs[name] = config
        if validator:
            self._validators[name] = validator
    
    def get(self, name: str, default=None):
        """Get a configuration component by name."""
        return self._configs.get(name, default)
    
    def get_typed(self, name: str, config_type: Type[T]) -> Optional[T]:
        """Get a configuration component with type checking."""
        config = self.get(name)
        if config is not None and isinstance(config, config_type):
            return config
        return None
    
    def validate(self, name: str):
        """Validate a configuration component."""
        if name in self._validators:
            return self._validators[name](self._configs.get(name))
        return True
    
    def validate_all(self):
        """Validate all registered configurations."""
        results = {}
        for name in self._configs:
            results[name] = self.validate(name)
        return results
    
    def __getattr__(self, name: str):
        """Allow attribute-style access to configurations."""
        if name in self._configs:
            return self._configs[name]
        raise AttributeError(f"Configuration '{name}' not found")
    
    def __contains__(self, name: str):
        """Check if a configuration exists."""
        return name in self._configs

@dataclass
class DimensionsConfig:
    """
    Configuration for problem dimensions and data structure.
    
    This class defines the key dimensions of the bundle choice problem,
    including the number of agents, items, features, and simulation runs.
    
    Attributes:
        num_agents: Number of agents in the problem
        num_items: Number of items available for choice
        num_features: Number of features per agent-item combination
        num_simuls: Number of simulation runs (default: 1)
    """
    num_agents: Optional[int] = None
    num_items: Optional[int] = None
    num_features: Optional[int] = None
    num_simuls: int = 1

@dataclass
class SubproblemConfig:
    """
    Configuration for subproblem algorithm settings.
    
    This class defines the subproblem algorithm to use and its specific
    configuration parameters.
    
    Attributes:
        name: Name of the subproblem algorithm
        settings: Dictionary of algorithm-specific settings
    """
    name: Optional[str] = None
    settings: dict = field(default_factory=dict)

@dataclass
class RowGenerationConfig:
    """
    Configuration for row generation solver parameters.
    
    This class defines the parameters used by the row generation estimation
    algorithm, including tolerances, iteration limits, and convergence criteria.
    
    Attributes:
        tol_certificate: Tolerance for certificate validation
        max_slack_counter: Maximum slack counter value
        tol_row_generation: Tolerance for row generation convergence
        row_generation_decay: Decay factor for row generation
        max_iters: Maximum number of iterations
        min_iters: Minimum number of iterations
        master_settings: Settings for the master problem solver
    """
    tol_certificate: float = 0.01
    max_slack_counter: float = float('inf')
    tol_row_generation: float = 0.0
    row_generation_decay: float = 0.0
    max_iters: float = float('inf')
    min_iters: int = 0
    master_settings: dict = field(default_factory=dict)

@dataclass
class EllipsoidConfig:
    """
    Configuration for ellipsoid method solver parameters.
    
    This class defines the parameters used by the ellipsoid method estimation
    algorithm, including convergence criteria and numerical parameters.
    
    Attributes:
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        initial_radius: Initial radius of the ellipsoid
        decay_factor: Factor for radius decay
        min_volume: Minimum volume threshold
        verbose: Whether to print progress information
    """
    max_iterations: int = 1000
    tolerance: float = 1e-6
    initial_radius: float = 1.0
    decay_factor: float = 0.95
    min_volume: float = 1e-12
    verbose: bool = True

@dataclass
class BundleChoiceConfig:
    """
    Unified configuration for the BundleChoice framework.
    
    This class contains all configuration components needed for bundle choice
    estimation, including dimensions, subproblem settings, and solver parameters.
    Uses a registry pattern for flexible configuration management.
    
    Attributes:
        registry: Configuration registry containing all components
        dimensions: Problem dimensions configuration
        subproblem: Subproblem algorithm configuration
        row_generation: Row generation solver configuration
        ellipsoid: Ellipsoid method solver configuration
    """
    registry: ConfigRegistry = field(default_factory=ConfigRegistry)
    dimensions: DimensionsConfig = field(default_factory=DimensionsConfig)
    subproblem: SubproblemConfig = field(default_factory=SubproblemConfig)
    row_generation: RowGenerationConfig = field(default_factory=RowGenerationConfig)
    ellipsoid: EllipsoidConfig = field(default_factory=EllipsoidConfig)

    def __post_init__(self):
        """Initialize the registry with all configuration components."""
        self.registry.register("dimensions", self.dimensions)
        self.registry.register("subproblem", self.subproblem)
        self.registry.register("row_generation", self.row_generation)
        self.registry.register("ellipsoid", self.ellipsoid)

    def get_config(self, name: str, default=None):
        """Get a configuration component by name."""
        return self.registry.get(name, default)

    def get_typed_config(self, name: str, config_type: Type[T]) -> Optional[T]:
        """Get a configuration component with type checking."""
        return self.registry.get_typed(name, config_type)

    def register_config(self, name: str, config: Any, validator: Optional[callable] = None):
        """Register a new configuration component."""
        self.registry.register(name, config, validator)
        # Also set as attribute for backward compatibility
        setattr(self, name, config)

    def validate_configs(self):
        """Validate all configuration components."""
        return self.registry.validate_all()

    @classmethod
    def from_dict(cls, cfg: dict):
        """
        Create configuration from a dictionary.
        
        Args:
            cfg: Configuration dictionary
            
        Returns:
            BundleChoiceConfig: Configuration instance
        """
        return cls(
            dimensions=DimensionsConfig(**cfg.get("dimensions", {})),
            subproblem=SubproblemConfig(**cfg.get("subproblem", {})),
            row_generation=RowGenerationConfig(**cfg.get("row_generation", {})),
            ellipsoid=EllipsoidConfig(**cfg.get("ellipsoid", {})),
        )

    @classmethod
    def from_yaml(cls, path: str):
        """
        Create configuration from a YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            BundleChoiceConfig: Configuration instance
        """
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        return cls.from_dict(cfg) 

    @classmethod
    def load(cls, cfg):
        """
        Load configuration from a YAML file path or a dictionary.
        
        This method provides a unified interface for loading configuration
        from either a file path or a dictionary.

        Args:
            cfg: Path to YAML file, Path object, or configuration dictionary

        Returns:
            BundleChoiceConfig: Loaded configuration object
            
        Raises:
            ValueError: If cfg is not a string, Path, or dictionary
        """
        from pathlib import Path
        if isinstance(cfg, (str, Path)):
            return cls.from_yaml(str(cfg))
        elif isinstance(cfg, dict):
            return cls.from_dict(cfg)
        else:
            raise ValueError("cfg must be a string, Path, or a dictionary.")

    def validate(self):
        """
        Validate the configuration parameters.
        
        This method checks that all configuration parameters are within
        valid ranges and raises appropriate errors for invalid values.
        
        Raises:
            ValueError: If any parameters are invalid
        """
        if self.dimensions.num_agents is not None and self.dimensions.num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if self.dimensions.num_items is not None and self.dimensions.num_items <= 0:
            raise ValueError("num_items must be positive")
        if self.dimensions.num_features is not None and self.dimensions.num_features <= 0:
            raise ValueError("num_features must be positive")
        if self.dimensions.num_simuls <= 0:
            raise ValueError("num_simuls must be positive")
        
        if self.row_generation.tol_certificate <= 0:
            raise ValueError("tol_certificate must be positive")
        if self.row_generation.max_iters <= 0:
            raise ValueError("max_iters must be positive")
        if self.row_generation.min_iters < 0:
            raise ValueError("min_iters must be non-negative")
        
        if self.ellipsoid.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.ellipsoid.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.ellipsoid.initial_radius <= 0:
            raise ValueError("initial_radius must be positive")
        if not 0 < self.ellipsoid.decay_factor < 1:
            raise ValueError("decay_factor must be between 0 and 1")
        if self.ellipsoid.min_volume <= 0:
            raise ValueError("min_volume must be positive") 


