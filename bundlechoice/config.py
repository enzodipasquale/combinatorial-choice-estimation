from dataclasses import dataclass, field, fields
from typing import Optional, List, Any
import yaml


class AutoUpdateMixin:
    """
    Mixin class that provides automatic update_in_place functionality for dataclasses.
    This eliminates the need to implement update_in_place in every config class.
    """
    def update_in_place(self, other) -> None:
        """
        Update this configuration in place with values from another configuration.
        This method automatically updates all fields using dataclass reflection.
        
        Args:
            other: Configuration to merge from
        """
        for field in fields(self):
            if hasattr(self, field.name) and hasattr(other, field.name):
                current_value = getattr(self, field.name)
                other_value = getattr(other, field.name)
                
                if other_value is not None:
                    if isinstance(current_value, dict):
                        # Merge dictionaries
                        current_value.update(other_value)
                    elif hasattr(current_value, 'update_in_place') and hasattr(other_value, 'update_in_place'):
                        # Recursively update nested dataclasses
                        current_value.update_in_place(other_value)
                    else:
                        # Direct assignment for simple types
                        setattr(self, field.name, other_value)


@dataclass
class DimensionsConfig(AutoUpdateMixin):
    """
    Configuration for problem dimensions.
    
    This class defines the basic dimensions of the bundle choice problem,
    including the number of agents, items, features, and simulations.
    
    Attributes:
        num_agents: Number of agents in the problem
        num_items: Number of items available for choice
        num_features: Number of features per item
        num_simuls: Number of simulations to run
    """
    num_agents: Optional[int] = None
    num_items: Optional[int] = None
    num_features: Optional[int] = None
    num_simuls: int = 1


@dataclass
class SubproblemConfig(AutoUpdateMixin):
    """
    Configuration for subproblem solver settings.
    
    This class defines the configuration for the subproblem solver,
    including the algorithm name and any algorithm-specific settings.
    
    Attributes:
        name: Name of the subproblem algorithm to use
        settings: Dictionary of algorithm-specific settings
    """
    name: Optional[str] = None
    settings: dict = field(default_factory=dict)


@dataclass
class RowGenerationConfig(AutoUpdateMixin):
    """
    Configuration for row generation solver parameters.
    
    This class defines the parameters used by the row generation estimation
    algorithm, including tolerances, iteration limits, and convergence criteria.
    
    Attributes:
        tolerance_optimality: Tolerance for certificate validation
        max_slack_counter: Maximum slack counter value
        tol_row_generation: Tolerance for row generation convergence
        row_generation_decay: Decay factor for row generation
        max_iters: Maximum number of iterations
        min_iters: Minimum number of iterations
        gurobi_settings: Settings for the master problem solver
        theta_ubs: Upper bounds for theta parameters
        theta_lbs: Lower bounds for theta parameters
        parameters_to_log: List of parameter indices to log
    """
    tolerance_optimality: float = 1e-6
    max_slack_counter: float = float('inf')
    tol_row_generation: float = 0.0
    row_generation_decay: float = 0.0
    max_iters: float = float('inf')
    min_iters: int = 0
    gurobi_settings: dict = field(default_factory=dict)
    theta_ubs: Any = 1000
    theta_lbs: Any = None
    parameters_to_log: Optional[List[int]] = None


@dataclass
class EllipsoidConfig(AutoUpdateMixin):
    """
    Configuration for ellipsoid method solver parameters.
    
    This class defines the parameters used by the ellipsoid method estimation
    algorithm, including convergence criteria and numerical parameters.
    
    Attributes:
        max_iterations: Maximum number of iterations
        num_iters: Number of iterations to run (overrides convergence)
        tolerance: Convergence tolerance
        initial_radius: Initial radius of the ellipsoid
        decay_factor: Factor for radius decay
        min_volume: Minimum volume threshold
        verbose: Whether to print progress information
    """
    max_iterations: int = 1000
    num_iters: Optional[int] = None
    tolerance: float = 1e-6
    initial_radius: float = 1.0
    decay_factor: float = 0.95
    min_volume: float = 1e-12
    verbose: bool = True


@dataclass
class BundleChoiceConfig(AutoUpdateMixin):
    """
    Unified configuration for the BundleChoice framework.
    
    This class contains all configuration components needed for bundle choice
    estimation, including dimensions, subproblem settings, and solver parameters.
    
    Attributes:
        dimensions: Problem dimensions configuration
        subproblem: Subproblem algorithm configuration
        row_generation: Row generation solver configuration
        ellipsoid: Ellipsoid method solver configuration
    """
    dimensions: DimensionsConfig = field(default_factory=DimensionsConfig)
    subproblem: SubproblemConfig = field(default_factory=SubproblemConfig)
    row_generation: RowGenerationConfig = field(default_factory=RowGenerationConfig)
    ellipsoid: EllipsoidConfig = field(default_factory=EllipsoidConfig)

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
        
        if self.row_generation.tolerance_optimality <= 0:
            raise ValueError("tolerance_optimality must be positive")
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


