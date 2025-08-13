from dataclasses import dataclass, field
from typing import Optional, List, Any
import numpy as np
import yaml

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
        tolerance_optimality: Tolerance for certificate validation
        max_slack_counter: Maximum slack counter value
        tol_row_generation: Tolerance for row generation convergence
        row_generation_decay: Decay factor for row generation
        max_iters: Maximum number of iterations
        min_iters: Minimum number of iterations
        gurobi_settings: Settings for the master problem solver
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
class EllipsoidConfig:
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
class BundleChoiceConfig:
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

    def merge(self, other: 'BundleChoiceConfig') -> 'BundleChoiceConfig':
        """
        Merge another configuration into this one, updating only fields that are set in other.
        
        Args:
            other: Configuration to merge from
            
        Returns:
            BundleChoiceConfig: New merged configuration
        """
        # Merge dimensions
        merged_dimensions = DimensionsConfig(
            num_agents=other.dimensions.num_agents if other.dimensions.num_agents is not None else self.dimensions.num_agents,
            num_items=other.dimensions.num_items if other.dimensions.num_items is not None else self.dimensions.num_items,
            num_features=other.dimensions.num_features if other.dimensions.num_features is not None else self.dimensions.num_features,
            num_simuls=other.dimensions.num_simuls
        )
        
        # Merge subproblem (replace if name is set, otherwise merge settings)
        if other.subproblem.name is not None:
            merged_subproblem = other.subproblem
        else:
            merged_settings = self.subproblem.settings.copy()
            merged_settings.update(other.subproblem.settings)
            merged_subproblem = SubproblemConfig(name=self.subproblem.name, settings=merged_settings)
        
        # Merge row generation
        merged_rowgen = RowGenerationConfig(
            tolerance_optimality=other.row_generation.tolerance_optimality,
            max_slack_counter=other.row_generation.max_slack_counter,
            tol_row_generation=other.row_generation.tol_row_generation,
            row_generation_decay=other.row_generation.row_generation_decay,
            max_iters=other.row_generation.max_iters,
            min_iters=other.row_generation.min_iters,
            gurobi_settings={**self.row_generation.gurobi_settings, **other.row_generation.gurobi_settings}
        )
        
        # Merge ellipsoid
        merged_ellipsoid = EllipsoidConfig(
            max_iterations=other.ellipsoid.max_iterations,
            num_iters=other.ellipsoid.num_iters if other.ellipsoid.num_iters is not None else self.ellipsoid.num_iters,
            tolerance=other.ellipsoid.tolerance,
            initial_radius=other.ellipsoid.initial_radius,
            decay_factor=other.ellipsoid.decay_factor,
            min_volume=other.ellipsoid.min_volume,
            verbose=other.ellipsoid.verbose
        )
        
        return BundleChoiceConfig(
            dimensions=merged_dimensions,
            subproblem=merged_subproblem,
            row_generation=merged_rowgen,
            ellipsoid=merged_ellipsoid
        ) 

    def update_in_place(self, other: 'BundleChoiceConfig') -> None:
        """
        Update this configuration in place with values from another configuration.
        This method modifies the existing object instead of creating a new one.
        
        Args:
            other: Configuration to merge from
        """
        # Update dimensions in place
        if other.dimensions.num_agents is not None:
            self.dimensions.num_agents = other.dimensions.num_agents
        if other.dimensions.num_items is not None:
            self.dimensions.num_items = other.dimensions.num_items
        if other.dimensions.num_features is not None:
            self.dimensions.num_features = other.dimensions.num_features
        self.dimensions.num_simuls = other.dimensions.num_simuls
        
        # Update subproblem in place
        if other.subproblem.name is not None:
            self.subproblem.name = other.subproblem.name
        self.subproblem.settings.update(other.subproblem.settings)
        
        # Update row generation in place
        self.row_generation.tolerance_optimality = other.row_generation.tolerance_optimality
        self.row_generation.max_slack_counter = other.row_generation.max_slack_counter
        self.row_generation.tol_row_generation = other.row_generation.tol_row_generation
        self.row_generation.row_generation_decay = other.row_generation.row_generation_decay
        self.row_generation.max_iters = other.row_generation.max_iters
        self.row_generation.min_iters = other.row_generation.min_iters
        self.row_generation.gurobi_settings.update(other.row_generation.gurobi_settings)
        
        # Update theta bounds
        if other.row_generation.theta_ubs is not None:
            self.row_generation.theta_ubs = other.row_generation.theta_ubs
        if other.row_generation.theta_lbs is not None:
            self.row_generation.theta_lbs = other.row_generation.theta_lbs
        
        # Update ellipsoid in place
        self.ellipsoid.max_iterations = other.ellipsoid.max_iterations
        if other.ellipsoid.num_iters is not None:
            self.ellipsoid.num_iters = other.ellipsoid.num_iters
        self.ellipsoid.tolerance = other.ellipsoid.tolerance
        self.ellipsoid.initial_radius = other.ellipsoid.initial_radius
        self.ellipsoid.decay_factor = other.ellipsoid.decay_factor
        self.ellipsoid.min_volume = other.ellipsoid.min_volume
        self.ellipsoid.verbose = other.ellipsoid.verbose

    def merge_in_place(self, other: 'BundleChoiceConfig') -> 'BundleChoiceConfig':
        """
        Convenience method that calls update_in_place and returns self for chaining.
        
        Args:
            other: Configuration to merge from
            
        Returns:
            BundleChoiceConfig: self (for method chaining)
        """
        self.update_in_place(other)
        return self 


