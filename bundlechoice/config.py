from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np
import yaml

@dataclass
class DimensionsConfig:
    """
    Configuration for the problem/data dimensions.
    """
    num_agents: Optional[int] = None
    num_items: Optional[int] = None
    num_features: Optional[int] = None
    num_simuls: Optional[int] = None
    
    def __post_init__(self):
        """
        Set default values after initialization.
        """
        if self.num_simuls is None:
            self.num_simuls = 1

@dataclass
class SubproblemConfig:
    """
    Configuration for subproblem algorithm options.
    """
    name: Optional[str] = None
    settings: dict = field(default_factory=dict)

@dataclass
class RowGenConfig:
    """
    Configuration for row generation solver options.
    """
    item_fixed_effects: bool = False
    tol_certificate: float = 0.01
    max_slack_counter: int = None  # Will be set to float('inf') if None
    tol_row_generation: float = 0.0
    row_generation_decay: float = 0.0
    max_iters: int = None  # Will be set to float('inf') if None
    min_iters: int = 0  
    master_settings: dict = field(default_factory=dict)
    master_ubs: dict = None
    master_lbs: list = None

    def __post_init__(self):
        if self.max_slack_counter is None:
            self.max_slack_counter = float('inf')
        if self.max_iters is None:
            self.max_iters = float('inf')
        if self.min_iters is None:
            self.min_iters = 0
        if self.tol_certificate is None:
            self.tol_certificate = 1e-6
        if self.tol_row_generation is None:
            self.tol_row_generation = 0.0
        if self.row_generation_decay is None:
            self.row_generation_decay = 0.0
        if self.master_settings is None:
            self.master_settings = {}

def load_config(bundle_choice, cfg: dict):
    """
    Loads configuration from a YAML file or dictionary.
    """
    # Accept YAML path or dict
    if isinstance(cfg, str):
        with open(cfg, "r") as f:
            cfg = yaml.safe_load(f)
    dimensions_cfg = DimensionsConfig(**cfg.get("dimensions", {}))
    rowgen_cfg = RowGenConfig(**cfg.get("rowgen", {}))
    subproblem_cfg = SubproblemConfig(**cfg.get("subproblem", {}))

    bundle_choice.dimensions_cfg = dimensions_cfg
    bundle_choice.rowgen_cfg = rowgen_cfg
    bundle_choice.subproblem_cfg = subproblem_cfg

    return bundle_choice 


