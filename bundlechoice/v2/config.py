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
    max_slack_counter: Optional[int] = None  
    tol_row_generation: float = 0.0
    row_generation_decay: float = 0.0
    max_iters: Optional[int] = None
    min_iters: int = 0
    master_settings: dict = field(default_factory=dict)
    master_ubs: Optional[dict] = None
    master_lbs: Optional[list] = None

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


