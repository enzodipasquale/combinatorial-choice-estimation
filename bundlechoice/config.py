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
    num_simuls: int = 1

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
    max_slack_counter: float = float('inf')
    tol_row_generation: float = 0.0
    row_generation_decay: float = 0.0
    max_iters: float = float('inf')
    min_iters: int = 0
    master_settings: dict = field(default_factory=dict)
    master_ubs: dict = None
    master_lbs: list = None

@dataclass
class BundleChoiceConfig:
    """
    Unified configuration for BundleChoice, containing all sub-configs.
    """
    dimensions: DimensionsConfig = field(default_factory=DimensionsConfig)
    subproblem: SubproblemConfig = field(default_factory=SubproblemConfig)
    rowgen: RowGenConfig = field(default_factory=RowGenConfig)

    @classmethod
    def from_dict(cls, cfg: dict):
        return cls(
            dimensions=DimensionsConfig(**cfg.get("dimensions", {})),
            subproblem=SubproblemConfig(**cfg.get("subproblem", {})),
            rowgen=RowGenConfig(**cfg.get("rowgen", {})),
        )

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        return cls.from_dict(cfg) 

    @classmethod
    def load(cls, cfg):
        """
        Load configuration from a YAML file path or a dictionary.

        Args:
            cfg (str or dict): Path to YAML file or configuration dictionary.

        Returns:
            BundleChoiceConfig: Loaded configuration object.
        """
        from pathlib import Path
        if isinstance(cfg, (str, Path)):
             cls.from_yaml(str(cfg))
        elif isinstance(cfg, dict):
            return cls.from_dict(cfg)
        else:
            raise ValueError("cfg must be a string, Path, or a dictionary.") 


