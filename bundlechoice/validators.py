"""
Validation utilities for BundleChoice framework.

This module centralizes all validation logic for setup, configuration, and data.
It provides a single source of truth for validation across the framework.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from bundlechoice.errors import (
    SetupError, ValidationError, DimensionMismatchError, 
    DataError, ConfigurationError
)


@dataclass
class ValidationResult:
    """
    Result of a validation check.
    
    Attributes:
        valid (bool): Whether validation passed
        error_type (type): Exception class to raise if invalid
        message (str): Error message
        suggestion (str): Suggested fix
        missing (list): List of missing components
        details (dict): Additional details
    """
    valid: bool
    error_type: type = None
    message: str = ""
    suggestion: str = None
    missing: list = None
    details: dict = None
    
    def __post_init__(self):
        if self.missing is None:
            self.missing = []
        if self.details is None:
            self.details = {}


class SetupValidator:
    """
    Validates BundleChoice setup and component initialization.
    
    This class provides validation methods for checking prerequisites
    before initializing each manager component.
    """
    
    @staticmethod
    def validate_for_data_manager(config) -> ValidationResult:
        """
        Check if config has required dimensions for DataManager.
        
        Args:
            config: BundleChoiceConfig instance
            
        Returns:
            ValidationResult indicating success or failure
        """
        if config is None or config.dimensions is None:
            return ValidationResult(
                valid=False,
                error_type=SetupError,
                message="Cannot initialize data manager - dimensions configuration not loaded",
                suggestion="Call bc.load_config(config_dict) with 'dimensions' section before accessing bc.data"
            )
        
        return ValidationResult(valid=True)
    
    @staticmethod
    def validate_for_feature_manager(config, data_manager) -> ValidationResult:
        """
        Check prerequisites for FeatureManager.
        
        Args:
            config: BundleChoiceConfig instance
            data_manager: DataManager instance (can be None)
            
        Returns:
            ValidationResult indicating success or failure
        """
        missing = []
        
        if config is None or config.dimensions is None:
            missing.append("dimensions config")
        
        if data_manager is None:
            missing.append("data manager")
        
        if missing:
            return ValidationResult(
                valid=False,
                error_type=SetupError,
                message="Cannot initialize feature manager - missing prerequisites",
                suggestion=(
                    "Complete these steps:\n"
                    "  1. bc.load_config(config_dict)  # Include 'dimensions' section\n"
                    "  2. bc.data.load_and_scatter(input_data)"
                ),
                missing=missing
            )
        
        return ValidationResult(valid=True)
    
    @staticmethod
    def validate_for_subproblem_manager(config, data_manager, feature_manager) -> ValidationResult:
        """
        Check prerequisites for SubproblemManager.
        
        Args:
            config: BundleChoiceConfig instance
            data_manager: DataManager instance
            feature_manager: FeatureManager instance
            
        Returns:
            ValidationResult indicating success or failure
        """
        missing = []
        
        if config is None or config.subproblem is None:
            missing.append("config with 'subproblem' section")
        
        if data_manager is None:
            missing.append("data")
        
        if feature_manager is None:
            missing.append("features")
        
        if missing:
            return ValidationResult(
                valid=False,
                error_type=SetupError,
                message="Cannot initialize subproblem manager - missing prerequisites",
                suggestion=(
                    "Complete these steps:\n"
                    "  1. bc.load_config(config_dict)  # Include 'subproblem' section\n"
                    "  2. bc.data.load_and_scatter(input_data)\n"
                    "  3. bc.features.build_from_data() or bc.features.set_oracle(fn)"
                ),
                missing=missing
            )
        
        return ValidationResult(valid=True)
    
    @staticmethod
    def validate_for_row_generation(config, data_manager, feature_manager, subproblem_manager) -> ValidationResult:
        """
        Check prerequisites for RowGenerationSolver.
        
        Args:
            config: BundleChoiceConfig instance
            data_manager: DataManager instance
            feature_manager: FeatureManager instance
            subproblem_manager: SubproblemManager instance
            
        Returns:
            ValidationResult indicating success or failure
        """
        missing = []
        
        if data_manager is None:
            missing.append("data (call bc.data.load_and_scatter(input_data))")
        
        if feature_manager is None:
            missing.append("features (call bc.features.set_oracle(fn) or bc.features.build_from_data())")
        
        if subproblem_manager is None:
            missing.append("subproblem (call bc.subproblems.load())")
        
        if config is None or config.row_generation is None:
            missing.append("row_generation config (add 'row_generation' to your config)")
        
        if missing:
            return ValidationResult(
                valid=False,
                error_type=SetupError,
                message="Cannot initialize row generation solver - missing setup",
                suggestion=(
                    "Missing components:\n  " +
                    "\n  ".join(missing) +
                    "\n\nRun bc.validate_setup('row_generation') to check your configuration."
                ),
                missing=missing
            )
        
        return ValidationResult(valid=True)
    
    @staticmethod
    def validate_for_ellipsoid(config, data_manager, feature_manager, subproblem_manager) -> ValidationResult:
        """
        Check prerequisites for EllipsoidSolver.
        
        Args:
            config: BundleChoiceConfig instance
            data_manager: DataManager instance
            feature_manager: FeatureManager instance
            subproblem_manager: SubproblemManager instance
            
        Returns:
            ValidationResult indicating success or failure
        """
        missing = []
        
        if data_manager is None:
            missing.append("data (call bc.data.load_and_scatter(input_data))")
        
        if feature_manager is None:
            missing.append("features (call bc.features.set_oracle(fn) or bc.features.build_from_data())")
        
        if subproblem_manager is None:
            missing.append("subproblem (call bc.subproblems.load())")
        
        if config is None or config.ellipsoid is None:
            missing.append("ellipsoid config (add 'ellipsoid' to your config)")
        
        if missing:
            return ValidationResult(
                valid=False,
                error_type=SetupError,
                message="Cannot initialize ellipsoid solver - missing setup",
                suggestion=(
                    "Missing components:\n  " +
                    "\n  ".join(missing) +
                    "\n\nRun bc.validate_setup('ellipsoid') to check your configuration."
                ),
                missing=missing
            )
        
        return ValidationResult(valid=True)
    
    @staticmethod
    def validate_for_inequalities(config, data_manager, feature_manager, subproblem_manager) -> ValidationResult:
        """
        Check prerequisites for InequalitiesSolver.
        
        Args:
            config: BundleChoiceConfig instance
            data_manager: DataManager instance  
            feature_manager: FeatureManager instance
            subproblem_manager: SubproblemManager instance
            
        Returns:
            ValidationResult indicating success or failure
        """
        missing_managers = []
        
        if data_manager is None:
            missing_managers.append("DataManager")
        
        if feature_manager is None:
            missing_managers.append("FeatureManager")
        
        if subproblem_manager is None:
            missing_managers.append("SubproblemManager")
        
        if config is None or config.dimensions is None:
            missing_managers.append("DimensionsConfig")
        
        if missing_managers:
            return ValidationResult(
                valid=False,
                error_type=SetupError,
                message="Cannot initialize inequalities solver - missing prerequisites",
                suggestion=(
                    f"Missing: {', '.join(missing_managers)}\n\n"
                    "Ensure you've called:\n"
                    "  1. bc.load_config(config_dict)\n"
                    "  2. bc.data.load_and_scatter(input_data)\n"
                    "  3. bc.features.set_oracle(fn)"
                ),
                missing=missing_managers
            )
        
        return ValidationResult(valid=True)


class DataValidator:
    """
    Validates input data structure and values.
    
    This class provides comprehensive validation for input data including
    dimension checks, structure validation, and value range checks.
    """
    
    @staticmethod
    def validate_agent_data(agent_data: Dict[str, np.ndarray], num_agents: int, num_items: int) -> None:
        """
        Validate agent-specific data arrays.
        
        Args:
            agent_data: Dictionary containing agent data arrays
            num_agents: Expected number of agents
            num_items: Expected number of items
            
        Raises:
            DimensionMismatchError: If dimensions don't match
            DataError: If data contains invalid values
        """
        if agent_data is None:
            return
        
        for key, arr in agent_data.items():
            if not isinstance(arr, np.ndarray):
                raise ValidationError(
                    f"agent_data['{key}'] must be a numpy array, got {type(arr).__name__}"
                )
            
            # Check dimensions
            if key == "modular":
                if arr.ndim != 3 or arr.shape[0] != num_agents or arr.shape[1] != num_items:
                    raise DimensionMismatchError(
                        f"agent_data['modular'] has incorrect shape",
                        expected={"shape": f"({num_agents}, {num_items}, num_features)"},
                        actual={"shape": str(arr.shape)}
                    )
            
            elif key == "quadratic":
                if arr.ndim != 4 or arr.shape[0] != num_agents or arr.shape[1] != num_items or arr.shape[2] != num_items:
                    raise DimensionMismatchError(
                        f"agent_data['quadratic'] has incorrect shape",
                        expected={"shape": f"({num_agents}, {num_items}, {num_items}, num_features)"},
                        actual={"shape": str(arr.shape)}
                    )
            
            # Check for invalid values
            if np.isnan(arr).any() or np.isinf(arr).any():
                invalid_info = {}
                if np.isnan(arr).any():
                    invalid_info[f"agent_data.{key}"] = f"{np.isnan(arr).sum()} NaN values"
                if np.isinf(arr).any():
                    invalid_info[f"agent_data.{key}"] = f"{np.isinf(arr).sum()} Inf values"
                
                raise DataError(
                    f"agent_data['{key}'] contains invalid values",
                    invalid_fields=invalid_info
                )
    
    @staticmethod
    def validate_item_data(item_data: Dict[str, np.ndarray], num_items: int) -> None:
        """
        Validate item-specific data arrays.
        
        Args:
            item_data: Dictionary containing item data arrays
            num_items: Expected number of items
            
        Raises:
            DimensionMismatchError: If dimensions don't match
            DataError: If data contains invalid values
        """
        if item_data is None:
            return
        
        for key, arr in item_data.items():
            if not isinstance(arr, np.ndarray):
                raise ValidationError(
                    f"item_data['{key}'] must be a numpy array, got {type(arr).__name__}"
                )
            
            # Check dimensions
            if key == "modular":
                if arr.ndim != 2 or arr.shape[0] != num_items:
                    raise DimensionMismatchError(
                        f"item_data['modular'] has incorrect shape",
                        expected={"shape": f"({num_items}, num_features)"},
                        actual={"shape": str(arr.shape)}
                    )
            
            elif key == "quadratic":
                if arr.ndim != 3 or arr.shape[0] != num_items or arr.shape[1] != num_items:
                    raise DimensionMismatchError(
                        f"item_data['quadratic'] has incorrect shape",
                        expected={"shape": f"({num_items}, {num_items}, num_features)"},
                        actual={"shape": str(arr.shape)}
                    )
            
            # Check for invalid values
            if np.isnan(arr).any() or np.isinf(arr).any():
                invalid_info = {}
                if np.isnan(arr).any():
                    invalid_info[f"item_data.{key}"] = f"{np.isnan(arr).sum()} NaN values"
                if np.isinf(arr).any():
                    invalid_info[f"item_data.{key}"] = f"{np.isinf(arr).sum()} Inf values"
                
                raise DataError(
                    f"item_data['{key}'] contains invalid values",
                    invalid_fields=invalid_info
                )
    
    @staticmethod
    def validate_errors(errors: np.ndarray, num_agents: int, num_items: int, num_simuls: int = 1) -> None:
        """
        Validate error array.
        
        Args:
            errors: Error array
            num_agents: Expected number of agents
            num_items: Expected number of items
            num_simuls: Number of simulations
            
        Raises:
            DimensionMismatchError: If dimensions don't match
            DataError: If errors contain invalid values
        """
        if errors is None:
            raise ValidationError("errors array is required but was None")
        
        # Check shape
        expected_shapes = [
            (num_agents, num_items),
            (num_simuls, num_agents, num_items)
        ]
        
        if errors.shape not in expected_shapes:
            raise DimensionMismatchError(
                "errors array has incorrect shape",
                expected={"shape": f"({num_agents}, {num_items}) or ({num_simuls}, {num_agents}, {num_items})"},
                actual={"shape": str(errors.shape)}
            )
        
        # Check for invalid values
        if np.isnan(errors).any() or np.isinf(errors).any():
            invalid_info = {}
            if np.isnan(errors).any():
                invalid_info["errors"] = f"{np.isnan(errors).sum()} NaN values"
            if np.isinf(errors).any():
                invalid_info["errors"] = f"{np.isinf(errors).sum()} Inf values"
            
            raise DataError(
                "errors array contains invalid values",
                invalid_fields=invalid_info
            )
    
    @staticmethod
    def validate_obs_bundle(obs_bundle: np.ndarray, num_agents: int, num_items: int) -> None:
        """
        Validate observed bundle array.
        
        Args:
            obs_bundle: Observed bundle array
            num_agents: Expected number of agents
            num_items: Expected number of items
            
        Raises:
            DimensionMismatchError: If dimensions don't match
            ValidationError: If bundles are invalid
        """
        if obs_bundle is None:
            return  # obs_bundle is optional
        
        if obs_bundle.shape != (num_agents, num_items):
            raise DimensionMismatchError(
                "obs_bundle has incorrect shape",
                expected={"shape": f"({num_agents}, {num_items})"},
                actual={"shape": str(obs_bundle.shape)}
            )
        
        # Check that values are binary or boolean
        if not np.all(np.isin(obs_bundle, [0, 1, True, False])):
            raise ValidationError(
                "obs_bundle must contain only binary (0/1) or boolean values",
                details={"unique_values": str(np.unique(obs_bundle))}
            )


class ConfigValidator:
    """
    Validates configuration values.
    
    This class provides validation for configuration parameters to ensure
    they are within valid ranges and compatible with each other.
    """
    
    @staticmethod
    def validate_dimensions(dimensions_cfg) -> None:
        """
        Validate dimension configuration.
        
        Args:
            dimensions_cfg: DimensionsConfig instance
            
        Raises:
            ConfigurationError: If dimensions are invalid
        """
        if dimensions_cfg.num_agents is not None and dimensions_cfg.num_agents <= 0:
            raise ConfigurationError(
                "num_agents must be positive",
                config_field="dimensions.num_agents",
                suggestion="Set num_agents to a positive integer (e.g., 100)"
            )
        
        if dimensions_cfg.num_items is not None and dimensions_cfg.num_items <= 0:
            raise ConfigurationError(
                "num_items must be positive",
                config_field="dimensions.num_items",
                suggestion="Set num_items to a positive integer (e.g., 20)"
            )
        
        if dimensions_cfg.num_features is not None and dimensions_cfg.num_features <= 0:
            raise ConfigurationError(
                "num_features must be positive",
                config_field="dimensions.num_features",
                suggestion="Set num_features to a positive integer (e.g., 5)"
            )
        
        if dimensions_cfg.num_simuls <= 0:
            raise ConfigurationError(
                "num_simuls must be positive",
                config_field="dimensions.num_simuls",
                suggestion="Set num_simuls to a positive integer (default is 1)"
            )
    
    @staticmethod
    def validate_row_generation(row_generation_cfg) -> None:
        """
        Validate row generation configuration.
        
        Args:
            row_generation_cfg: RowGenerationConfig instance
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        if row_generation_cfg.tolerance_optimality <= 0:
            raise ConfigurationError(
                "tolerance_optimality must be positive",
                config_field="row_generation.tolerance_optimality",
                suggestion="Set tolerance_optimality to a small positive value (e.g., 1e-6)"
            )
        
        if row_generation_cfg.max_iters <= 0:
            raise ConfigurationError(
                "max_iters must be positive",
                config_field="row_generation.max_iters",
                suggestion="Set max_iters to a positive integer (e.g., 100)"
            )
        
        if row_generation_cfg.min_iters < 0:
            raise ConfigurationError(
                "min_iters must be non-negative",
                config_field="row_generation.min_iters",
                suggestion="Set min_iters to 0 or a positive integer"
            )
    
    @staticmethod
    def validate_ellipsoid(ellipsoid_cfg) -> None:
        """
        Validate ellipsoid configuration.
        
        Args:
            ellipsoid_cfg: EllipsoidConfig instance
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        if ellipsoid_cfg.max_iterations <= 0:
            raise ConfigurationError(
                "max_iterations must be positive",
                config_field="ellipsoid.max_iterations",
                suggestion="Set max_iterations to a positive integer (e.g., 1000)"
            )
        
        if ellipsoid_cfg.tolerance <= 0:
            raise ConfigurationError(
                "tolerance must be positive",
                config_field="ellipsoid.tolerance",
                suggestion="Set tolerance to a small positive value (e.g., 1e-6)"
            )
        
        if ellipsoid_cfg.initial_radius <= 0:
            raise ConfigurationError(
                "initial_radius must be positive",
                config_field="ellipsoid.initial_radius",
                suggestion="Set initial_radius to a positive value (e.g., 1.0)"
            )
        
        if not 0 < ellipsoid_cfg.decay_factor < 1:
            raise ConfigurationError(
                "decay_factor must be between 0 and 1",
                config_field="ellipsoid.decay_factor",
                suggestion="Set decay_factor to a value like 0.95"
            )
        
        if ellipsoid_cfg.min_volume <= 0:
            raise ConfigurationError(
                "min_volume must be positive",
                config_field="ellipsoid.min_volume",
                suggestion="Set min_volume to a small positive value (e.g., 1e-12)"
            )

