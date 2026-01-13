"""Validation utilities for BundleChoice."""

import numpy as np
from typing import Dict, Any, List
from bundlechoice.errors import ValidationError, DimensionMismatchError, DataError


def validate_agent_data(agent_data: Dict[str, np.ndarray], num_agents: int, num_items: int) -> None:
    """Validate agent data dimensions."""
    if not agent_data:
        return
    
    for key, arr in agent_data.items():
        if not isinstance(arr, np.ndarray):
            raise DataError(f"agent_data['{key}'] must be a numpy array, got {type(arr).__name__}")
        
        if arr.shape[0] != num_agents:
            raise DimensionMismatchError(
                f"agent_data['{key}']: expected {num_agents} agents, got {arr.shape[0]}"
            )
        
        if key == 'modular' and arr.ndim >= 2 and arr.shape[1] != num_items:
            raise DimensionMismatchError(
                f"agent_data['modular']: expected {num_items} items, got {arr.shape[1]}"
            )


def validate_item_data(item_data: Dict[str, np.ndarray], num_items: int) -> None:
    """Validate item data dimensions."""
    if not item_data:
        return
    
    for key, arr in item_data.items():
        if not isinstance(arr, np.ndarray):
            raise DataError(f"item_data['{key}'] must be a numpy array, got {type(arr).__name__}")
        
        if arr.shape[0] != num_items:
            raise DimensionMismatchError(
                f"item_data['{key}']: expected {num_items} items, got {arr.shape[0]}"
            )


def validate_errors(errors: np.ndarray, num_agents: int, num_items: int, num_simulations: int, required: bool = True) -> None:
    """Validate error array dimensions."""
    if errors is None:
        if required:
            raise DataError("errors array is required")
        return
    
    if not isinstance(errors, np.ndarray):
        raise DataError(f"errors must be a numpy array, got {type(errors).__name__}")
    
    if errors.ndim == 2:
        expected = (num_agents, num_items)
        if errors.shape != expected:
            raise DimensionMismatchError(f"errors shape: expected {expected}, got {errors.shape}")
    elif errors.ndim == 3:
        expected = (num_simulations, num_agents, num_items)
        if errors.shape != expected:
            raise DimensionMismatchError(f"errors shape: expected {expected}, got {errors.shape}")
    else:
        raise DataError(f"errors must be 2D or 3D, got {errors.ndim}D")


def validate_obs_bundles(obs_bundles: np.ndarray, num_agents: int, num_items: int) -> None:
    """Validate observed bundles dimensions."""
    if obs_bundles is None:
        return
    
    if not isinstance(obs_bundles, np.ndarray):
        raise DataError(f"obs_bundle must be a numpy array, got {type(obs_bundles).__name__}")
    
    expected = (num_agents, num_items)
    if obs_bundles.shape != expected:
        raise DimensionMismatchError(f"obs_bundle shape: expected {expected}, got {obs_bundles.shape}")


def check_nan_inf(data: Dict[str, Any], prefix: str = "") -> List[str]:
    """Check for NaN or Inf values in nested data dictionaries."""
    problems = []
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            problems.extend(check_nan_inf(value, path))
        elif isinstance(value, np.ndarray):
            if np.isnan(value).any():
                problems.append(f"{path}: {np.isnan(value).sum()} NaN values")
            if np.isinf(value).any():
                problems.append(f"{path}: {np.isinf(value).sum()} Inf values")
    
    return problems


def validate_quadratic_features(item_data: Dict[str, np.ndarray]) -> None:
    """Validate quadratic features for supermodular problems."""
    if 'quadratic' not in item_data:
        return
    
    quad = item_data['quadratic']
    issues = []
    
    if (quad < 0).any():
        issues.append(f"{(quad < 0).sum()} negative values")
    
    num_features = quad.shape[2] if quad.ndim == 3 else 1
    for k in range(num_features):
        quad_k = quad[:, :, k] if quad.ndim == 3 else quad
        if not np.allclose(np.diag(quad_k), 0):
            issues.append(f"feature {k}: non-zero diagonal")
    
    if issues:
        raise ValidationError(f"Invalid quadratic features: {'; '.join(issues)}")


def count_features_from_data(input_data: Dict[str, Any]) -> int:
    """Count total features from data structure."""
    agent_data = input_data.get("agent_data", {})
    item_data = input_data.get("item_data", {})
    
    total = 0
    if "modular" in agent_data:
        arr = agent_data["modular"]
        total += arr.shape[2] if arr.ndim == 3 else 0
    if "modular" in item_data:
        arr = item_data["modular"]
        total += arr.shape[1] if arr.ndim == 2 else 0
    if "quadratic" in agent_data:
        arr = agent_data["quadratic"]
        total += arr.shape[3] if arr.ndim == 4 else 0
    if "quadratic" in item_data:
        arr = item_data["quadratic"]
        total += arr.shape[2] if arr.ndim == 3 else 0
    
    return total


def validate_feature_count(input_data: Dict[str, Any], expected_features: int) -> None:
    """Validate that feature count matches data structure."""
    actual = count_features_from_data(input_data)
    if actual != expected_features:
        raise ValidationError(
            f"Feature count mismatch: config has {expected_features}, data has {actual}"
        )


def validate_input_data_comprehensive(input_data: Dict[str, Any], dimensions_cfg, errors_required: bool = True) -> None:
    """Comprehensive validation of input data dimensions and quality."""
    validate_agent_data(input_data.get('agent_data'), dimensions_cfg.num_agents, dimensions_cfg.num_items)
    validate_item_data(input_data.get('item_data'), dimensions_cfg.num_items)
    validate_errors(input_data.get('errors'), dimensions_cfg.num_agents, dimensions_cfg.num_items, dimensions_cfg.num_simulations, required=errors_required)
    validate_obs_bundles(input_data.get('obs_bundle'), dimensions_cfg.num_agents, dimensions_cfg.num_items)
    
    problems = check_nan_inf(input_data)
    if problems:
        raise DataError(f"Data contains invalid values: {'; '.join(problems)}")
    
    if 'item_data' in input_data:
        validate_quadratic_features(input_data['item_data'])
