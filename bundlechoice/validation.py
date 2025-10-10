"""
Validation utilities for BundleChoice.
"""

import numpy as np
from typing import Dict, Any, List
from bundlechoice.errors import ValidationError, DimensionMismatchError, DataError, SetupError


# ============================================================================
# Setup Validation
# ============================================================================

WORKFLOW_REQUIREMENTS = {
    'solve_row_generation': ['config', 'data', 'features', 'subproblems', 'obs_bundles'],
    'solve_ellipsoid': ['config', 'data', 'features', 'subproblems', 'obs_bundles'],
    'solve_inequalities': ['config', 'data', 'features', 'obs_bundles'],
    'generate_observations': ['config', 'data', 'features', 'subproblems'],
    'compute_features': ['config', 'data', 'features'],
    'solve_subproblems': ['config', 'data', 'features', 'subproblems'],
}


def check_component(bc, component: str) -> bool:
    """Check if a specific component is initialized."""
    checks = {
        'config': lambda: bc.config is not None,
        'data': lambda: bc.data_manager is not None and bc.data_manager.local_data is not None,
        'features': lambda: bc.feature_manager is not None and bc.feature_manager._features_oracle is not None,
        'subproblems': lambda: bc.subproblem_manager is not None and bc.subproblem_manager.demand_oracle is not None,
        'obs_bundles': lambda: (bc.data_manager is not None and bc.data_manager.local_data is not None 
                                and bc.data_manager.local_data.get('obs_bundles') is not None),
    }
    return checks.get(component, lambda: False)()


def get_completed_steps(bc) -> List[str]:
    """Get list of completed setup steps."""
    return [comp for comp in ['config', 'data', 'features', 'subproblems', 'obs_bundles'] 
            if check_component(bc, comp)]


def validate_workflow(bc, operation: str) -> None:
    """Validate that required components are initialized for an operation."""
    if operation not in WORKFLOW_REQUIREMENTS:
        raise ValueError(f"Unknown operation: {operation}")
    
    required = WORKFLOW_REQUIREMENTS[operation]
    completed = get_completed_steps(bc)
    missing = [step for step in required if step not in completed]
    
    if missing:
        # Build helpful step-by-step instructions
        steps = {
            'config': "bc.load_config(config_dict)",
            'data': "bc.data.load_and_scatter(input_data)",
            'features': "bc.features.build_from_data() or bc.features.set_oracle(fn)",
            'subproblems': "bc.subproblems.load()",
            'obs_bundles': "Add 'obs_bundle' to input_data or bc.subproblems.init_and_solve(theta_true)"
        }
        
        instructions = "\n".join(
            f"  {'✗' if step in missing else '✓'} {steps.get(step, step)}"
            for step in required
        )
        
        raise SetupError(
            f"Cannot execute '{operation}' - missing: {', '.join(missing)}",
            suggestion=f"Complete these steps:\n{instructions}",
            context={'missing': missing, 'completed': completed}
        )


# ============================================================================
# Dimension Validation
# ============================================================================

def validate_agent_data(agent_data: Dict[str, np.ndarray], num_agents: int, num_items: int) -> None:
    """Validate agent data dimensions."""
    if not agent_data:
        return
    
    for key, arr in agent_data.items():
        if not isinstance(arr, np.ndarray):
            raise DataError(
                f"agent_data['{key}'] must be a numpy array, got {type(arr).__name__}",
                suggestion=f"Convert to numpy: agent_data['{key}'] = np.array(...)"
            )
        
        if arr.shape[0] != num_agents:
            raise DimensionMismatchError(
                f"agent_data['{key}'] has wrong number of agents: expected {num_agents}, got {arr.shape[0]}",
                suggestion="Ensure your data arrays match config dimensions",
                context={'expected_agents': num_agents, 'actual_agents': arr.shape[0]}
            )
        
        # Check modular features
        if key == 'modular' and arr.ndim >= 2 and arr.shape[1] != num_items:
            raise DimensionMismatchError(
                f"agent_data['modular'] has wrong number of items: expected {num_items}, got {arr.shape[1]}",
                suggestion="Modular features should have shape (num_agents, num_items, num_features)",
                context={'expected_items': num_items, 'actual_items': arr.shape[1]}
            )


def validate_item_data(item_data: Dict[str, np.ndarray], num_items: int) -> None:
    """Validate item data dimensions."""
    if not item_data:
        return
    
    for key, arr in item_data.items():
        if not isinstance(arr, np.ndarray):
            raise DataError(
                f"item_data['{key}'] must be a numpy array, got {type(arr).__name__}",
                suggestion=f"Convert to numpy: item_data['{key}'] = np.array(...)"
            )
        
        if arr.shape[0] != num_items:
            raise DimensionMismatchError(
                f"item_data['{key}'] has wrong number of items: expected {num_items}, got {arr.shape[0]}",
                suggestion="Ensure item_data arrays have shape (num_items, ...)",
                context={'expected': num_items, 'actual': arr.shape[0]}
            )


def validate_errors(errors: np.ndarray, num_agents: int, num_items: int, num_simuls: int) -> None:
    """Validate error array dimensions."""
    if errors is None:
        raise DataError(
            "errors array is required",
            suggestion="Add errors to input_data: input_data['errors'] = np.random.normal(0, sigma, (num_agents, num_items))"
        )
    
    if not isinstance(errors, np.ndarray):
        raise DataError(
            f"errors must be a numpy array, got {type(errors).__name__}",
            suggestion="Convert to numpy: errors = np.array(...)"
        )
    
    # Check dimensions
    if errors.ndim == 2:
        expected = (num_agents, num_items)
        if errors.shape != expected:
            raise DimensionMismatchError(
                f"errors shape mismatch: expected {expected}, got {errors.shape}",
                suggestion="errors should have shape (num_agents, num_items)",
                context={'expected_shape': expected, 'actual_shape': errors.shape}
            )
    elif errors.ndim == 3:
        expected = (num_simuls, num_agents, num_items)
        if errors.shape != expected:
            raise DimensionMismatchError(
                f"errors shape mismatch: expected {expected}, got {errors.shape}",
                suggestion="errors should have shape (num_simuls, num_agents, num_items)",
                context={'expected_shape': expected, 'actual_shape': errors.shape}
            )
    else:
        raise DataError(
            f"errors must be 2D or 3D, got {errors.ndim}D with shape {errors.shape}",
            suggestion="errors should have shape (num_agents, num_items) or (num_simuls, num_agents, num_items)"
        )


def validate_obs_bundles(obs_bundles: np.ndarray, num_agents: int, num_items: int) -> None:
    """Validate observed bundles dimensions."""
    if obs_bundles is None:
        return
    
    if not isinstance(obs_bundles, np.ndarray):
        raise DataError(
            f"obs_bundle must be a numpy array, got {type(obs_bundles).__name__}",
            suggestion="Convert to numpy: obs_bundle = np.array(...)"
        )
    
    expected = (num_agents, num_items)
    if obs_bundles.shape != expected:
        raise DimensionMismatchError(
            f"obs_bundle shape mismatch: expected {expected}, got {obs_bundles.shape}",
            suggestion="obs_bundle should have shape (num_agents, num_items)",
            context={'expected_shape': expected, 'actual_shape': obs_bundles.shape}
        )


# ============================================================================
# Data Quality Validation
# ============================================================================

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
    
    # Check for negative values
    if (quad < 0).any():
        issues.append(f"{(quad < 0).sum()} negative values (should be non-negative for supermodularity)")
    
    # Check diagonal is zero
    num_features = quad.shape[2] if quad.ndim == 3 else 1
    for k in range(num_features):
        quad_k = quad[:, :, k] if quad.ndim == 3 else quad
        if not np.allclose(np.diag(quad_k), 0):
            issues.append(f"Feature {k}: non-zero diagonal (should be zero, no self-interaction)")
    
    if issues:
        raise ValidationError(
            "Invalid quadratic features: " + "; ".join(issues),
            suggestions=["Quadratic features should be non-negative with zero diagonal"]
        )


def count_features_from_data(input_data: Dict[str, Any]) -> int:
    """
    Count total features from data structure.
    
    Sums features from: modular_agent + modular_item + quadratic_agent + quadratic_item
    
    Args:
        input_data: Input data dictionary with agent_data and/or item_data
        
    Returns:
        Total number of features in the data structure
    """
    agent_data = input_data.get("agent_data", {})
    item_data = input_data.get("item_data", {})
    
    total = 0
    
    # Modular agent features
    if "modular" in agent_data:
        arr = agent_data["modular"]
        total += arr.shape[2] if arr.ndim == 3 else 0
    
    # Modular item features
    if "modular" in item_data:
        arr = item_data["modular"]
        total += arr.shape[1] if arr.ndim == 2 else 0
    
    # Quadratic agent features
    if "quadratic" in agent_data:
        arr = agent_data["quadratic"]
        total += arr.shape[3] if arr.ndim == 4 else 0
    
    # Quadratic item features
    if "quadratic" in item_data:
        arr = item_data["quadratic"]
        total += arr.shape[2] if arr.ndim == 3 else 0
    
    return total


def validate_feature_count(input_data: Dict[str, Any], expected_features: int) -> None:
    """
    Validate that feature count matches data structure.
    
    Args:
        input_data: Input data dictionary
        expected_features: Expected number of features from config
        
    Raises:
        ValidationError: If feature count doesn't match
    """
    actual = count_features_from_data(input_data)
    if actual != expected_features:
        raise ValidationError(
            f"Feature count mismatch: config has {expected_features} features but data has {actual}",
            suggestions=["Ensure num_features in config matches your data structure"],
            details={'expected': expected_features, 'actual': actual}
        )


# ============================================================================
# Comprehensive Validation
# ============================================================================

def validate_input_data_comprehensive(input_data: Dict[str, Any], dimensions_cfg) -> None:
    """
    Comprehensive validation of input data.
    
    Validates dimensions and data quality in one pass.
    """
    # Dimension validation
    validate_agent_data(input_data.get('agent_data'), dimensions_cfg.num_agents, dimensions_cfg.num_items)
    validate_item_data(input_data.get('item_data'), dimensions_cfg.num_items)
    validate_errors(input_data.get('errors'), dimensions_cfg.num_agents, dimensions_cfg.num_items, dimensions_cfg.num_simuls)
    validate_obs_bundles(input_data.get('obs_bundle'), dimensions_cfg.num_agents, dimensions_cfg.num_items)
    
    # Data quality validation
    problems = check_nan_inf(input_data)
    if problems:
        raise DataError(
            "Data contains invalid values: " + "; ".join(problems),
            suggestion="Check for NaN/Inf values. Common causes: division by zero, log of negative numbers, missing data"
        )
    
    # Quadratic feature validation
    if 'item_data' in input_data:
        validate_quadratic_features(input_data['item_data'])


# ============================================================================
# Legacy compatibility (for tests and existing code)
# ============================================================================

class SetupValidator:
    """Legacy wrapper for setup validation."""
    WORKFLOW_REQUIREMENTS = WORKFLOW_REQUIREMENTS
    
    @staticmethod
    def check_component(bc, component: str) -> bool:
        return check_component(bc, component)
    
    @staticmethod
    def get_completed_steps(bc) -> List[str]:
        return get_completed_steps(bc)
    
    @staticmethod
    def validate_for_operation(bc, operation: str) -> None:
        validate_workflow(bc, operation)
    
    @staticmethod
    def validate_all(bc) -> Dict[str, bool]:
        return {comp: check_component(bc, comp) 
                for comp in ['config', 'data', 'features', 'subproblems', 'obs_bundles']}


class DimensionValidator:
    """Legacy wrapper for dimension validation."""
    
    @staticmethod
    def validate_agent_data(agent_data, num_agents, num_items):
        validate_agent_data(agent_data, num_agents, num_items)
    
    @staticmethod
    def validate_item_data(item_data, num_items):
        validate_item_data(item_data, num_items)
    
    @staticmethod
    def validate_errors(errors, num_agents, num_items, num_simuls):
        validate_errors(errors, num_agents, num_items, num_simuls)
    
    @staticmethod
    def validate_obs_bundles(obs_bundles, num_agents, num_items):
        validate_obs_bundles(obs_bundles, num_agents, num_items)
    
    @staticmethod
    def validate_all_data(input_data, num_agents, num_items, num_simuls):
        from bundlechoice.config import DimensionsConfig
        dims = DimensionsConfig(num_agents=num_agents, num_items=num_items, 
                                num_features=1, num_simuls=num_simuls)
        validate_input_data_comprehensive(input_data, dims)


class DataQualityValidator:
    """Legacy wrapper for data quality validation."""
    
    @staticmethod
    def check_for_invalid_values(data, context="data"):
        return check_nan_inf(data, context)
    
    @staticmethod
    def validate_quadratic_features(item_data):
        validate_quadratic_features(item_data)
    
    @staticmethod
    def validate_data_quality(input_data):
        problems = check_nan_inf(input_data)
        if problems:
            raise DataError(
                "Data contains invalid values: " + "; ".join(problems),
                suggestion="Check for NaN/Inf. Use np.nan_to_num() or imputation to fix."
            )
        if 'item_data' in input_data:
            validate_quadratic_features(input_data['item_data'])
