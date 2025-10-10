"""
Custom exceptions for BundleChoice framework.

This module defines a hierarchy of exceptions that provide clear, actionable
error messages to help users debug setup and configuration issues.
"""


class BundleChoiceError(Exception):
    """Base exception for all BundleChoice errors."""
    pass


class SetupError(BundleChoiceError):
    """
    Raised when setup/initialization is incomplete or invalid.
    
    This error indicates that required components haven't been initialized
    in the correct order or are missing necessary configuration.
    
    Attributes:
        suggestion (str): Suggested action to fix the error
        missing (list): List of missing components
        context (dict): Additional context about the error
    """
    
    def __init__(self, message: str, suggestion: str = None, missing: list = None, context: dict = None):
        self.suggestion = suggestion
        self.missing = missing or []
        self.context = context or {}
        super().__init__(self._format_message(message))
    
    def _format_message(self, message):
        """Format error message with helpful visual indicators."""
        msg = f"❌ {message}"
        
        if self.missing:
            msg += f"\n\nMissing components: {', '.join(self.missing)}"
        
        if self.suggestion:
            msg += f"\n\n💡 Suggestion:\n{self.suggestion}"
        
        if self.context:
            msg += f"\n\nContext: {self.context}"
        
        return msg


class ValidationError(BundleChoiceError):
    """
    Raised when data or config validation fails.
    
    This error indicates that input data or configuration doesn't meet
    the required structure, dimensions, or value constraints.
    
    Attributes:
        details (dict): Detailed information about validation failures
        suggestions (list): List of suggested fixes
    """
    
    def __init__(self, message: str, details: dict = None, suggestions: list = None):
        self.details = details or {}
        self.suggestions = suggestions or []
        super().__init__(self._format_message(message))
    
    def _format_message(self, message):
        """Format validation error with detailed feedback."""
        msg = f"❌ {message}"
        
        if self.details:
            msg += "\n\nValidation failures:"
            for key, value in self.details.items():
                msg += f"\n  • {key}: {value}"
        
        if self.suggestions:
            msg += "\n\n💡 Suggestions:"
            for suggestion in self.suggestions:
                msg += f"\n  • {suggestion}"
        
        return msg


class DimensionMismatchError(ValidationError):
    """
    Raised when data dimensions don't match configuration.
    
    This specialized validation error occurs when array shapes or sizes
    don't align with the configured problem dimensions.
    """
    
    def __init__(self, message: str, expected: dict = None, actual: dict = None):
        self.expected = expected or {}
        self.actual = actual or {}
        
        details = {}
        if expected and actual:
            for key in expected:
                if key in actual and expected[key] != actual[key]:
                    details[key] = f"expected {expected[key]}, got {actual[key]}"
        
        super().__init__(message, details=details)


class DataError(ValidationError):
    """
    Raised when input data contains invalid values.
    
    This error occurs when data contains NaN, Inf, or other invalid values
    that would cause numerical issues during estimation.
    """
    
    def __init__(self, message: str, invalid_fields: dict = None):
        self.invalid_fields = invalid_fields or {}
        
        details = {}
        suggestions = []
        
        for field, issues in self.invalid_fields.items():
            details[field] = issues
            if 'NaN' in issues:
                suggestions.append(f"Replace NaN in '{field}' using np.nan_to_num() or mean imputation")
            if 'Inf' in issues:
                suggestions.append(f"Clip values in '{field}' using np.clip() to prevent overflow")
        
        super().__init__(message, details=details, suggestions=suggestions)


class SolverError(BundleChoiceError):
    """
    Raised when solver encounters issues during estimation.
    
    This error occurs during the optimization process, such as
    convergence failures or numerical instability.
    
    Attributes:
        solver_type (str): Type of solver that failed
        iteration (int): Iteration at which error occurred
        details (dict): Additional solver-specific details
    """
    
    def __init__(self, message: str, solver_type: str = None, iteration: int = None, details: dict = None):
        self.solver_type = solver_type
        self.iteration = iteration
        self.details = details or {}
        super().__init__(self._format_message(message))
    
    def _format_message(self, message):
        """Format solver error with diagnostic information."""
        msg = f"❌ {message}"
        
        if self.solver_type:
            msg += f"\n\nSolver: {self.solver_type}"
        
        if self.iteration is not None:
            msg += f"\nIteration: {self.iteration}"
        
        if self.details:
            msg += "\n\nDiagnostics:"
            for key, value in self.details.items():
                msg += f"\n  • {key}: {value}"
        
        return msg


class SubproblemError(SolverError):
    """
    Raised when subproblem solving fails.
    
    This error occurs when the optimization subproblem cannot be solved,
    often due to infeasibility or numerical issues.
    """
    pass


class ConfigurationError(BundleChoiceError):
    """
    Raised when configuration is invalid or inconsistent.
    
    This error occurs when configuration values are out of range,
    incompatible with each other, or missing required fields.
    """
    
    def __init__(self, message: str, config_field: str = None, suggestion: str = None):
        self.config_field = config_field
        self.suggestion = suggestion
        super().__init__(self._format_message(message))
    
    def _format_message(self, message):
        """Format configuration error."""
        msg = f"❌ {message}"
        
        if self.config_field:
            msg += f"\n\nConfiguration field: {self.config_field}"
        
        if self.suggestion:
            msg += f"\n\n💡 Suggestion:\n{self.suggestion}"
        
        return msg
