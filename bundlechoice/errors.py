"""Custom exceptions for BundleChoice framework."""

from typing import Optional, Dict, List, Any


class BundleChoiceError(Exception):
    """Base exception for all BundleChoice errors."""
    pass


class SetupError(BundleChoiceError):
    """Raised when setup/initialization is incomplete or invalid."""
    
    def __init__(self, message: str, suggestion: Optional[str] = None, 
                 missing: Optional[List[str]] = None, context: Optional[Dict[str, Any]] = None):
        self.suggestion = suggestion
        self.missing = missing or []
        self.context = context or {}
        
        msg = message
        if self.missing:
            msg += f"\n\nMissing components: {', '.join(self.missing)}"
        if self.suggestion:
            msg += f"\n\nSuggestion:\n{self.suggestion}"
        if self.context:
            msg += f"\n\nContext: {self.context}"
        super().__init__(msg)


class ValidationError(BundleChoiceError):
    """Raised when data or config validation fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, 
                 suggestions: Optional[List[str]] = None):
        self.details = details or {}
        self.suggestions = suggestions or []
        
        msg = message
        if self.details:
            msg += "\n\nValidation failures:"
            for key, value in self.details.items():
                msg += f"\n  - {key}: {value}"
        if self.suggestions:
            msg += "\n\nSuggestions:"
            for suggestion in self.suggestions:
                msg += f"\n  - {suggestion}"
        super().__init__(msg)


class DimensionMismatchError(ValidationError):
    """Raised when data dimensions don't match configuration."""
    
    def __init__(self, message: str, expected: Optional[Dict[str, Any]] = None,
                 actual: Optional[Dict[str, Any]] = None, suggestion: Optional[str] = None,
                 suggestions: Optional[List[str]] = None, context: Optional[Dict[str, Any]] = None):
        self.expected = expected or {}
        self.actual = actual or {}
        self.context = context or {}
        
        details = {}
        if expected and actual:
            for key in expected:
                if key in actual and expected[key] != actual[key]:
                    details[key] = f"expected {expected[key]}, got {actual[key]}"
        if self.context:
            details.update({f"context[{k}]": v for k, v in self.context.items()})
        
        suggestion_list = []
        if suggestion:
            suggestion_list.append(suggestion)
        if suggestions:
            suggestion_list.extend(suggestions)
        
        super().__init__(message, details=details, suggestions=suggestion_list)


class DataError(ValidationError):
    """Raised when input data contains invalid values (NaN, Inf, etc.)."""
    
    def __init__(self, message: str, invalid_fields: Optional[Dict[str, str]] = None,
                 suggestion: Optional[str] = None):
        self.invalid_fields = invalid_fields or {}
        
        details = {}
        suggestions = []
        
        if suggestion:
            suggestions.append(suggestion)
        
        for field, issues in self.invalid_fields.items():
            details[field] = issues
            if 'NaN' in issues:
                suggestions.append(f"Replace NaN in '{field}' using np.nan_to_num()")
            if 'Inf' in issues:
                suggestions.append(f"Clip values in '{field}' using np.clip()")
        
        super().__init__(message, details=details, suggestions=suggestions)


class SolverError(BundleChoiceError):
    """Raised when solver encounters issues during estimation."""
    
    def __init__(self, message: str, solver_type: Optional[str] = None, 
                 iteration: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        self.solver_type = solver_type
        self.iteration = iteration
        self.details = details or {}
        
        msg = message
        if self.solver_type:
            msg += f"\nSolver: {self.solver_type}"
        if self.iteration is not None:
            msg += f"\nIteration: {self.iteration}"
        if self.details:
            msg += "\nDiagnostics:"
            for key, value in self.details.items():
                msg += f"\n  - {key}: {value}"
        super().__init__(msg)


class SubproblemError(SolverError):
    """Raised when subproblem solving fails."""
    pass


class ConfigurationError(BundleChoiceError):
    """Raised when configuration is invalid or inconsistent."""
    
    def __init__(self, message: str, config_field: Optional[str] = None, 
                 suggestion: Optional[str] = None):
        self.config_field = config_field
        self.suggestion = suggestion
        
        msg = message
        if self.config_field:
            msg += f"\nConfiguration field: {self.config_field}"
        if self.suggestion:
            msg += f"\nSuggestion: {self.suggestion}"
        super().__init__(msg)
