"""
Custom exceptions for BundleChoice framework.

Hierarchy of exceptions with helpful error messages for debugging.
"""

from typing import Optional, Dict, List, Any


# ============================================================================
# Base Exception
# ============================================================================

class BundleChoiceError(Exception):
    """Base exception for all BundleChoice errors."""
    pass


# ============================================================================
# Setup & Validation Errors
# ============================================================================

class SetupError(BundleChoiceError):
    """Raised when setup/initialization is incomplete or invalid."""
    
    def __init__(self, message: str, suggestion: Optional[str] = None, 
                 missing: Optional[List[str]] = None, context: Optional[Dict[str, Any]] = None):
        self.suggestion = suggestion
        self.missing = missing or []
        self.context = context or {}
        super().__init__(self._format_message(message))
    
    def _format_message(self, message: str) -> str:
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
    """Raised when data or config validation fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, 
                 suggestions: Optional[List[str]] = None):
        self.details = details or {}
        self.suggestions = suggestions or []
        super().__init__(self._format_message(message))
    
    def _format_message(self, message: str) -> str:
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
    """Raised when data dimensions don't match configuration."""
    
    def __init__(self, message: str, expected: Optional[Dict[str, Any]] = None, 
                 actual: Optional[Dict[str, Any]] = None):
        self.expected = expected or {}
        self.actual = actual or {}
        
        details = {}
        if expected and actual:
            for key in expected:
                if key in actual and expected[key] != actual[key]:
                    details[key] = f"expected {expected[key]}, got {actual[key]}"
        
        super().__init__(message, details=details)


class DataError(ValidationError):
    """Raised when input data contains invalid values (NaN, Inf, etc.)."""
    
    def __init__(self, message: str, invalid_fields: Optional[Dict[str, str]] = None):
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


# ============================================================================
# Solver Errors
# ============================================================================

class SolverError(BundleChoiceError):
    """Raised when solver encounters issues during estimation."""
    
    def __init__(self, message: str, solver_type: Optional[str] = None, 
                 iteration: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        self.solver_type = solver_type
        self.iteration = iteration
        self.details = details or {}
        super().__init__(self._format_message(message))
    
    def _format_message(self, message: str) -> str:
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
    """Raised when subproblem solving fails (infeasibility, numerical issues)."""
    pass

# ============================================================================
# Configuration Errors
# ============================================================================

class ConfigurationError(BundleChoiceError):
    """Raised when configuration is invalid or inconsistent."""
    
    def __init__(self, message: str, config_field: Optional[str] = None, 
                 suggestion: Optional[str] = None):
        self.config_field = config_field
        self.suggestion = suggestion
        super().__init__(self._format_message(message))
    
    def _format_message(self, message: str) -> str:
        """Format configuration error."""
        msg = f"❌ {message}"
        
        if self.config_field:
            msg += f"\n\nConfiguration field: {self.config_field}"
        
        if self.suggestion:
            msg += f"\n\n💡 Suggestion:\n{self.suggestion}"
        
        return msg
