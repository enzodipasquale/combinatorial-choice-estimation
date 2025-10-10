"""
Custom exceptions for BundleChoice with helpful error messages.
"""

from typing import Optional, Dict, Any


class BundleChoiceError(Exception):
    """Base exception for all BundleChoice errors."""
    
    def __init__(self, message: str, suggestion: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.suggestion = suggestion
        self.context = context or {}
        
        # Simple, clear formatting - no fancy boxes
        parts = [f"\n{self.__class__.__name__}: {message}"]
        
        if context:
            parts.append("Context: " + ", ".join(f"{k}={v}" for k, v in context.items()))
        
        if suggestion:
            parts.append(f"Suggestion: {suggestion}")
        
        super().__init__("\n".join(parts) + "\n")


# Specific exception types for targeted error handling
class SetupError(BundleChoiceError):
    """Raised when BundleChoice setup is incomplete or incorrect."""
    pass


class DataError(BundleChoiceError):
    """Raised when data validation fails."""
    pass


class DimensionMismatchError(DataError):
    """Raised when data dimensions don't match configuration."""
    pass


class ValidationError(BundleChoiceError):
    """Raised when validation fails."""
    pass


class SolverError(BundleChoiceError):
    """Raised when solver encounters an error."""
    pass


class MPIError(BundleChoiceError):
    """Raised when MPI operations fail."""
    pass


class ConfigurationError(BundleChoiceError):
    """Raised when configuration is invalid."""
    pass
