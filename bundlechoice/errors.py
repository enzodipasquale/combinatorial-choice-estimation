"""Custom exceptions for BundleChoice framework."""


class BundleChoiceError(Exception):
    """Base exception for all BundleChoice errors."""


class SetupError(BundleChoiceError):
    """Raised when setup/initialization is incomplete or invalid."""
    
    def __init__(self, message: str, suggestion: str = None, missing: list = None):
        self.suggestion = suggestion
        self.missing = missing or []
        super().__init__(message)


class ValidationError(BundleChoiceError):
    """Raised when data or config validation fails."""


class DimensionMismatchError(ValidationError):
    """Raised when data dimensions don't match configuration."""


class DataError(ValidationError):
    """Raised when input data contains invalid values (NaN, Inf, etc.)."""
