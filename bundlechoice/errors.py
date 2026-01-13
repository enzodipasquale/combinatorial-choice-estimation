"""Custom exceptions for BundleChoice framework."""


class BundleChoiceError(Exception):
    """Base exception for all BundleChoice errors."""


class SetupError(BundleChoiceError):
    """Raised when setup/initialization is incomplete or invalid."""


class ValidationError(BundleChoiceError):
    """Raised when data or config validation fails."""


class DimensionMismatchError(ValidationError):
    """Raised when data dimensions don't match configuration."""


class DataError(ValidationError):
    """Raised when input data contains invalid values (NaN, Inf, etc.)."""
