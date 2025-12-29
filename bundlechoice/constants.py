"""
Constants used throughout the BundleChoice framework.

Centralizes magic numbers and default values for consistency and maintainability.
"""

# ============================================================================
# Numerical Tolerances
# ============================================================================

# Default optimality tolerance for row generation
DEFAULT_TOLERANCE_OPTIMALITY = 1e-6

# Default tolerance for checking constraint violations
DEFAULT_CONSTRAINT_TOLERANCE = 1e-5

# Default tolerance for checking if constraint is binding
DEFAULT_BINDING_TOLERANCE = 1e-6

# Default tolerance for checking slack constraints
DEFAULT_SLACK_TOLERANCE = 1e-6

# Default solver precision
DEFAULT_SOLVER_PRECISION = 1e-6

# Default ellipsoid tolerance
DEFAULT_ELLIPSOID_TOLERANCE = 1e-6

# Default minimum volume for ellipsoid
DEFAULT_MIN_VOLUME = 1e-12

# ============================================================================
# Default Bounds
# ============================================================================

# Default upper bound for theta parameters
DEFAULT_THETA_UPPER_BOUND = 1000.0

# Default lower bound for theta parameters (non-negativity)
DEFAULT_THETA_LOWER_BOUND = 0.0

# Default upper bound for inequalities method
DEFAULT_INEQUALITIES_UPPER_BOUND = 100.0

# ============================================================================
# Default Iteration Limits
# ============================================================================

# Default maximum iterations (infinity)
DEFAULT_MAX_ITERATIONS = float('inf')

# Default minimum iterations
DEFAULT_MIN_ITERATIONS = 0

# Default ellipsoid max iterations
DEFAULT_ELLIPSOID_MAX_ITERATIONS = 1000

# ============================================================================
# Default Decay Factors
# ============================================================================

# Default row generation tolerance decay
DEFAULT_ROW_GENERATION_DECAY = 0.0

# Default ellipsoid decay factor
DEFAULT_ELLIPSOID_DECAY_FACTOR = 0.95

# ============================================================================
# Default Radii
# ============================================================================

# Default initial radius for ellipsoid method
DEFAULT_ELLIPSOID_INITIAL_RADIUS = 1.0

