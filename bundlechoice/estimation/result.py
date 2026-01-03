"""
Result objects for estimation methods.

Provides structured results with diagnostics for all estimation algorithms.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import numpy as np
from numpy.typing import NDArray


@dataclass
class EstimationResult:
    """
    Container for estimation results with diagnostics.
    
    Attributes:
        theta_hat: Estimated parameter vector (shape: num_features,)
        converged: True if converged, False if hit max iterations
        num_iterations: Number of iterations performed
        final_objective: Final objective value (if applicable)
        timing: Timing statistics dictionary
        iteration_history: Optional iteration-by-iteration history (None by default)
        warnings: List of warning messages
        metadata: Additional method-specific metadata
    """
    theta_hat: NDArray[np.float64]
    converged: bool
    num_iterations: int
    final_objective: Optional[float] = None
    timing: Optional[Dict[str, float]] = None
    iteration_history: Optional[Dict[str, List]] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Return human-readable summary of results."""
        lines = ["=== Estimation Results ==="]
        lines.append(f"Converged: {self.converged}")
        lines.append(f"Iterations: {self.num_iterations}")
        if self.final_objective is not None:
            lines.append(f"Final objective: {self.final_objective:.6f}")
        if self.timing:
            lines.append(f"Total time: {self.timing.get('total_time', 0):.2f}s")
        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


