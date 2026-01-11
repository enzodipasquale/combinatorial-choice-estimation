"""
Result objects for estimation methods.

Provides structured results with diagnostics for all estimation algorithms.
"""

import csv
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
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
    
    def export_csv(
        self,
        path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        append: bool = True,
    ) -> None:
        """
        Export results to CSV file with metadata and theta values.
        
        Args:
            path: CSV file path (will create directory if needed)
            metadata: Additional metadata columns (e.g., {"delta": 4, "winners_only": True})
            feature_names: Optional feature names for theta columns (uses indices if None)
            append: If True, append to existing file; if False, overwrite
        
        Example:
            result.export_csv("results/theta_hat.csv", 
                metadata={"delta": 4, "num_agents": 256},
                feature_names=["bidder_pop", "FE_0", ..., "travel_survey"])
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build row data
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "converged": self.converged,
            "num_iterations": self.num_iterations,
        }
        
        if self.final_objective is not None:
            row["final_objective"] = self.final_objective
        
        # Add timing stats
        if self.timing:
            for key, val in self.timing.items():
                row[key] = val
        
        # Add custom metadata
        if metadata:
            row.update(metadata)
        
        # Add theta values with optional naming
        for i, val in enumerate(self.theta_hat):
            name = feature_names[i] if feature_names and i < len(feature_names) else f"theta_{i}"
            row[f"theta_{name}" if feature_names else name] = val
        
        # Handle append logic with column consistency
        file_exists = path.exists()
        
        if append and file_exists:
            # Read existing to merge columns
            with open(path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
                existing_cols = list(reader.fieldnames) if reader.fieldnames else []
            
            # Merge columns (existing + new)
            all_cols = existing_cols.copy()
            for col in row.keys():
                if col not in all_cols:
                    all_cols.append(col)
            
            # Rewrite with consistent columns
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=all_cols)
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow({k: v for k, v in existing_row.items() if k})
                writer.writerow(row)
        else:
            # Create new or overwrite
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
                writer.writerow(row)
    
    def save_npy(self, path: Union[str, Path]) -> None:
        """Save theta_hat as numpy array."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.theta_hat)
