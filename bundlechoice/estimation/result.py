import csv
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np

@dataclass
class EstimationResult:
    theta_hat: np.ndarray
    converged: bool
    num_iterations: int
    final_objective: None
    n_constraints: int = None
    final_reduced_cost: float = None
    timing: dict = field(default_factory=dict)
    iteration_history: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict) 

    def summary(self):
        lines = ['=== Estimation Results ===']
        lines.append(f'Converged: {self.converged}')
        lines.append(f'Iterations: {self.num_iterations}')
        if self.final_objective is not None:
            lines.append(f'Final objective: {self.final_objective:.6f}')
        if self.timing:
            lines.append(f"Total time: {self.timing.get('total_time', 0):.2f}s")
        if self.warnings:
            lines.append(f'Warnings: {len(self.warnings)}')
            for w in self.warnings:
                lines.append(f'  - {w}')
        return '\n'.join(lines)

    def export_csv(self, path, metadata=None, feature_names=None, append=True):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {'timestamp': datetime.now().isoformat(timespec='seconds'), 'converged': self.converged, 'num_iterations': self.num_iterations}
        if self.final_objective is not None:
            row['final_objective'] = self.final_objective
        if self.timing:
            for key, val in self.timing.items():
                row[key] = val
        if metadata:
            row.update(metadata)
        for i, val in enumerate(self.theta_hat):
            name = feature_names[i] if feature_names and i < len(feature_names) else f'theta_{i}'
            row[f'theta_{name}' if feature_names else name] = val
        file_exists = path.exists()
        if append and file_exists:
            with open(path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
                existing_cols = list(reader.fieldnames) if reader.fieldnames else []
            all_cols = existing_cols.copy()
            for col in row.keys():
                if col not in all_cols:
                    all_cols.append(col)
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=all_cols)
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow({k: v for k, v in existing_row.items() if k})
                writer.writerow(row)
        else:
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
                writer.writerow(row)

    def save_npy(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.theta_hat)