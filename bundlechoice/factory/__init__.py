"""
Scenario factory builders used for tests, benchmarks, and experiments.

These helpers centralize the generation of synthetic data, observed bundles,
and configuration dictionaries for various BundleChoice subproblem settings.

Scenarios expose a builder-style API so callers can customize dimensions,
noise parameters, and seeds before materializing a prepared scenario that is
ready for estimation workflows.
"""

from __future__ import annotations

from . import utils
from .base import (
    FeatureSpec,
    PreparedScenario,
    ScenarioBuilder,
    SyntheticScenario,
)
from .library import ScenarioLibrary


__all__ = [
    "FeatureSpec",
    "PreparedScenario",
    "ScenarioBuilder",
    "SyntheticScenario",
    "ScenarioLibrary",
    "utils",
]

