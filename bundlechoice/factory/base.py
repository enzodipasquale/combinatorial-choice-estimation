"""Base classes for scenario factory builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol

from mpi4py import MPI

from bundlechoice.core import BundleChoice


class FeatureInitializer(Protocol):
    """Callable that sets up features on a ``BundleChoice`` instance."""

    def __call__(self, bc: BundleChoice) -> None: ...


@dataclass(frozen=True)
class FeatureSpec:
    """
    Encapsulates how features should be initialized for a scenario.

    ``mode`` is either ``"oracle"`` (call ``set_oracle``) or ``"build"``
    (call ``build_from_data``).  ``initializer`` applies the correct setup to
    a ``BundleChoice`` instance.
    """

    mode: str
    initializer: FeatureInitializer

    @staticmethod
    def oracle(oracle_fn: Callable) -> "FeatureSpec":
        def _init(bc: BundleChoice) -> None:
            bc.features.set_oracle(oracle_fn)

        return FeatureSpec(mode="oracle", initializer=_init)

    @staticmethod
    def build() -> "FeatureSpec":
        def _init(bc: BundleChoice) -> None:
            bc.features.build_from_data()

        return FeatureSpec(mode="build", initializer=_init)


@dataclass
class PreparedScenario:
    """
    Output of ``SyntheticScenario.prepare``.

    ``config`` and ``feature_spec`` can be applied to any ``BundleChoice``
    instance.  ``generation_data`` is the payload used to simulate observed
    bundles, while ``estimation_data`` includes observed bundles and fresh
    estimation errors.  These dictionaries are only populated on rank 0.
    """

    name: str
    config: Dict[str, Any]
    feature_spec: FeatureSpec
    theta_star: Optional[Any]
    generation_data: Optional[Dict[str, Any]]
    estimation_data: Optional[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def apply(
        self,
        bc: BundleChoice,
        *,
        comm: MPI.Comm = MPI.COMM_WORLD,
        stage: str = "estimation",
    ) -> None:
        """Configure ``bc`` with the scenario payload."""

        data = {
            "generation": self.generation_data,
            "estimation": self.estimation_data,
        }.get(stage)

        if data is None and comm.Get_rank() == 0:
            raise ValueError(
                f"No data available on root for stage '{stage}' in scenario '{self.name}'."
            )

        bc.load_config(self.config)
        bc.data.load_and_scatter(data if comm.Get_rank() == 0 else None)
        self.feature_spec.initializer(bc)

        if stage == "estimation":
            bc.subproblems.load()


class ScenarioBuilder(Protocol):
    """Protocol shared by concrete scenario builders."""

    def with_seed(self, seed: int) -> "ScenarioBuilder": ...

    def with_timeout(self, seconds: Optional[int]) -> "ScenarioBuilder": ...

    def build(self) -> "SyntheticScenario": ...


@dataclass
class SyntheticScenario:
    """Encapsulates shared preparation logic across scenario variants."""

    name: str
    config_factory: Callable[[], Dict[str, Any]]
    feature_spec: FeatureSpec
    payload_factory: Callable[
        [BundleChoice, MPI.Comm, FeatureSpec, Optional[int], Optional[int], Any],
        Dict[str, Dict[str, Any]],
    ]
    theta_factory: Callable[[], Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def prepare(
        self,
        *,
        comm: MPI.Comm = MPI.COMM_WORLD,
        timeout_seconds: Optional[int] = None,
        seed: Optional[int] = None,
        theta: Optional[Any] = None,
    ) -> PreparedScenario:
        """
        Materialize the scenario by generating both the simulation and
        estimation datasets.  Returns a ``PreparedScenario``.
        
        Args:
            comm: MPI communicator
            timeout_seconds: Timeout for operations
            seed: Random seed for data generation
            theta: Optional theta vector to use for bundle generation. 
                   If None, uses default theta_star (all ones).
        """

        config = self.config_factory()
        theta_star = theta if theta is not None else self.theta_factory()

        bc = BundleChoice()
        bc.load_config(config)

        payload = self.payload_factory(
            bc,
            comm,
            self.feature_spec,
            timeout_seconds,
            seed,
            theta_star,
        )

        generation_data = payload.get("generation")
        estimation_data = payload.get("estimation")

        return PreparedScenario(
            name=self.name,
            config=config,
            feature_spec=self.feature_spec,
            theta_star=theta_star,
            generation_data=generation_data,
            estimation_data=estimation_data,
            metadata=self.metadata,
        )


__all__ = [
    "FeatureInitializer",
    "FeatureSpec",
    "PreparedScenario",
    "ScenarioBuilder",
    "SyntheticScenario",
]


