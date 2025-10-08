#!/usr/bin/env python3
"""
Prototype implementation of Gurobi-style design patterns for BundleChoice.
This demonstrates how the user experience could be improved.
"""

import numpy as np
from mpi4py import MPI
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field

# Import existing BundleChoice for comparison
from bundlechoice.core import BundleChoice

@dataclass
class Dimensions:
    """Simple dimensions container."""
    agents: int
    items: int
    features: int
    simuls: int = 1

@dataclass
class Data:
    """Simple data container."""
    agent_features: np.ndarray
    obs_bundles: np.ndarray
    errors: np.ndarray

@dataclass
class Features:
    """Simple features container."""
    oracle: Callable

@dataclass
class Subproblem:
    """Simple subproblem container."""
    name: str
    settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Estimator:
    """Simple estimator container."""
    name: str
    settings: Dict[str, Any] = field(default_factory=dict)

class BundleChoiceModel:
    """
    Prototype Gurobi-style model for BundleChoice.
    Demonstrates step-by-step workflow.
    """
    
    def __init__(self):
        self.dimensions: Optional[Dimensions] = None
        self.data: Optional[Data] = None
        self.features: Optional[Features] = None
        self.subproblem: Optional[Subproblem] = None
        self.estimator: Optional[Estimator] = None
        self._bc: Optional[BundleChoice] = None
        
    def add_dimensions(self, agents: int, items: int, features: int, simuls: int = 1) -> 'BundleChoiceModel':
        """Add dimensions to the model."""
        self.dimensions = Dimensions(agents=agents, items=items, features=features, simuls=simuls)
        print(f"✅ Added dimensions: {agents} agents, {items} items, {features} features")
        return self
        
    def add_data(self, agent_features: np.ndarray, obs_bundles: np.ndarray, errors: np.ndarray) -> 'BundleChoiceModel':
        """Add data to the model."""
        self.data = Data(agent_features=agent_features, obs_bundles=obs_bundles, errors=errors)
        print(f"✅ Added data: {agent_features.shape}, {obs_bundles.shape}, {errors.shape}")
        return self
        
    def add_features(self, oracle: Callable) -> 'BundleChoiceModel':
        """Add feature computation to the model."""
        self.features = Features(oracle=oracle)
        print(f"✅ Added feature oracle: {oracle.__name__}")
        return self
        
    def add_subproblem(self, name: str, **settings) -> 'BundleChoiceModel':
        """Add subproblem to the model."""
        self.subproblem = Subproblem(name=name, settings=settings)
        print(f"✅ Added subproblem: {name}")
        return self
        
    def add_estimation_method(self, name: str, **settings) -> 'BundleChoiceModel':
        """Add estimation method to the model."""
        self.estimator = Estimator(name=name, settings=settings)
        print(f"✅ Added estimation method: {name}")
        return self
        
    def validate(self) -> None:
        """Validate that all required components are present."""
        missing = []
        if self.dimensions is None:
            missing.append("dimensions")
        if self.data is None:
            missing.append("data")
        if self.features is None:
            missing.append("features")
        if self.subproblem is None:
            missing.append("subproblem")
        if self.estimator is None:
            missing.append("estimation method")
            
        if missing:
            raise ValueError(f"Missing required components: {', '.join(missing)}")
        print("✅ Model validation passed")
        
    def _build_bundlechoice(self) -> BundleChoice:
        """Build the underlying BundleChoice instance."""
        if self._bc is not None:
            return self._bc
            
        # Create config
        cfg = {
            "dimensions": {
                "num_agents": self.dimensions.agents,
                "num_items": self.dimensions.items,
                "num_features": self.dimensions.features,
                "num_simuls": self.dimensions.simuls,
            },
            "subproblem": {
                "name": self.subproblem.name,
                **self.subproblem.settings
            },
            self.estimator.name.lower(): self.estimator.settings
        }
        
        # Create input data
        input_data = {
            "agent_data": {"features": self.data.agent_features},
            "obs_bundle": self.data.obs_bundles,
            "errors": self.data.errors
        }
        
        # Initialize BundleChoice
        self._bc = BundleChoice()
        self._bc.load_config(cfg)
        self._bc.data.load_and_scatter(input_data)
        self._bc.features.set_oracle(self.features.oracle)
        self._bc.subproblems.load()
        
        return self._bc
        
    def solve(self) -> np.ndarray:
        """Solve the model and return estimated parameters."""
        self.validate()
        bc = self._build_bundlechoice()
        
        # Call appropriate solver
        if self.estimator.name.lower() == 'ellipsoid':
            return bc.ellipsoid.solve()
        elif self.estimator.name.lower() == 'row_generation':
            return bc.row_generation.solve()
        else:
            raise ValueError(f"Unknown estimation method: {self.estimator.name}")

class BundleChoiceBuilder:
    """
    Prototype builder pattern for BundleChoice.
    Demonstrates fluent interface with validation.
    """
    
    def __init__(self):
        self._dimensions = None
        self._data = None
        self._features = None
        self._subproblem = None
        self._estimator = None
        
    def dimensions(self, agents: int, items: int, features: int, simuls: int = 1) -> 'BundleChoiceBuilder':
        """Set dimensions."""
        self._dimensions = Dimensions(agents=agents, items=items, features=features, simuls=simuls)
        return self
        
    def data(self, agent_features: np.ndarray, obs_bundles: np.ndarray, errors: np.ndarray) -> 'BundleChoiceBuilder':
        """Set data."""
        self._data = Data(agent_features=agent_features, obs_bundles=obs_bundles, errors=errors)
        return self
        
    def features(self, oracle: Callable) -> 'BundleChoiceBuilder':
        """Set features."""
        self._features = Features(oracle=oracle)
        return self
        
    def subproblem(self, name: str, **settings) -> 'BundleChoiceBuilder':
        """Set subproblem."""
        self._subproblem = Subproblem(name=name, settings=settings)
        return self
        
    def estimation(self, name: str, **settings) -> 'BundleChoiceBuilder':
        """Set estimation method."""
        self._estimator = Estimator(name=name, settings=settings)
        return self
        
    def build(self) -> BundleChoiceModel:
        """Build the model."""
        if not all([self._dimensions, self._data, self._features, self._subproblem, self._estimator]):
            missing = []
            if not self._dimensions: missing.append("dimensions")
            if not self._data: missing.append("data")
            if not self._features: missing.append("features")
            if not self._subproblem: missing.append("subproblem")
            if not self._estimator: missing.append("estimation")
            raise ValueError(f"Missing required components: {', '.join(missing)}")
            
        model = BundleChoiceModel()
        model.dimensions = self._dimensions
        model.data = self._data
        model.features = self._features
        model.subproblem = self._subproblem
        model.estimator = self._estimator
        
        return model

def simple_features_oracle(i_id, B_j, data):
    """Simple feature oracle for testing."""
    agent_features = data["agent_data"]["features"][i_id]
    bundle_sum = np.sum(B_j, axis=0)
    features = agent_features * bundle_sum
    return features

def test_prototype_workflows():
    """Test the prototype Gurobi-style workflows."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n" + "="*60)
        print("PROTOTYPE GUROBI-STYLE WORKFLOWS")
        print("="*60)
        
        # Generate test data
        num_agents, num_items, num_features = 20, 5, 2
        agent_features = np.random.normal(0, 1, (num_agents, num_features))
        obs_bundles = np.random.choice([0, 1], size=(num_agents, num_items), p=[0.6, 0.4])
        errors = np.random.normal(0, 0.1, size=(num_agents, num_items))
        
        print("\n1. STEP-BY-STEP WORKFLOW:")
        print("-" * 30)
        
        # Step-by-step workflow
        model = BundleChoiceModel()
        model.add_dimensions(agents=num_agents, items=num_items, features=num_features)
        model.add_data(agent_features, obs_bundles, errors)
        model.add_features(simple_features_oracle)
        model.add_subproblem('Greedy')
        model.add_estimation_method('Ellipsoid', num_iters=10, verbose=False)
        
        theta_hat = model.solve()
        print(f"Result: {theta_hat}")
        
        print("\n2. BUILDER PATTERN WORKFLOW:")
        print("-" * 30)
        
        # Builder pattern workflow
        builder = BundleChoiceBuilder()
        model2 = (builder
            .dimensions(agents=num_agents, items=num_items, features=num_features)
            .data(agent_features, obs_bundles, errors)
            .features(simple_features_oracle)
            .subproblem('Greedy')
            .estimation('Ellipsoid', num_iters=10, verbose=False)
            .build())
            
        theta_hat2 = model2.solve()
        print(f"Result: {theta_hat2}")
        
        print("\n3. FLUENT INTERFACE WORKFLOW:")
        print("-" * 30)
        
        # Fluent interface workflow
        theta_hat3 = (BundleChoiceModel()
            .add_dimensions(agents=num_agents, items=num_items, features=num_features)
            .add_data(agent_features, obs_bundles, errors)
            .add_features(simple_features_oracle)
            .add_subproblem('Greedy')
            .add_estimation_method('Ellipsoid', num_iters=10, verbose=False)
            .solve())
            
        print(f"Result: {theta_hat3}")
        
        print("\n" + "="*60)
        print("PROTOTYPE COMPARISON WITH CURRENT WORKFLOW")
        print("="*60)
        
        print("\nCURRENT WORKFLOW:")
        print("-" * 20)
        print("bc = BundleChoice()")
        print("bc.load_config(cfg)  # Manual config dict")
        print("bc.data.load_and_scatter(data)  # Manual data loading")
        print("bc.features.set_oracle(oracle)  # Manual feature setup")
        print("bc.subproblems.load()  # Required but not obvious!")
        print("bc.ellipsoid.solve()  # Manual solver selection")
        
        print("\nPROTOTYPE WORKFLOW:")
        print("-" * 20)
        print("model = BundleChoiceModel()")
        print("model.add_dimensions(agents=20, items=5, features=2)")
        print("model.add_data(agent_features, obs_bundles, errors)")
        print("model.add_features(simple_features_oracle)")
        print("model.add_subproblem('Greedy')")
        print("model.add_estimation_method('Ellipsoid', num_iters=10)")
        print("theta_hat = model.solve()")
        
        print("\nBENEFITS OF PROTOTYPE:")
        print("-" * 20)
        print("✅ Self-documenting code")
        print("✅ Clear workflow steps")
        print("✅ Validation at each step")
        print("✅ Better error messages")
        print("✅ No hidden dependencies")
        print("✅ Easy to modify and experiment")
        
        print("\n" + "="*60)
        print("PROTOTYPE TESTING COMPLETED")
        print("="*60)

if __name__ == "__main__":
    test_prototype_workflows() 