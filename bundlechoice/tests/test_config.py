import pytest
from bundlechoice.config import (
    BundleChoiceConfig, DimensionsConfig, SubproblemConfig,
    RowGenerationConfig, EllipsoidConfig, StandardErrorsConfig
)

def test_dimensions_config():
    dc = DimensionsConfig(num_obs=10, num_items=5, num_features=3, num_simulations=2)
    assert dc.num_obs == 10
    assert dc.num_items == 5
    assert dc.num_features == 3
    assert dc.num_simulations == 2
    assert dc.num_agents == 20

def test_dimensions_config_num_agents_none():
    dc = DimensionsConfig(num_obs=None, num_items=5, num_features=3, num_simulations=2)
    assert dc.num_agents is None

def test_subproblem_config():
    sc = SubproblemConfig(name='Greedy', settings={'param': 1})
    assert sc.name == 'Greedy'
    assert sc.settings['param'] == 1

def test_row_generation_config():
    rgc = RowGenerationConfig(tolerance_optimality=1e-5, max_iters=100, verbose=False)
    assert rgc.tolerance_optimality == 1e-5
    assert rgc.max_iters == 100
    assert rgc.verbose == False

def test_ellipsoid_config():
    ec = EllipsoidConfig(max_iterations=500, solver_precision=1e-6)
    assert ec.max_iterations == 500
    assert ec.solver_precision == 1e-6

def test_standard_errors_config():
    sec = StandardErrorsConfig(num_simulations=20, step_size=0.05, seed=123)
    assert sec.num_simulations == 20
    assert sec.step_size == 0.05
    assert sec.seed == 123

def test_bundle_choice_config():
    bc = BundleChoiceConfig()
    assert isinstance(bc.dimensions, DimensionsConfig)
    assert isinstance(bc.subproblem, SubproblemConfig)
    assert isinstance(bc.row_generation, RowGenerationConfig)

def test_config_update_in_place():
    bc1 = BundleChoiceConfig()
    bc2 = BundleChoiceConfig()
    bc2.dimensions.num_obs = 20
    bc2.subproblem.name = 'LinearKnapsack'
    bc1.update_in_place(bc2)
    assert bc1.dimensions.num_obs == 20
    assert bc1.subproblem.name == 'LinearKnapsack'

def test_config_update_in_place_nested():
    bc1 = BundleChoiceConfig()
    bc2 = BundleChoiceConfig()
    bc2.row_generation.tolerance_optimality = 1e-8
    bc2.subproblem.settings = {'param': 42}
    bc1.update_in_place(bc2)
    assert bc1.row_generation.tolerance_optimality == 1e-8
    assert bc1.subproblem.settings['param'] == 42

def test_config_from_dict():
    cfg_dict = {
        'dimensions': {'num_obs': 15, 'num_items': 8},
        'subproblem': {'name': 'Greedy'},
        'row_generation': {'max_iters': 50}
    }
    bc = BundleChoiceConfig.from_dict(cfg_dict)
    assert bc.dimensions.num_obs == 15
    assert bc.dimensions.num_items == 8
    assert bc.subproblem.name == 'Greedy'
    assert bc.row_generation.max_iters == 50

def test_config_from_dict_partial():
    cfg_dict = {
        'dimensions': {'num_obs': 15}
    }
    bc = BundleChoiceConfig.from_dict(cfg_dict)
    assert bc.dimensions.num_obs == 15
    assert bc.dimensions.num_items is None

def test_config_from_dict_empty():
    bc = BundleChoiceConfig.from_dict({})
    assert bc.dimensions.num_obs is None
