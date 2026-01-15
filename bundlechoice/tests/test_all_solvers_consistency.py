import numpy as np
import pytest
from mpi4py import MPI
from bundlechoice.core import BundleChoice

@pytest.mark.mpi
@pytest.mark.integration
def test_row_generation_mpi_optimizations():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_obs = 100
    num_items = 30
    num_features = 5
    num_simulations = 1
    sigma = 0.1
    np.random.seed(42)
    if rank == 0:
        modular_agent = np.random.normal(0, 1, (num_obs, num_items, num_features))
        errors = sigma * np.random.normal(0, 1, size=(num_simulations, num_obs, num_items))
        input_data = {'agent_data': {'modular': modular_agent}, 'errors': errors}
    else:
        input_data = None
    theta_true = np.array([1.0, 0.8, 1.2, 0.9, 1.1])
    bc_obs = BundleChoice()
    bc_obs.load_config({'dimensions': {'num_obs': num_obs, 'num_items': num_items, 'num_features': num_features, 'num_simulations': num_simulations}, 'subproblem': {'name': 'Greedy'}})
    bc_obs.data.load_input_data(input_data)
    bc_obs.oracles.build_quadratic_features_from_data()
    obs_bundles = bc_obs.subproblems.initialize_and_solve_subproblems(theta_true)
    if rank == 0:
        input_data['obs_bundle'] = obs_bundles
    if rank == 0:
        print('\n' + '=' * 70)
        print('SOLVER 1: Row Generation')
        print('=' * 70)
    bc_rg = BundleChoice()
    bc_rg.load_config({'dimensions': {'num_obs': num_obs, 'num_items': num_items, 'num_features': num_features, 'num_simulations': num_simulations}, 'subproblem': {'name': 'Greedy'}, 'row_generation': {'max_iters': 50, 'tolerance_optimality': 0.001, 'gurobi_settings': {'OutputFlag': 0}}})
    bc_rg.data.load_input_data(input_data)
    bc_rg.oracles.build_quadratic_features_from_data()
    bc_rg.subproblems.load()
    result_rg = bc_rg.row_generation.solve()
    if rank == 0:
        theta_rg = result_rg.theta_hat
        obj_rg = bc_rg.row_generation.master_model.ObjVal
        print(f'  Theta: {theta_rg}')
        print(f'  ObjVal: {obj_rg:.4f}')
        print(f'  Error from truth: {np.linalg.norm(theta_rg - theta_true):.4f}')
    if rank == 0:
        print('\n' + '=' * 70)
        print('VERIFICATION')
        print('=' * 70)
        error_from_truth = np.linalg.norm(theta_rg - theta_true)
        print(f'  True theta:      {theta_true}')
        print(f'  Estimated theta: {theta_rg}')
        print(f'  Error: {error_from_truth:.4f}')
        print(f'  ObjVal: {obj_rg:.4f}')
        assert error_from_truth < 0.5, f'Solution too far from truth: {error_from_truth}'
        assert obj_rg < 100, f'Objective too large: {obj_rg}'
        print('\n✅ MPI optimizations working correctly!')
        print('   - broadcast_array for theta')
        print('   - concatenate_array_at_root_fast for features/bundles')
        print('=' * 70)
    comm.Barrier()
if __name__ == '__main__':
    test_row_generation_mpi_optimizations()