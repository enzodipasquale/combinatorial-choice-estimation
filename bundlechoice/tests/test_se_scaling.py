import numpy as np
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.scenarios import ScenarioLibrary

def greedy_oracle(agent_idx: int, bundles: np.ndarray, data: dict):
    modular = data['agent_data']['modular'][agent_idx]
    modular = np.atleast_2d(modular)
    single = bundles.ndim == 1
    if single:
        bundles = bundles[:, None]
    modular_feat = modular.T @ bundles
    quad_feat = -np.sum(bundles, axis=0, keepdims=True) ** 2
    features = np.vstack((modular_feat, quad_feat))
    return features[:, 0] if single else features

def run_with_sample_size(comm, num_obs, seed):
    rank = comm.Get_rank()
    NUM_FEATURES = 3
    scenario = ScenarioLibrary.greedy().with_dimensions(num_obs=num_obs, num_items=30).with_num_features(NUM_FEATURES).build()
    prepared = scenario.prepare(comm=comm, timeout_seconds=60, seed=seed)
    bc = BundleChoice()
    config = {'dimensions': {'num_obs': num_obs, 'num_items': 30, 'num_features': NUM_FEATURES, 'num_simulations': 1}, 'subproblem': {'name': 'Greedy'}, 'row_generation': {'max_iters': 200, 'theta_ubs': 100, 'tolerance_optimality': 1e-08}}
    bc.load_config(config)
    bc.data.load_input_data(prepared.estimation_data if rank == 0 else None)
    bc.oracles.set_features_oracle(greedy_oracle)
    bc.subproblems.load()
    from bundlechoice.subproblems.registry.greedy import GreedySubproblem
    from bundlechoice.scenarios.greedy import _install_find_best_item
    if isinstance(bc.subproblems.subproblem_instance, GreedySubproblem):
        _install_find_best_item(bc.subproblems.subproblem_instance)
    bc.subproblems.initialize_local()
    result = bc.row_generation.solve()
    se_result = bc.standard_errors.compute(theta_hat=result.theta_hat if rank == 0 else None, num_simulations=15, step_size=0.01, seed=1995)
    if rank == 0 and se_result is not None:
        return {'theta': result.theta_hat, 'se': se_result.se, 'A_cond': np.linalg.cond(se_result.A_matrix), 'num_obs': num_obs}
    return None

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        print('\n' + '=' * 70)
        print('SE SCALING TEST: Verify SE ~ 1/√N')
        print('=' * 70)
    sample_sizes = [100, 200, 400]
    results = {}
    for N in sample_sizes:
        if rank == 0:
            print(f'\n--- N = {N} agents ---')
        res = run_with_sample_size(comm, N, seed=42 + N)
        comm.Barrier()
        if rank == 0:
            results[N] = res
            print(f"  SE: {res['se']}")
            print(f"  A cond: {res['A_cond']:.2e}")
    if rank == 0:
        print('\n' + '=' * 70)
        print('SCALING ANALYSIS')
        print('=' * 70)
        print('\nSE values:')
        print(f"{'N':<8} {'SE[0]':>10} {'SE[1]':>10} {'SE[2]':>10} {'Mean SE':>10}")
        print('-' * 50)
        for N in sample_sizes:
            r = results[N]
            print(f"{N:<8} {r['se'][0]:>10.4f} {r['se'][1]:>10.4f} {r['se'][2]:>10.4f} {r['se'].mean():>10.4f}")
        print('\nSE * √N (should be roughly constant if SE ~ 1/√N):')
        print(f"{'N':<8} {'SE[0]*√N':>12} {'SE[1]*√N':>12} {'SE[2]*√N':>12}")
        print('-' * 50)
        for N in sample_sizes:
            r = results[N]
            se_sqrt_n = r['se'] * np.sqrt(N)
            print(f'{N:<8} {se_sqrt_n[0]:>12.4f} {se_sqrt_n[1]:>12.4f} {se_sqrt_n[2]:>12.4f}')
        print('\nScaling ratios (should be ~√2 ≈ 1.41):')
        for i in range(len(sample_sizes) - 1):
            N1, N2 = (sample_sizes[i], sample_sizes[i + 1])
            ratio = results[N1]['se'].mean() / results[N2]['se'].mean()
            expected = np.sqrt(N2 / N1)
            print(f'  SE({N1})/SE({N2}) = {ratio:.3f} (expected: {expected:.3f})')
        ratio_100_200 = results[100]['se'].mean() / results[200]['se'].mean()
        ratio_200_400 = results[200]['se'].mean() / results[400]['se'].mean()
        print('\n' + '=' * 70)
        if 1.2 < ratio_100_200 < 1.6 and 1.2 < ratio_200_400 < 1.6:
            print('✓ SCALING LOOKS CORRECT: SE ~ 1/√N')
        else:
            print('⚠ SCALING MAY BE OFF - check results')
        print('=' * 70)
if __name__ == '__main__':
    main()