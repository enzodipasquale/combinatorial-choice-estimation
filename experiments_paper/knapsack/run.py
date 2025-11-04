import os
import time
import numpy as np
import pandas as pd
from mpi4py import MPI
from bundlechoice.core import BundleChoice
from bundlechoice.estimation import RowGeneration1SlackSolver, EllipsoidSolver


def load_yaml_config(path: str) -> dict:
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    base_dir = os.path.dirname(__file__)
    cfg_path = os.path.join(base_dir, 'config.yaml')
    cfg = load_yaml_config(cfg_path)

    num_agents = cfg['dimensions']['num_agents']
    num_items = cfg['dimensions']['num_items']
    num_features = cfg['dimensions']['num_features']
    num_simuls = cfg['dimensions'].get('num_simuls', 1)
    sigma = cfg.get('sigma', 1.0)
    num_replications = cfg.get('num_replications', 1)
    base_seed = cfg.get('base_seed', 12345)

    results_path = os.path.join(base_dir, cfg.get('results_csv', 'results.csv'))
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0 and not os.path.exists(results_path):
        # Create header with theta columns
        cols = ['replication','seed','method','time_s','obj_value',
                'num_agents','num_items','num_features','num_simuls','sigma','subproblem']
        # Add theta_true columns
        cols.extend([f'theta_true_{k}' for k in range(num_features)])
        # Add theta_est columns  
        cols.extend([f'theta_{k}' for k in range(num_features)])
        pd.DataFrame([], columns=cols).to_csv(results_path, index=False)

    modular_agent_features = cfg.get('modular_agent_features', 3)
    modular_item_features = cfg.get('modular_item_features', 2)

    for rep in range(num_replications):
        seed = int(base_seed + rep)
        np.random.seed(seed)

        if rank == 0:
            modular_agent = np.abs(np.random.normal(0, 1, (num_agents, num_items, modular_agent_features)))
            modular_item = np.abs(np.random.normal(0, 1, (num_items, modular_item_features)))
            weights = np.random.randint(1, 11, num_items)

            mean_capacity = int(0.5 * weights.sum())
            lo = int(0.85 * mean_capacity)
            hi = int(1.15 * mean_capacity)
            capacity = np.random.randint(lo, hi + 1, size=num_agents)

            agent_data = {"modular": modular_agent, "capacity": capacity}
            item_data = {"modular": modular_item, "weights": weights}
            errors = sigma * np.random.normal(0, 1, size=(num_agents, num_items))
            estimation_errors = sigma * np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
            data = {"agent_data": agent_data, "item_data": item_data, "errors": errors}
        else:
            data = None

        bc = BundleChoice()
        bc.load_config(cfg)
        bc.data.load_and_scatter(data)
        bc.features.build_from_data()
        bc.subproblems.load()

        theta_true = np.ones(num_features)
        _ = bc.generate_observations(theta_true)

        if rank == 0:
            bc.data.input_data["errors"] = estimation_errors
        bc.data.load_and_scatter(bc.data.input_data if rank == 0 else None)

        cfg_dims = cfg.get('dimensions', {}).copy()
        cfg_dims['num_simuls'] = num_simuls
        bc.load_config({"dimensions": cfg_dims})

        try:
            results = {}
            t0 = time.time()
            theta_row = bc.row_generation.solve()
            time_row = time.time() - t0
            obj_row = bc.row_generation.objective(theta_row)
            results['row_generation'] = (theta_row, time_row, obj_row)

            rg1 = RowGeneration1SlackSolver(
                comm_manager=bc.comm_manager,
                dimensions_cfg=bc.config.dimensions,
                row_generation_cfg=bc.config.row_generation,
                data_manager=bc.data_manager,
                feature_manager=bc.feature_manager,
                subproblem_manager=bc.subproblem_manager,
            )
            t1 = time.time()
            theta_row1 = rg1.solve()
            time_row1 = time.time() - t1
            obj_row1 = bc.row_generation.objective(theta_row1)
            results['row_generation_1slack'] = (theta_row1, time_row1, obj_row1)

            # Warm start ellipsoid with row generation solution
            ell = EllipsoidSolver(
                comm_manager=bc.comm_manager,
                dimensions_cfg=bc.config.dimensions,
                ellipsoid_cfg=bc.config.ellipsoid,
                data_manager=bc.data_manager,
                feature_manager=bc.feature_manager,
                subproblem_manager=bc.subproblem_manager,
                theta_init=theta_row.copy()
            )
            t2 = time.time()
            theta_ell = ell.solve()
            time_ell = time.time() - t2
            obj_ell = bc.row_generation.objective(theta_ell)
            results['ellipsoid'] = (theta_ell, time_ell, obj_ell)

            if rank == 0:
                # Write rows with theta_true and theta_est
                rows = []
                for method, (theta, elapsed, objv) in results.items():
                    row = {
                        'replication': rep,
                        'seed': seed,
                        'method': method,
                        'time_s': elapsed,
                        'obj_value': objv,
                        'num_agents': num_agents,
                        'num_items': num_items,
                        'num_features': num_features,
                        'num_simuls': num_simuls,
                        'sigma': sigma,
                        'subproblem': cfg['subproblem']['name']
                    }
                    # Add theta_true columns
                    for k in range(num_features):
                        row[f'theta_true_{k}'] = theta_true[k]
                    # Add theta_est columns
                    for k in range(num_features):
                        row[f'theta_{k}'] = theta[k]
                    rows.append(row)
                pd.DataFrame(rows).to_csv(results_path, mode='a', header=False, index=False)

                th_rg, _, obj_rg = results['row_generation']
                th_r1, _, obj_r1 = results['row_generation_1slack']
                th_el, _, obj_el = results['ellipsoid']
                print(f"[rep {rep}] theta close rg vs 1slack: {np.allclose(th_rg, th_r1, atol=1e-2, rtol=0)}; obj close: {abs(obj_rg-obj_r1) < 1e-3}")
                print(f"[rep {rep}] theta close rg vs ellipsoid: {np.allclose(th_rg, th_el, atol=1e-2, rtol=0)}; obj close: {abs(obj_rg-obj_el) < 1e-3}")
        except Exception as e:
            if rank == 0:
                pd.DataFrame([{
                    'replication': rep,
                    'seed': seed,
                    'method': 'ERROR',
                    'time_s': np.nan,
                    'obj_value': np.nan,
                    'num_agents': num_agents,
                    'num_items': num_items,
                    'num_features': num_features,
                    'num_simuls': num_simuls,
                    'sigma': sigma,
                    'subproblem': cfg['subproblem']['name'],
                    'error': str(e)
                }]).to_csv(results_path, mode='a', header=False, index=False)
            raise


if __name__ == '__main__':
    main()


