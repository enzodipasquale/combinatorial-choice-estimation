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

    # Dimensions - allow override via environment variables
    num_agents = int(os.environ.get('NUM_AGENTS', cfg['dimensions']['num_agents']))
    num_items = int(os.environ.get('NUM_ITEMS', cfg['dimensions']['num_items']))
    num_features = cfg['dimensions']['num_features']
    num_simuls = cfg['dimensions'].get('num_simuls', 1)
    sigma = cfg.get('sigma', 2.0)
    num_replications = cfg.get('num_replications', 1)
    base_seed = cfg.get('base_seed', 12345)

    results_path = os.path.join(base_dir, cfg.get('results_csv', 'results.csv'))

    if rank == 0 and not os.path.exists(results_path):
        # Create header with theta columns and timing breakdown
        cols = ['replication','seed','method','time_s','obj_value',
                'num_agents','num_items','num_features','num_simuls','sigma','subproblem']
        # Add timing breakdown columns (universal across all methods)
        cols.extend(['timing_compute', 'timing_solve', 'timing_comm',
                    'timing_compute_pct', 'timing_solve_pct', 'timing_comm_pct'])
        # Add objective consistency check columns
        cols.extend(['obj_diff_rg_1slack', 'obj_diff_rg_ellipsoid', 'obj_diff_1slack_ellipsoid', 'obj_close_all'])
        # Add theta_true columns
        cols.extend([f'theta_true_{k}' for k in range(num_features)])
        # Add theta_est columns  
        cols.extend([f'theta_{k}' for k in range(num_features)])
        pd.DataFrame([], columns=cols).to_csv(results_path, index=False)

    modular_agent_features = cfg.get('modular_agent_features', 4)
    quadratic_item_features = cfg.get('quadratic_item_features', 1)
    modular_item_features = cfg.get('modular_item_features', 1)

    for rep in range(num_replications):
        seed = int(base_seed + rep)
        np.random.seed(seed)

        if rank == 0:
            # Modular agent features
            modular_agent = np.abs(np.random.normal(0, 1, (num_agents, num_items, modular_agent_features)))

            # Quadratic item features (supermodular part)
            quadratic_item = 1 * np.random.choice([0, 1], size=(num_items, num_items, quadratic_item_features), p=[0.85, 0.15])
            quadratic_item *= (1 - np.eye(num_items, dtype=int))[:, :, None]

            # Modular item features and knapsack weights
            modular_item = np.abs(np.random.normal(0, 1, (num_items, modular_item_features)))
            weights = np.random.randint(1, 11, num_items)

            # Agent capacities
            mean_capacity = int(0.5 * weights.sum())
            lo = int(0.85 * mean_capacity)
            hi = int(1.15 * mean_capacity)
            capacity = np.random.randint(lo, hi + 1, size=num_agents)

            agent_data = {"modular": modular_agent, "capacity": capacity}
            item_data = {"modular": modular_item, "quadratic": quadratic_item, "weights": weights}

            errors = sigma * np.random.normal(0, 1, size=(num_agents, num_items))
            estimation_errors = sigma * np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))
            data = {"agent_data": agent_data, "item_data": item_data, "errors": errors}
        else:
            data = None

        # Setup BundleChoice
        bc = BundleChoice()
        # Override dimensions if set via environment variables
        cfg_override = cfg.copy()
        if 'dimensions' in cfg_override:
            cfg_override['dimensions'] = cfg_override['dimensions'].copy()
            cfg_override['dimensions']['num_agents'] = num_agents
            cfg_override['dimensions']['num_items'] = num_items
        bc.load_config(cfg_override)
        bc.data.load_and_scatter(data)
        bc.features.build_from_data()
        bc.subproblems.load()

        theta_true = np.ones(num_features)
        _ = bc.generate_observations(theta_true)

        if rank == 0:
            bc.data.input_data["errors"] = estimation_errors
        bc.data.load_and_scatter(bc.data.input_data if rank == 0 else None)

        # Ensure num_simuls for estimation - need to override again
        cfg_dims_est = cfg.get('dimensions', {}).copy()
        cfg_dims_est['num_agents'] = num_agents
        cfg_dims_est['num_items'] = num_items
        cfg_dims_est['num_simuls'] = num_simuls
        bc.load_config({"dimensions": cfg_dims_est})

        # Run solvers and time them
        try:
            results = {}

            t0 = time.time()
            theta_row = bc.row_generation.solve()
            time_row = time.time() - t0
            obj_row = bc.row_generation.objective(theta_row)
            # Extract timing stats
            timing_rg = bc.row_generation.timing_stats if bc.row_generation.timing_stats else None
            results['row_generation'] = (theta_row, time_row, obj_row, timing_rg)

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
            # Extract timing stats
            timing_rg1 = rg1.timing_stats if rg1.timing_stats else None
            results['row_generation_1slack'] = (theta_row1, time_row1, obj_row1, timing_rg1)

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
            # Extract timing stats
            timing_ell = ell.timing_stats if ell.timing_stats else None
            results['ellipsoid'] = (theta_ell, time_ell, obj_ell, timing_ell)

            if rank == 0:
                # Compute objective consistency checks (same for all methods in this replication)
                th_rg, _, obj_rg, _ = results['row_generation']
                th_r1, _, obj_r1, _ = results['row_generation_1slack']
                th_el, _, obj_el, _ = results['ellipsoid']
                
                obj_diff_rg_1slack = abs(obj_rg - obj_r1)
                obj_diff_rg_ellipsoid = abs(obj_rg - obj_el)
                obj_diff_1slack_ellipsoid = abs(obj_r1 - obj_el)
                obj_tolerance = 1e-3
                obj_close_all = (obj_diff_rg_1slack < obj_tolerance and 
                                obj_diff_rg_ellipsoid < obj_tolerance and 
                                obj_diff_1slack_ellipsoid < obj_tolerance)
                
                # Write rows with theta_true, theta_est, and timing breakdown
                rows = []
                for method, (theta, elapsed, objv, timing_stats) in results.items():
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
                    # Add timing breakdown (map different methods to common fields)
                    if timing_stats:
                        # Row generation: pricing_time -> compute, master_time -> solve, mpi_time -> comm
                        # Ellipsoid: gradient_time -> compute, update_time -> solve, mpi_time -> comm
                        row['timing_compute'] = timing_stats.get('pricing_time', 
                                                                 timing_stats.get('gradient_time', np.nan))
                        row['timing_solve'] = timing_stats.get('master_time',
                                                                timing_stats.get('update_time', np.nan))
                        row['timing_comm'] = timing_stats.get('mpi_time', np.nan)
                        row['timing_compute_pct'] = timing_stats.get('pricing_time_pct',
                                                                     timing_stats.get('gradient_time_pct', np.nan))
                        row['timing_solve_pct'] = timing_stats.get('master_time_pct',
                                                                    timing_stats.get('update_time_pct', np.nan))
                        row['timing_comm_pct'] = timing_stats.get('mpi_time_pct', np.nan)
                    else:
                        row['timing_compute'] = np.nan
                        row['timing_solve'] = np.nan
                        row['timing_comm'] = np.nan
                        row['timing_compute_pct'] = np.nan
                        row['timing_solve_pct'] = np.nan
                        row['timing_comm_pct'] = np.nan
                    # Add objective consistency checks (same for all methods in replication)
                    row['obj_diff_rg_1slack'] = obj_diff_rg_1slack
                    row['obj_diff_rg_ellipsoid'] = obj_diff_rg_ellipsoid
                    row['obj_diff_1slack_ellipsoid'] = obj_diff_1slack_ellipsoid
                    row['obj_close_all'] = obj_close_all
                    # Add theta_true columns
                    for k in range(num_features):
                        row[f'theta_true_{k}'] = theta_true[k]
                    # Add theta_est columns
                    for k in range(num_features):
                        row[f'theta_{k}'] = theta[k]
                    rows.append(row)
                df = pd.DataFrame(rows)
                df.to_csv(results_path, mode='a', header=False, index=False)

                # Print quick consistency checks
                print(f"[rep {rep}] theta close rg vs 1slack: {np.allclose(th_rg, th_r1, atol=1e-2, rtol=0)}; obj diff: {obj_diff_rg_1slack:.6f} (close: {obj_diff_rg_1slack < obj_tolerance})")
                print(f"[rep {rep}] theta close rg vs ellipsoid: {np.allclose(th_rg, th_el, atol=1e-2, rtol=0)}; obj diff: {obj_diff_rg_ellipsoid:.6f} (close: {obj_diff_rg_ellipsoid < obj_tolerance})")
                print(f"[rep {rep}] obj close all methods: {obj_close_all} (tolerance: {obj_tolerance})")
        except Exception as e:
            if rank == 0:
                # Create error row with all required columns
                error_row = {
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
                    'timing_compute': np.nan,
                    'timing_solve': np.nan,
                    'timing_comm': np.nan,
                    'timing_compute_pct': np.nan,
                    'timing_solve_pct': np.nan,
                    'timing_comm_pct': np.nan,
                    'obj_diff_rg_1slack': np.nan,
                    'obj_diff_rg_ellipsoid': np.nan,
                    'obj_diff_1slack_ellipsoid': np.nan,
                    'obj_close_all': False,
                    'error': str(e)
                }
                # Add theta columns
                for k in range(num_features):
                    error_row[f'theta_true_{k}'] = np.nan
                    error_row[f'theta_{k}'] = np.nan
                df = pd.DataFrame([error_row])
                df.to_csv(results_path, mode='a', header=False, index=False)
            raise


if __name__ == '__main__':
    main()


