import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from mpi4py import MPI
from bundlechoice.utils import get_logger, suppress_output, format_number

logger = get_logger(__name__)


class DistributedBootstrapMixin:

    def compute_distributed_bootstrap(self, num_bootstrap=100, seed=None, verbose=False,
                                      method='bayesian',
                                      row_gen_iteration_callback=None,
                                      row_gen_initialization_callback=None,
                                      bootstrap_callback=None):
        self.verbose = verbose
        self.row_gen = self.row_generation_manager
        comm = self.comm_manager
        K = num_bootstrap
        t0 = time.perf_counter()

        # === Phase 1: Point estimation ===
        uniform_weights = np.ones(self.data_manager.num_local_agent)
        self.point_result = self.row_gen.solve(
            local_obs_weights=uniform_weights,
            initialize_master=True, initialize_subproblems=True,
            iteration_callback=row_gen_iteration_callback,
            initialization_callback=row_gen_initialization_callback,
            verbose=verbose)

        # === Phase 2: Extract and broadcast base model data ===
        base_data = self._extract_and_broadcast_base_model()

        # === Phase 3: Generate and distribute bootstrap weights ===
        if method == 'bayesian':
            weights = self.generate_weights_bayesian_bootstrap(seed, K)
        else:
            weights = self.generate_weights_standard_bootstrap(seed, K)
        local_weights = comm.Scatterv_by_row(
            weights, row_counts=self.data_manager.agent_counts,
            dtype=np.float64, shape=(self.dim.n_agents, K))
        # Broadcast full weights so each master rank has its u_obj column
        all_weights = comm.bcast(weights if comm.is_root() else None)

        # === Phase 4: Build master models (1 Allreduce + 1 bcast, then local builds) ===
        local_features = self.oracles_manager.features_oracle(self.data_manager.local_obs_bundles)
        master = self._setup_all_masters(K, base_data, local_weights, local_features, all_weights)
        del all_weights

        # === Phase 5: Distributed row generation ===
        theta_boots = self._distributed_rg_loop(
            K, local_weights, master,
            row_gen_iteration_callback, bootstrap_callback)

        # === Phase 6: Statistics ===
        total_time = time.perf_counter() - t0
        return self._gather_and_compute_stats(K, theta_boots, total_time)

    # -------------------------------------------------------------------------
    # Phase 2
    # -------------------------------------------------------------------------

    def _extract_and_broadcast_base_model(self):
        comm = self.comm_manager
        if comm.is_root():
            model = self.row_gen.master_model
            A = model.getA()
            rhs = np.array(model.getAttr('RHS', model.getConstrs()))
            sense = np.array([c.Sense for c in model.getConstrs()])
            tv, uv = self.row_gen.master_variables
            theta_lb = np.array([tv[i].LB for i in range(self.dim.n_features)])
            theta_ub = np.array([tv[i].UB for i in range(self.dim.n_features)])
            data = {'A': A, 'rhs': rhs, 'sense': sense,
                    'theta_lb': theta_lb, 'theta_ub': theta_ub}
        else:
            data = None
        return comm.bcast(data)

    # -------------------------------------------------------------------------
    # Phase 4
    # -------------------------------------------------------------------------

    def _setup_all_masters(self, K, base_data, local_weights, local_features, all_weights):
        """One Allreduce for theta objectives, then each master builds its model."""
        comm = self.comm_manager

        # All theta_obj_coefs at once: (K, n_features)
        local_theta_obj_all = -local_weights.T @ local_features       # (K, n_features)
        theta_obj_all = np.empty_like(local_theta_obj_all)
        comm.comm.Allreduce(local_theta_obj_all, theta_obj_all, op=MPI.SUM)

        master = None
        if comm.rank < K:
            k = comm.rank
            master = self._build_master_model(
                base_data, theta_obj_all[k], all_weights[:, k])
        return master

    def _build_master_model(self, base_data, theta_obj_coef, u_obj_weights):
        A, rhs, sense = base_data['A'], base_data['rhs'], base_data['sense']

        with suppress_output():
            model = gp.Model()
            params = {"Method": 0, "LPWarmStart": 2, "OutputFlag": 0}
            params.update(self.config.row_generation.master_GRB_Params or {})
            for p, v in params.items():
                if v is not None:
                    model.setParam(p, v)

            theta = model.addMVar(self.dim.n_features, obj=theta_obj_coef,
                                  lb=base_data['theta_lb'], ub=base_data['theta_ub'],
                                  name='parameter')
            u = model.addMVar(self.dim.n_agents, lb=0, obj=u_obj_weights, name='utility')
            model.update()

            all_vars = model.getVars()
            all_mvar = gp.MVar.fromlist(all_vars)
            for s in np.unique(sense):
                mask = sense == s
                model.addMConstr(A[mask], all_mvar, s, rhs[mask])

            model.update()
            model.optimize()

        return {'model': model, 'theta': theta, 'u': u}

    # -------------------------------------------------------------------------
    # Phase 5
    # -------------------------------------------------------------------------

    def _distributed_rg_loop(self, K, local_weights, master,
                             iteration_callback, bootstrap_callback):
        comm = self.comm_manager
        cfg = self.config.row_generation
        rank = comm.rank
        n_features, n_agents = self.dim.n_features, self.dim.n_agents
        agent_counts = self.data_manager.agent_counts
        n_local = self.data_manager.num_local_agent

        active = np.ones(K, dtype=np.int32)
        boot_iters = np.zeros(K, dtype=np.int32)
        theta_results = [None] * K
        n_converged_total = 0

        # Precompute local agent offset (constant across rounds)
        local_agent_start = int(agent_counts[:rank].sum())

        # Pre-allocate buffers for Allreduce of thetas and u's
        all_thetas_buf = np.zeros((K, n_features), dtype=np.float64)
        all_u_buf = np.zeros((K, n_agents), dtype=np.float64)
        all_thetas = np.empty_like(all_thetas_buf)
        all_u = np.empty_like(all_u_buf)

        # Info fields sent from converged master to root:
        # [obj_val, n_constraints, reduced_cost, n_iters]
        _INFO_LEN = 4

        if self.verbose and comm.is_root():
            logger.info(" ")
            logger.info(" DISTRIBUTED BOOTSTRAP (%d samples, %d ranks)", K, comm.comm_size)

        t_boot_start = time.perf_counter()
        rg_round = 0
        while rg_round < cfg.max_iters:
            n_active = int(active.sum())
            if n_active == 0:
                break

            active_masters = np.where(active)[0]

            # --- Callback before pricing ---
            if iteration_callback is not None:
                iteration_callback(int(boot_iters[active_masters[0]]), self.row_gen)

            # === Step 1: Broadcast all thetas and u's (2 Allreduces) ===
            t_price = time.perf_counter()

            all_thetas_buf[:] = 0.0
            all_u_buf[:] = 0.0
            if rank < K and active[rank]:
                all_thetas_buf[rank] = master['theta'].X
                all_u_buf[rank] = master['u'].X

            comm.comm.Allreduce(all_thetas_buf, all_thetas, op=MPI.SUM)
            comm.comm.Allreduce(all_u_buf, all_u, op=MPI.SUM)

            # === Step 2: Solve subproblems for each active theta ===
            local_viols = {}
            for k in active_masters:
                theta_k = all_thetas[k]
                u_local_k = all_u[k, local_agent_start:local_agent_start + n_local]

                bundles = self.subproblem_manager.solve_subproblems(theta_k)
                local_viols[k] = self.row_gen._compute_local_violations(
                    bundles, theta_k, u_local_k, local_weights[:, k])
            t_price = time.perf_counter() - t_price

            # === Step 3: Batched convergence check (1 Allgather) ===
            t_comm = time.perf_counter()

            local_meta_all = np.zeros((K, 2), dtype=np.float64)
            for k in active_masters:
                max_rc, n_viol = local_viols[k][0], local_viols[k][1]
                local_meta_all[k] = [max_rc, n_viol]

            all_meta_all = comm.Allgather(local_meta_all.ravel()).reshape(comm.comm_size, K, 2)
            global_max_rc = all_meta_all[:, :, 0].max(axis=0)       # (K,)
            global_n_viol = all_meta_all[:, :, 1].sum(axis=0).astype(np.int64)  # (K,)
            viol_counts_all = all_meta_all[:, :, 1].astype(np.int64) # (comm_size, K)
            convergence_flags = global_max_rc <= cfg.tolerance       # (K,)

            # Gatherv violations to each master (only non-converged)
            gathered_data = {}
            for k in active_masters:
                if convergence_flags[k]:
                    continue
                viol_counts = viol_counts_all[:, k]
                _, _, viol_ids, viol_bun, viol_fe = local_viols[k]
                g_bun = comm.Gatherv_by_row(viol_bun, row_counts=viol_counts, root=k)
                g_fe = comm.Gatherv_by_row(viol_fe, row_counts=viol_counts, root=k)
                g_ids = comm.Gatherv_by_row(viol_ids, row_counts=viol_counts, root=k)
                if rank == k:
                    gathered_data[k] = (g_ids, g_fe[:, :-1], g_fe[:, -1])
            t_comm = time.perf_counter() - t_comm

            # === Step 4: All masters solve in parallel (no communication) ===
            t_master = time.perf_counter()
            if rank < K and active[rank] and not convergence_flags[rank]:
                ids, features, errors = gathered_data[rank]
                master['model'].addConstr(
                    master['u'][ids] >= features @ master['theta'] + errors)
                master['model'].optimize()
            t_master = time.perf_counter() - t_master

            # === Step 5: Update bookkeeping ===
            newly_converged = []
            for k in active_masters:
                boot_iters[k] += 1
                if convergence_flags[k] and boot_iters[k] >= cfg.min_iters:
                    newly_converged.append(k)

            # Collect theta + info from converged masters
            converged_info = {}
            for k in newly_converged:
                active[k] = 0
                if rank == k:
                    theta_results[k] = master['theta'].X.copy()
                    info_k = np.array([
                        master['model'].ObjVal,
                        master['model'].NumConstrs,
                        global_max_rc[k],
                        boot_iters[k]
                    ], dtype=np.float64)
                if k == 0:
                    if rank == 0:
                        converged_info[k] = info_k
                else:
                    if rank == k:
                        comm.comm.Send(np.ascontiguousarray(theta_results[k]), dest=0, tag=k)
                        comm.comm.Send(info_k, dest=0, tag=K + k)
                    elif rank == 0:
                        theta_results[k] = np.empty(n_features, dtype=np.float64)
                        comm.comm.Recv(theta_results[k], source=k, tag=k)
                        info_k = np.empty(_INFO_LEN, dtype=np.float64)
                        comm.comm.Recv(info_k, source=k, tag=K + k)
                        converged_info[k] = info_k

            n_converged_total += len(newly_converged)

            # === Logging ===
            if self.verbose and comm.is_root():
                param_indices = cfg.parameters_to_log or list(range(min(5, n_features)))
                self._log_rg_round_header(rg_round, param_indices)
                active_rc = global_max_rc[active_masters]
                active_viol = global_n_viol[active_masters]
                max_rc = active_rc.max()
                total_viol = int(active_viol.sum())
                logger.info(
                    " %5d  %4d  %8.1fs  %8.1fs  %8.1fs  %s  %11d",
                    rg_round, n_active, t_price, t_comm, t_master,
                    self._fmt_rc(max_rc), total_viol)

                if newly_converged:
                    param_labels = ' '.join(f"{'θ['+str(i)+']':>10}" for i in param_indices)
                    logger.info(
                        "       %s  %s  %s  %s  %s  %s",
                        'Boot'.rjust(6), '#Constr'.rjust(7),
                        'Reduced Cost'.rjust(14), 'Objective'.rjust(12),
                        'Range θ'.center(15), param_labels)
                    for k in newly_converged:
                        theta_k = theta_results[k]
                        info = converged_info.get(k)
                        if info is not None:
                            obj_val, n_constr, red_cost, n_iters = info
                            param_vals = ' '.join(
                                format_number(theta_k[i], width=10, precision=5)
                                for i in param_indices)
                            rng = f"[{theta_k.min():.1f}, {theta_k.max():.1f}]"
                            logger.info(
                                "       ↳  %4d  %7d  %s  %s  %-15s  %s",
                                k, int(n_constr),
                                self._fmt_rc(red_cost),
                                format_number(obj_val, width=12, precision=5),
                                rng, param_vals)

            rg_round += 1

        # Collect remaining active masters (hit max_iters)
        for k in np.where(active)[0]:
            if rank == k:
                theta_results[k] = master['theta'].X.copy()
            if k != 0:
                if rank == k:
                    comm.comm.Send(np.ascontiguousarray(theta_results[k]), dest=0, tag=2 * K + k)
                elif rank == 0:
                    theta_results[k] = np.empty(n_features, dtype=np.float64)
                    comm.comm.Recv(theta_results[k], source=k, tag=2 * K + k)

        return theta_results if rank == 0 else None

    # -------------------------------------------------------------------------
    # Logging helpers
    # -------------------------------------------------------------------------

    def _log_rg_round_header(self, rg_round, param_indices):
        """Print header every 80 rounds."""
        if rg_round % 80 != 0:
            return
        h1 = f" {'Round':>5}  {'Act.':>4}  {'Pricing':>9}  {'Comm':>9}  {'Master':>9}  {'Max Reduced':>14}  {'Total':>11}"
        h2 = f" {'':>5}  {'':>4}  {'(s)':>9}  {'(s)':>9}  {'(s)':>9}  {'Cost':>14}  {'#Viol':>11}"
        sep = "-" * len(h1)
        logger.info(sep)
        logger.info(h1)
        logger.info(h2)
        logger.info(sep)

    @staticmethod
    def _fmt_rc(val, width=14):
        """Format reduced cost: scientific notation for tiny values."""
        if abs(val) < 1e-6 and val != 0:
            return f"{val:.5e}".rjust(width)
        return format_number(val, width=width, precision=6)

    # -------------------------------------------------------------------------
    # Phase 6
    # -------------------------------------------------------------------------

    def _gather_and_compute_stats(self, K, theta_boots, total_time):
        if not self.comm_manager.is_root():
            return None
        theta_hat = self.point_result.theta_hat
        stats = self.compute_bootstrap_stats(theta_boots, theta_hat=theta_hat)
        if self.verbose:
            self._log_distributed_bootstrap_summary(K, total_time, stats)
        return stats

    def _log_distributed_bootstrap_summary(self, n_bootstrap, total_time, result):
        if not self.comm_manager.is_root():
            return
        theta_hat = self.point_result.theta_hat
        idx = self.config.row_generation.parameters_to_log or list(range(min(5, self.dim.n_features)))
        logger.info(" ")
        logger.info("-" * 70)
        logger.info(" DISTRIBUTED BOOTSTRAP: %d samples in %.1fs", n_bootstrap, total_time)
        logger.info("-" * 70)
        logger.info(f"{'Param':>8} | {'Point Est':>12} | {'Boot Mean':>12} | {'SE':>12} | {'t-stat':>10}")
        logger.info("-" * 70)
        for i in idx:
            logger.info(f"  θ[{i:>3}] | {theta_hat[i]:>12.5f} | {result.mean[i]:>12.5f} | "
                        f"{result.se[i]:>12.5f} | {result.t_stats[i]:>10.2f}")
        logger.info("-" * 70)