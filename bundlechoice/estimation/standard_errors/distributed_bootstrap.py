import os
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
                                      bootstrap_callback=None,
                                      save_model_dir=None):
        self.verbose = verbose
        self.row_gen = self.row_generation_manager
        comm = self.comm_manager
        K = num_bootstrap
        t0 = time.perf_counter()

        # === Phase 1: Point estimation ===
        self.point_result = self.row_gen.solve(
            local_obs_weights=np.ones(self.data_manager.num_local_agent),
            initialize_master=True, initialize_subproblems=True,
            iteration_callback=row_gen_iteration_callback,
            initialization_callback=row_gen_initialization_callback,
            verbose=verbose)

        # Save point estimate
        save_dir = None
        if save_model_dir is not None and comm.is_root():
            save_dir = os.path.join(save_model_dir, "bootstrap_solutions")
            pt_dir = os.path.join(save_dir, "point_estimate")
            os.makedirs(pt_dir, exist_ok=True)
            self.row_gen.master_model.write(os.path.join(pt_dir, "master.lp"))
            self.row_gen.master_model.write(os.path.join(pt_dir, "master.sol"))
        save_dir = comm.bcast(save_dir)

        # === Phase 2: Extract and broadcast base model ===
        base_data = self._extract_and_broadcast_base_model()

        # === Phase 3: Generate and distribute bootstrap weights ===
        gen = self.generate_weights_bayesian_bootstrap if method == 'bayesian' \
              else self.generate_weights_standard_bootstrap
        weights = gen(seed, K)
        local_weights = comm.Scatterv_by_row(
            weights, row_counts=self.data_manager.agent_counts,
            dtype=np.float64, shape=(self.dim.n_agents, K))
        all_weights = comm.bcast(weights if comm.is_root() else None)

        # === Phase 4: Build master models ===
        local_features = self.oracles_manager.features_oracle(self.data_manager.local_obs_bundles)
        master = self._setup_all_masters(K, base_data, local_weights, local_features, all_weights)
        del all_weights

        # === Phase 5: Distributed row generation ===
        theta_boots = self._distributed_rg_loop(
            K, local_weights, master,
            row_gen_iteration_callback, bootstrap_callback, save_dir=save_dir)

        # === Phase 6: Statistics ===
        total_time = time.perf_counter() - t0
        return self._gather_and_compute_stats(K, theta_boots, total_time)

    # -------------------------------------------------------------------------
    # Phase 2: Extract base model
    # -------------------------------------------------------------------------

    def _extract_and_broadcast_base_model(self):
        comm = self.comm_manager
        if comm.is_root():
            model = self.row_gen.master_model
            tv, uv = self.row_gen.master_variables
            data = {
                'A': model.getA(),
                'rhs': np.array(model.getAttr('RHS', model.getConstrs())),
                'sense': np.array([c.Sense for c in model.getConstrs()]),
                'theta_lb': np.array([tv[i].LB for i in range(self.dim.n_features)]),
                'theta_ub': np.array([tv[i].UB for i in range(self.dim.n_features)]),
            }
        else:
            data = None
        return comm.bcast(data)

    # -------------------------------------------------------------------------
    # Phase 4: Build all master models
    # -------------------------------------------------------------------------

    def _setup_all_masters(self, K, base_data, local_weights, local_features, all_weights):
        comm = self.comm_manager
        local_theta_obj = -local_weights.T @ local_features  # (K, n_features)
        theta_obj_all = np.empty_like(local_theta_obj)
        comm.comm.Allreduce(local_theta_obj, theta_obj_all, op=MPI.SUM)

        if comm.rank < K:
            return self._build_master_model(base_data, theta_obj_all[comm.rank], all_weights[:, comm.rank])
        return None

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
                                  lb=base_data['theta_lb'], ub=base_data['theta_ub'], name='parameter')
            u = model.addMVar(self.dim.n_agents, lb=0, obj=u_obj_weights, name='utility')
            model.update()
            all_mvar = gp.MVar.fromlist(model.getVars())
            for s in np.unique(sense):
                mask = sense == s
                model.addMConstr(A[mask], all_mvar, s, rhs[mask])
            model.update()
            model.optimize()
        return {'model': model, 'theta': theta, 'u': u}

    # -------------------------------------------------------------------------
    # Phase 5: Distributed row generation loop
    # -------------------------------------------------------------------------

    def _distributed_rg_loop(self, K, local_weights, master,
                             iteration_callback, bootstrap_callback, save_dir=None):
        comm = self.comm_manager
        cfg = self.config.row_generation
        rank, P = comm.rank, comm.comm_size
        n_features, n_agents = self.dim.n_features, self.dim.n_agents
        agent_counts = self.data_manager.agent_counts
        n_local = self.data_manager.num_local_agent
        local_agent_start = int(agent_counts[:rank].sum())
        ROW_WIDTH = n_features + 2  # [id_as_float, features..., error]

        active = np.ones(K, dtype=np.int32)
        boot_iters = np.zeros(K, dtype=np.int32)
        theta_results = [None] * K

        # Pre-allocate Allreduce buffers
        thetas_send = np.zeros((K, n_features), dtype=np.float64)
        u_send = np.zeros((K, n_agents), dtype=np.float64)
        thetas_all = np.empty_like(thetas_send)
        u_all = np.empty_like(u_send)

        # Pre-allocate Alltoallv count buffers
        send_counts = np.zeros(P, dtype=np.int64)
        recv_counts = np.empty(P, dtype=np.int64)
        sdispls = np.empty(P, dtype=np.int64)
        rdispls = np.empty(P, dtype=np.int64)

        _INFO_LEN = 4

        if self.verbose and comm.is_root():
            logger.info(" ")
            logger.info(" DISTRIBUTED BOOTSTRAP (%d samples, %d ranks)", K, P)

        for rg_round in range(int(cfg.max_iters)):
            n_active = int(active.sum())
            if n_active == 0:
                break
            active_k = np.where(active)[0]

            if iteration_callback is not None:
                iteration_callback(int(boot_iters[active_k[0]]), self.row_gen)

            # ------ Step 1: Allreduce thetas and u's (COMM) ------
            t_comm_start = time.perf_counter()

            thetas_send[:] = 0.0
            u_send[:] = 0.0
            if rank < K and active[rank]:
                thetas_send[rank] = master['theta'].X
                u_send[rank] = master['u'].X
            comm.comm.Allreduce(thetas_send, thetas_all, op=MPI.SUM)
            comm.comm.Allreduce(u_send, u_all, op=MPI.SUM)

            t_step1 = time.perf_counter() - t_comm_start

            # ------ Step 2: Price subproblems for each active boot (PRICING) ------
            t_price_start = time.perf_counter()

            local_viols = {}
            for k in active_k:
                u_local_k = u_all[k, local_agent_start:local_agent_start + n_local]
                bundles = self.subproblem_manager.solve_subproblems(thetas_all[k])
                local_viols[k] = self.row_gen._compute_local_violations(
                    bundles, thetas_all[k], u_local_k, local_weights[:, k])

            t_price = time.perf_counter() - t_price_start

            # ------ Step 3: Convergence check + Alltoallv violations (COMM) ------
            t_comm2_start = time.perf_counter()

            # 3a. Allgather reduced costs and violation counts
            local_meta = np.zeros((K, 2), dtype=np.float64)
            for k in active_k:
                local_meta[k] = [local_viols[k][0], local_viols[k][1]]

            all_meta = comm.Allgather(local_meta.ravel()).reshape(P, K, 2)
            global_max_rc = all_meta[:, :, 0].max(axis=0)
            global_n_viol = all_meta[:, :, 1].sum(axis=0).astype(np.int64)
            converged = global_max_rc <= cfg.tolerance
            nc_set = set(k for k in active_k if not converged[k])

            # 3b. Pack send buffer in rank order and Alltoallv
            send_counts[:] = 0
            chunks = {}
            for k in nc_set:
                _, n_viol_k, viol_ids, _, viol_fe = local_viols[k]
                if n_viol_k > 0:
                    row = np.empty((n_viol_k, ROW_WIDTH), dtype=np.float64)
                    row[:, 0] = viol_ids
                    row[:, 1:] = viol_fe
                    chunks[k] = row.ravel()
                    send_counts[k] = n_viol_k * ROW_WIDTH

            sendbuf = np.concatenate([chunks[r] for r in sorted(chunks)]) \
                      if chunks else np.empty(0, dtype=np.float64)

            comm.comm.Alltoall(send_counts, recv_counts)
            sdispls[0] = rdispls[0] = 0
            np.cumsum(send_counts[:-1], out=sdispls[1:])
            np.cumsum(recv_counts[:-1], out=rdispls[1:])
            recvbuf = np.empty(int(recv_counts.sum()), dtype=np.float64)

            comm.comm.Alltoallv(
                [sendbuf, send_counts, sdispls, MPI.DOUBLE],
                [recvbuf, recv_counts, rdispls, MPI.DOUBLE])

            # Unpack on master ranks
            gathered_data = {}
            if rank < K and rank in nc_set:
                n_rows = int(recv_counts.sum()) // ROW_WIDTH
                if n_rows > 0:
                    d = recvbuf.reshape(n_rows, ROW_WIDTH)
                    gathered_data[rank] = (d[:, 0].astype(np.int64),
                                           np.ascontiguousarray(d[:, 1:-1]),
                                           np.ascontiguousarray(d[:, -1]))

            t_comm = t_step1 + (time.perf_counter() - t_comm2_start)

            # ------ Step 4: Master solves (NO COMM) ------
            t_master_start = time.perf_counter()

            if rank < K and active[rank] and not converged[rank]:
                if rank in gathered_data:
                    ids, feats, errs = gathered_data[rank]
                    if len(ids) > 0:
                        master['model'].addConstr(
                            master['u'][ids] >= feats @ master['theta'] + errs)
                master['model'].optimize()

            t_master = time.perf_counter() - t_master_start

            # ------ Step 5: Bookkeeping ------
            newly_converged = []
            for k in active_k:
                boot_iters[k] += 1
                if converged[k] and boot_iters[k] >= cfg.min_iters:
                    newly_converged.append(k)

            converged_info = {}
            for k in newly_converged:
                active[k] = 0
                if rank == k:
                    theta_results[k] = master['theta'].X.copy()
                    info_k = np.array([master['model'].ObjVal, master['model'].NumConstrs,
                                       global_max_rc[k], boot_iters[k]], dtype=np.float64)
                    if save_dir is not None:
                        d = os.path.join(save_dir, f"boot_{k:04d}")
                        os.makedirs(d, exist_ok=True)
                        master['model'].write(os.path.join(d, "master.lp"))
                        master['model'].write(os.path.join(d, "master.sol"))
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

            # ------ Logging ------
            if self.verbose and comm.is_root():
                param_idx = cfg.parameters_to_log or list(range(min(5, n_features)))
                self._log_rg_round_header(rg_round, param_idx)
                logger.info(
                    " %5d  %4d  %8.1fs  %8.1fs  %8.1fs  %s  %11d",
                    rg_round, n_active, t_price, t_comm, t_master,
                    self._fmt_rc(global_max_rc[active_k].max()),
                    int(global_n_viol[active_k].sum()))
                if newly_converged:
                    self._log_converged_boots(newly_converged, theta_results,
                                              converged_info, param_idx)

        # ------ Collect non-converged boots ------
        remaining = np.where(active)[0]
        remaining_info = self._collect_remaining(remaining, theta_results, master,
                                                  global_max_rc, boot_iters, K, rank, comm)
        if self.verbose and comm.is_root() and len(remaining) > 0:
            param_idx = cfg.parameters_to_log or list(range(min(5, n_features)))
            logger.info(" ")
            logger.info(" WARNING: %d boots did not converge (max_iters=%d)", len(remaining), int(cfg.max_iters))
            self._log_converged_boots(remaining, theta_results, remaining_info, param_idx)

        return theta_results if rank == 0 else None

    def _collect_remaining(self, remaining, theta_results, master, global_max_rc,
                           boot_iters, K, rank, comm):
        """Collect theta and info from masters that hit max_iters."""
        n_features = self.dim.n_features
        info = {}
        for k in remaining:
            if rank == k:
                theta_results[k] = master['theta'].X.copy()
                info_k = np.array([master['model'].ObjVal, master['model'].NumConstrs,
                                   global_max_rc[k], boot_iters[k]], dtype=np.float64)
            if k == 0:
                if rank == 0:
                    info[k] = info_k
            else:
                if rank == k:
                    comm.comm.Send(np.ascontiguousarray(theta_results[k]), dest=0, tag=2*K + k)
                    comm.comm.Send(info_k, dest=0, tag=3*K + k)
                elif rank == 0:
                    theta_results[k] = np.empty(n_features, dtype=np.float64)
                    comm.comm.Recv(theta_results[k], source=k, tag=2*K + k)
                    info_k = np.empty(4, dtype=np.float64)
                    comm.comm.Recv(info_k, source=k, tag=3*K + k)
                    info[k] = info_k
        return info

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_rg_round_header(self, rg_round, param_indices):
        if rg_round % 80 != 0:
            return
        h1 = f" {'Round':>5}  {'Act.':>4}  {'Pricing':>9}  {'Comm':>9}  {'Master':>9}  {'Max Reduced':>14}  {'Total':>11}"
        h2 = f" {'':>5}  {'':>4}  {'(s)':>9}  {'(s)':>9}  {'(s)':>9}  {'Cost':>14}  {'#Viol':>11}"
        sep = "-" * len(h1)
        logger.info(sep)
        logger.info(h1)
        logger.info(h2)
        logger.info(sep)

    def _log_converged_boots(self, boots, theta_results, info_dict, param_idx):
        param_labels = ' '.join(f"{'θ['+str(i)+']':>10}" for i in param_idx)
        logger.info("       %s  %s  %s  %s  %s  %s",
                     'Boot'.rjust(6), '#Constr'.rjust(7), 'Reduced Cost'.rjust(14),
                     'Objective'.rjust(12), 'Range θ'.center(15), param_labels)
        for k in boots:
            theta_k = theta_results[k]
            info = info_dict.get(k)
            if info is None or theta_k is None:
                continue
            obj_val, n_constr, red_cost, _ = info
            param_vals = ' '.join(format_number(theta_k[i], width=10, precision=5) for i in param_idx)
            rng = f"[{theta_k.min():.1f}, {theta_k.max():.1f}]"
            logger.info("       ↳  %4d  %7d  %s  %s  %-15s  %s",
                         k, int(n_constr), self._fmt_rc(red_cost),
                         format_number(obj_val, width=12, precision=5), rng, param_vals)

    @staticmethod
    def _fmt_rc(val, width=14):
        if abs(val) < 1e-6 and val != 0:
            return f"{val:.5e}".rjust(width)
        return format_number(val, width=width, precision=6)

    # -------------------------------------------------------------------------
    # Phase 6: Statistics
    # -------------------------------------------------------------------------

    def _gather_and_compute_stats(self, K, theta_boots, total_time):
        if not self.comm_manager.is_root():
            return None
        stats = self.compute_bootstrap_stats(theta_boots, theta_hat=self.point_result.theta_hat)
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