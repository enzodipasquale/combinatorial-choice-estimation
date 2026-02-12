import os
import time
import numpy as np
import gurobipy as gp
from mpi4py import MPI
from bundlechoice.utils import get_logger, suppress_output, format_number
from bundlechoice.estimation.result import RowGenerationEstimationResult

logger = get_logger(__name__)


def _load_gurobi_model(lp_path, sol_path=None):
    with suppress_output():
        model = gp.read(lp_path)
        if sol_path and os.path.exists(sol_path):
            model.read(sol_path)
        model.setParam('OutputFlag', 0)
        model.optimize()
    return model


def _extract_master_vars(model, n_features, n_agents):
    vs = model.getVars()
    return gp.MVar.fromlist(vs[:n_features]), gp.MVar.fromlist(vs[n_features:n_features + n_agents])


class DistributedBootstrapMixin:

    def compute_distributed_bootstrap(self, num_bootstrap=100, seed=None, verbose=False,
                                      method='bayesian',
                                      row_gen_iteration_callback=None,
                                      row_gen_initialization_callback=None,
                                      bootstrap_callback=None,
                                      save_model_dir=None,
                                      load_model_dir=None):
        self.verbose = verbose
        self.row_gen = self.row_generation_manager
        comm = self.comm_manager
        K = num_bootstrap
        t0 = time.perf_counter()

        load_dir = comm.bcast(
            os.path.join(load_model_dir, "checkpoints") if load_model_dir else None)

        # === Phase 1: Point estimation ===
        self._phase_point_estimate(load_dir,
                                   row_gen_iteration_callback,
                                   row_gen_initialization_callback,
                                   verbose)

        save_dir = self._phase_save_point_estimate(save_model_dir)

        # === Phase 2: Broadcast base model ===
        base_data = self._extract_base_model()

        # === Phase 3: Bootstrap weights ===
        local_weights, boot_agent_weights = self._phase_bootstrap_weights(K, seed, method)

        # === Phase 4: Build or load master models ===
        master, pre_converged = self._phase_build_masters(
            K, base_data, local_weights, boot_agent_weights, load_dir)

        # === Phase 5: Row generation ===
        boot_save = os.path.join(save_dir, "bootstrap") if save_dir else None
        theta_boots = self._distributed_rg_loop(
            K, local_weights, master,
            iteration_callback=row_gen_iteration_callback,
            bootstrap_callback=bootstrap_callback,
            save_dir=boot_save,
            pre_converged=pre_converged)

        # === Phase 6: Statistics ===
        total_time = time.perf_counter() - t0
        return self._compute_and_log_stats(K, theta_boots, total_time)

    # -------------------------------------------------------------------------
    # Phase 1: Point estimation
    # -------------------------------------------------------------------------

    def _phase_point_estimate(self, load_dir, iteration_callback, init_callback, verbose):
        comm = self.comm_manager
        pt_dir = os.path.join(load_dir, "point_estimate") if load_dir else None

        load_ok = comm.bcast(
            pt_dir is not None and os.path.exists(os.path.join(pt_dir, "master.lp"))
            if comm.is_root() else None)

        if load_ok:
            self._load_point_estimate(pt_dir)
            self.subproblem_manager.initialize_subproblems()
        else:
            self.point_result = self.row_gen.solve(
                local_obs_weights=np.ones(self.data_manager.num_local_agent),
                initialize_master=True, initialize_subproblems=True,
                iteration_callback=iteration_callback,
                initialization_callback=init_callback,
                verbose=verbose)

    def _load_point_estimate(self, pt_dir):
        comm = self.comm_manager
        if comm.is_root():
            model = _load_gurobi_model(
                os.path.join(pt_dir, "master.lp"),
                os.path.join(pt_dir, "master.sol"))
            theta, u = _extract_master_vars(model, self.dim.n_features, self.dim.n_agents)
            self.row_gen.install_master_model(model, (theta, u))
            theta_hat = theta.X.copy()
        else:
            theta_hat = np.empty(self.dim.n_features, dtype=np.float64)

        comm.Bcast(theta_hat)
        self.point_result = RowGenerationEstimationResult(
            theta_hat=theta_hat, converged=True, num_iterations=0, final_objective=None)

        if self.verbose and comm.is_root():
            idx = self.config.row_generation.parameters_to_log \
                  or list(range(min(5, self.dim.n_features)))
            logger.info(" LOADED point estimate from %s", pt_dir)
            logger.info(" θ = [%s]", ', '.join(f'{theta_hat[i]:.5f}' for i in idx))

    def _phase_save_point_estimate(self, save_model_dir):
        comm = self.comm_manager
        save_dir = None
        if save_model_dir is not None:
            save_dir = os.path.join(save_model_dir, "checkpoints")
            if comm.is_root():
                pt_dir = os.path.join(save_dir, "point_estimate")
                os.makedirs(pt_dir, exist_ok=True)
                self.row_gen.master_model.write(os.path.join(pt_dir, "master.lp"))
                self.row_gen.master_model.write(os.path.join(pt_dir, "master.sol"))
        return comm.bcast(save_dir)

    # -------------------------------------------------------------------------
    # Phase 2: Extract base model
    # -------------------------------------------------------------------------

    def _extract_base_model(self):
        comm = self.comm_manager
        if comm.is_root():
            model = self.row_gen.master_model
            tv, uv = self.row_gen.master_variables
            data = {
                'A': model.getA().toarray(),
                'rhs': np.array(model.getAttr('RHS', model.getConstrs())),
                'sense': np.array([c.Sense for c in model.getConstrs()]),
                'theta_lb': np.array([tv[i].LB for i in range(self.dim.n_features)]),
                'theta_ub': np.array([tv[i].UB for i in range(self.dim.n_features)]),
            }
        else:
            data = {}
        return comm.bcast_dict(data)

    # -------------------------------------------------------------------------
    # Phase 3: Bootstrap weights
    # -------------------------------------------------------------------------

    def _phase_bootstrap_weights(self, K, seed, method):
        comm = self.comm_manager
        gen = self.generate_weights_bayesian_bootstrap if method == 'bayesian' \
              else self.generate_weights_standard_bootstrap
        weights_full = gen(seed, K)

        local_weights = comm.Scatterv_by_row(
            weights_full, row_counts=self.data_manager.agent_counts,
            dtype=np.float64, shape=(self.dim.n_agents, K))

        if comm.is_root():
            padded = np.zeros((comm.comm_size, self.dim.n_agents), dtype=np.float64)
            padded[:K] = weights_full.T[:K]
        else:
            padded = None
        boot_agent_weights = np.empty(self.dim.n_agents, dtype=np.float64)
        comm.comm.Scatter(padded, boot_agent_weights, root=0)

        return local_weights, boot_agent_weights

    # -------------------------------------------------------------------------
    # Phase 4: Build or load master models
    # -------------------------------------------------------------------------

    def _phase_build_masters(self, K, base_data, local_weights, boot_agent_weights, load_dir):
        comm = self.comm_manager
        rank = comm.rank

        local_features = self.oracles_manager.features_oracle(self.data_manager.local_obs_bundles)
        local_theta_obj = -local_weights.T @ local_features
        theta_obj_all = np.empty_like(local_theta_obj)
        comm.comm.Allreduce(local_theta_obj, theta_obj_all, op=MPI.SUM)

        if rank < K:
            boot_dir = os.path.join(load_dir, "bootstrap", f"boot_{rank:04d}") if load_dir else None
            if boot_dir and os.path.exists(os.path.join(boot_dir, "master.lp")):
                master = self._load_boot_master(boot_dir)
            else:
                master = self._build_master_model(base_data, theta_obj_all[rank], boot_agent_weights)
        else:
            master = None

        pre_converged = self._precheck_convergence(K, master)
        return master, pre_converged

    def _load_boot_master(self, boot_dir):
        model = _load_gurobi_model(
            os.path.join(boot_dir, "master.lp"),
            os.path.join(boot_dir, "master.sol"))
        theta, u = _extract_master_vars(model, self.dim.n_features, self.dim.n_agents)
        if self.verbose:
            logger.info(" Rank %d: loaded boot from %s", self.comm_manager.rank, boot_dir)
        return {'model': model, 'theta': theta, 'u': u}

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
            all_mvar = gp.MVar.fromlist(model.getVars())
            for s in np.unique(sense):
                mask = sense == s
                model.addMConstr(A[mask], all_mvar, s, rhs[mask])
            model.update()
            model.optimize()
        return {'model': model, 'theta': theta, 'u': u}

    def _precheck_convergence(self, K, master):
        comm = self.comm_manager
        rank = comm.rank
        nf = self.dim.n_features
        n_local = self.data_manager.num_local_agent
        local_start = int(self.data_manager.agent_counts[:rank].sum())
        cfg = self.config.row_generation

        theta_send = np.zeros((K, nf), dtype=np.float64)
        u_send = np.zeros((K, self.dim.n_agents), dtype=np.float64)
        if rank < K:
            theta_send[rank] = master['theta'].X
            u_send[rank] = master['u'].X
        theta_all = np.empty_like(theta_send)
        u_all = np.empty_like(u_send)
        comm.comm.Allreduce(theta_send, theta_all, op=MPI.SUM)
        comm.comm.Allreduce(u_send, u_all, op=MPI.SUM)

        local_max_rc = np.full(K, -np.inf, dtype=np.float64)
        local_ids = self.data_manager.local_agents_arange
        for k in range(K):
            bundles = self.subproblem_manager.solve_subproblems(theta_all[k])
            features = self.oracles_manager.features_oracle(bundles, local_ids)
            errors = self.oracles_manager.error_oracle(bundles, local_ids)
            u_sub = features @ theta_all[k] + errors
            u_master = u_all[k, local_start:local_start + n_local]
            local_max_rc[k] = (u_sub - u_master).max()

        global_max_rc = np.empty_like(local_max_rc)
        comm.comm.Allreduce(local_max_rc, global_max_rc, op=MPI.MAX)

        converged_mask = global_max_rc <= cfg.tolerance
        pre_converged = {}
        for k in range(K):
            if converged_mask[k]:
                pre_converged[k] = theta_all[k].copy()

        if self.verbose and self.comm_manager.is_root() and pre_converged:
            logger.info(" Pre-check: %d boots already converged", len(pre_converged))

        return pre_converged

    # -------------------------------------------------------------------------
    # Phase 5: Row generation loop
    # -------------------------------------------------------------------------

    def _distributed_rg_loop(self, K, local_weights, master,
                             iteration_callback=None, bootstrap_callback=None,
                             save_dir=None, pre_converged=None):
        rank = self.comm_manager.rank
        comm = self.comm_manager.comm
        cfg = self.config.row_generation
        P = self.comm_manager.comm_size
        nf = self.dim.n_features
        n_agents = self.dim.n_agents
        n_local = self.data_manager.num_local_agent
        local_start = int(self.data_manager.agent_counts[:rank].sum())
        local_ids = self.data_manager.local_agents_arange
        agent_ids = self.data_manager.agent_ids
        ROW_WIDTH = nf + 2

        active = np.ones(K, dtype=np.int32)
        theta_results = [None] * K

        if pre_converged:
            for k, theta_k in pre_converged.items():
                active[k] = 0
                theta_results[k] = theta_k

        master_send = np.zeros((K, nf + n_agents), dtype=np.float64)
        master_all = np.empty_like(master_send)

        if self.verbose and self.comm_manager.is_root():
            n_pre = len(pre_converged) if pre_converged else 0
            logger.info(" ")
            logger.info(" DISTRIBUTED BOOTSTRAP (%d samples, %d ranks%s)",
                        K, P, f", {n_pre} pre-converged" if n_pre else "")

        for rg_round in range(int(cfg.max_iters)):
            active_k = np.where(active)[0]
            if len(active_k) == 0:
                break

            if iteration_callback is not None:
                iteration_callback(rg_round, self.row_gen)

            # --- Step 1: Broadcast all (theta, u) pairs ---
            master_send[:] = 0.0
            if rank < K and active[rank]:
                master_send[rank, :nf] = master['theta'].X
                master_send[rank, nf:] = master['u'].X
            comm.Allreduce(master_send, master_all, op=MPI.SUM)
            thetas = master_all[:, :nf]
            us = master_all[:, nf:]

            # --- Step 2: Solve subproblems, compute reduced costs ---
            bundles = np.empty((len(active_k), n_local, self.dim.n_items), dtype=bool)
            for i, k in enumerate(active_k):
                bundles[i] = self.subproblem_manager.solve_subproblems(thetas[k])

            flat = bundles.reshape(-1, self.dim.n_items)
            flat_ids = np.tile(local_ids, len(active_k))
            features = self.oracles_manager.features_oracle(flat, flat_ids) \
                           .reshape(len(active_k), n_local, nf)
            errors = self.oracles_manager.error_oracle(flat, flat_ids) \
                         .reshape(len(active_k), n_local)

            u_sub = np.einsum('ijk,ik->ij', features, thetas[active_k]) + errors
            u_master = us[active_k, local_start:local_start + n_local]
            rc = local_weights[:, active_k].T * (u_sub - u_master)

            # --- Step 3: Convergence check ---
            local_max_rc = rc.max(axis=1)
            global_max_rc = np.empty_like(local_max_rc)
            comm.Allreduce(local_max_rc, global_max_rc, op=MPI.MAX)
            converged = global_max_rc <= cfg.tolerance

            # --- Step 4: Exchange violations for non-converged boots ---
            viol_mask = rc > cfg.tolerance
            rows_by_dest = {}
            for i, k in enumerate(active_k):
                if converged[i]:
                    continue
                v = np.where(viol_mask[i])[0]
                if len(v) > 0:
                    block = np.empty((len(v), ROW_WIDTH), dtype=np.float64)
                    block[:, 0] = agent_ids[v]
                    block[:, 1:-1] = features[i, v]
                    block[:, -1] = errors[i, v]
                    rows_by_dest[k] = block.ravel()

            recvbuf = self._alltoallv_rows(rows_by_dest, P, comm)

            # --- Step 5: Add cuts and re-solve ---
            if rank < K and active[rank] and not converged[active_k == rank][0]:
                if len(recvbuf) > 0:
                    d = recvbuf.reshape(-1, ROW_WIDTH)
                    ids = d[:, 0].astype(np.int64)
                    master['model'].addConstr(
                        master['u'][ids] >= d[:, 1:-1] @ master['theta'] + d[:, -1])
                master['model'].optimize()

            # --- Step 6: Retire converged boots ---
            newly_converged = []
            for i, k in enumerate(active_k):
                if converged[i] and rg_round >= cfg.min_iters:
                    active[k] = 0
                    newly_converged.append(k)
                    if rank == 0:
                        theta_results[k] = thetas[k].copy()
                    if rank == k and save_dir is not None:
                        d = os.path.join(save_dir, f"boot_{k:04d}")
                        os.makedirs(d, exist_ok=True)
                        master['model'].write(os.path.join(d, "master.lp"))
                        master['model'].write(os.path.join(d, "master.sol"))

            # --- Logging ---
            if self.verbose and self.comm_manager.is_root():
                self._log_rg_round(rg_round, len(active_k), global_max_rc,
                                   newly_converged, theta_results)

            if bootstrap_callback is not None:
                bootstrap_callback(rg_round, active, theta_results)

        # --- Collect remaining non-converged ---
        remaining = np.where(active)[0]
        if len(remaining) > 0:
            buf = np.zeros((K, nf), dtype=np.float64)
            if rank < K and active[rank]:
                buf[rank] = master['theta'].X
                if save_dir is not None:
                    d = os.path.join(save_dir, f"boot_{rank:04d}")
                    os.makedirs(d, exist_ok=True)
                    master['model'].write(os.path.join(d, "master.lp"))
                    master['model'].write(os.path.join(d, "master.sol"))
            recv = np.empty_like(buf) if rank == 0 else None
            comm.Reduce(buf, recv, op=MPI.SUM, root=0)
            if rank == 0:
                for k in remaining:
                    theta_results[k] = recv[k]

            if self.verbose and self.comm_manager.is_root():
                logger.info(" ")
                logger.info(" WARNING: %d boots did not converge (max_iters=%d)",
                            len(remaining), int(cfg.max_iters))

        return theta_results if rank == 0 else None

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _alltoallv_rows(rows_by_dest, P, comm):
        send_counts = np.zeros(P, dtype=np.int64)
        for dest, data in rows_by_dest.items():
            send_counts[dest] = len(data)

        recv_counts = np.empty(P, dtype=np.int64)
        comm.Alltoall(send_counts, recv_counts)

        sdispls = np.zeros(P, dtype=np.int64)
        rdispls = np.zeros(P, dtype=np.int64)
        np.cumsum(send_counts[:-1], out=sdispls[1:])
        np.cumsum(recv_counts[:-1], out=rdispls[1:])

        sendbuf = np.concatenate([rows_by_dest[k] for k in sorted(rows_by_dest)]) \
                  if rows_by_dest else np.empty(0, dtype=np.float64)
        recvbuf = np.empty(int(recv_counts.sum()), dtype=np.float64)

        comm.Alltoallv(
            [sendbuf, send_counts, sdispls, MPI.DOUBLE],
            [recvbuf, recv_counts, rdispls, MPI.DOUBLE])
        return recvbuf

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_rg_round(self, rg_round, n_active, global_max_rc, newly_converged, theta_results):
        param_idx = self.config.row_generation.parameters_to_log \
                    or list(range(min(5, self.dim.n_features)))
        if rg_round % 80 == 0:
            h1 = f" {'Round':>5}  {'Active':>6}  {'Max RC':>14}"
            logger.info("-" * len(h1))
            logger.info(h1)
            logger.info("-" * len(h1))
        logger.info(" %5d  %6d  %s", rg_round, n_active,
                     self._fmt_rc(global_max_rc.max()))
        if newly_converged:
            for k in newly_converged:
                t = theta_results[k]
                if t is not None:
                    vals = ' '.join(format_number(t[i], width=10, precision=5) for i in param_idx)
                    logger.info("   ↳ boot %d converged: [%s]", k, vals)

    @staticmethod
    def _fmt_rc(val, width=14):
        if abs(val) < 1e-6 and val != 0:
            return f"{val:.5e}".rjust(width)
        return format_number(val, width=width, precision=6)

    # -------------------------------------------------------------------------
    # Phase 6: Statistics
    # -------------------------------------------------------------------------

    def _compute_and_log_stats(self, K, theta_boots, total_time):
        if not self.comm_manager.is_root():
            return None
        stats = self.compute_bootstrap_stats(theta_boots, theta_hat=self.point_result.theta_hat)
        if self.verbose:
            theta_hat = self.point_result.theta_hat
            idx = self.config.row_generation.parameters_to_log \
                  or list(range(min(5, self.dim.n_features)))
            logger.info(" ")
            logger.info("-" * 70)
            logger.info(" DISTRIBUTED BOOTSTRAP: %d samples in %.1fs", K, total_time)
            logger.info("-" * 70)
            logger.info(f"{'Param':>8} | {'Point Est':>12} | {'Boot Mean':>12} | {'SE':>12} | {'t-stat':>10}")
            logger.info("-" * 70)
            for i in idx:
                logger.info(f"  θ[{i:>3}] | {theta_hat[i]:>12.5f} | {stats.mean[i]:>12.5f} | "
                            f"{stats.se[i]:>12.5f} | {stats.t_stats[i]:>10.2f}")
            logger.info("-" * 70)
        return stats