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

        # === Phase 4: Compute objectives (collective) and build master models (local) ===
        local_features = self.oracles_manager.features_oracle(self.data_manager.local_obs_bundles)
        master = self._setup_all_masters(K, base_data, local_weights, local_features)

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

    def _setup_all_masters(self, K, base_data, local_weights, local_features):
        """All ranks participate in collective ops; only rank k builds model k."""
        comm = self.comm_manager
        agent_counts = self.data_manager.agent_counts
        master = None

        for k in range(K):
            # Collective: all ranks must participate
            local_theta_obj = np.ascontiguousarray(
                (-local_weights[:, k][:, None] * local_features).sum(0))
            theta_obj_coef = np.empty_like(local_theta_obj)
            comm.comm.Reduce(local_theta_obj, theta_obj_coef, op=MPI.SUM, root=k)

            u_obj_weights = comm.Gatherv_by_row(
                local_weights[:, k], row_counts=agent_counts, root=k)

            if comm.rank == k:
                master = self._build_master_model(
                    base_data, theta_obj_coef, u_obj_weights)

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

        active = np.ones(K, dtype=np.int32)
        boot_iters = np.zeros(K, dtype=np.int32)
        consec_no_viol = np.zeros(K, dtype=np.int32)
        theta_results = [None] * K

        if self.verbose and comm.is_root():
            logger.info(" ")
            logger.info(" DISTRIBUTED BOOTSTRAP (%d samples, %d ranks)", K, comm.comm_size)

        rg_round = 0
        while rg_round < cfg.max_iters:
            n_active = int(active.sum())
            if n_active == 0:
                break

            active_masters = np.where(active)[0]

            # --- Step 1: Pricing (collective, sequential over k) ---
            t_price = time.perf_counter()
            local_viols = {}
            for k in active_masters:
                if rank == k:
                    theta_k = np.ascontiguousarray(master['theta'].X)
                    u_k = np.ascontiguousarray(master['u'].X)
                else:
                    theta_k = np.empty(n_features, dtype=np.float64)
                    u_k = np.empty(n_agents, dtype=np.float64)

                theta_k = comm.Bcast(theta_k, root=k)
                u_local_k = comm.Scatterv_by_row(u_k, row_counts=agent_counts,
                                                  dtype=np.float64, shape=(n_agents,), root=k)
                bundles = self.subproblem_manager.solve_subproblems(theta_k)
                local_viols[k] = self.row_gen._compute_local_violations(
                    bundles, theta_k, u_local_k, local_weights[:, k])
            t_price = time.perf_counter() - t_price

            # --- Step 2: Gather violations (collective, sequential over k) ---
            t_comm = time.perf_counter()
            convergence_flags = {}
            gathered_data = {}

            for k in active_masters:
                max_rc, n_viol, viol_ids, viol_bun, viol_fe = local_viols[k]
                local_meta = np.array([max_rc, n_viol], dtype=np.float64)
                all_meta = comm.Allgather(local_meta).reshape(-1, 2)
                global_max_rc = all_meta[:, 0].max()
                viol_counts = all_meta[:, 1].astype(np.int64)
                no_violations = global_max_rc <= cfg.tolerance

                if no_violations:
                    consec_no_viol[k] += 1
                    convergence_flags[k] = True
                else:
                    consec_no_viol[k] = 0
                    convergence_flags[k] = False
                    g_bun = comm.Gatherv_by_row(viol_bun, row_counts=viol_counts, root=k)
                    g_fe = comm.Gatherv_by_row(viol_fe, row_counts=viol_counts, root=k)
                    g_ids = comm.Gatherv_by_row(viol_ids, row_counts=viol_counts, root=k)
                    if rank == k:
                        gathered_data[k] = (g_ids, g_fe[:, :-1], g_fe[:, -1])
            t_comm = time.perf_counter() - t_comm

            # --- Step 3: All masters solve in parallel (no communication) ---
            t_master = time.perf_counter()
            if rank < K and active[rank] and not convergence_flags[rank]:
                ids, features, errors = gathered_data[rank]
                master['model'].addConstr(
                    master['u'][ids] >= features @ master['theta'] + errors)
                master['model'].optimize()
            t_master = time.perf_counter() - t_master

            # --- Step 4: Update bookkeeping ---
            newly_converged = []
            for k in active_masters:
                boot_iters[k] += 1
                if convergence_flags[k] and boot_iters[k] >= cfg.min_iters:
                    newly_converged.append(k)

            if iteration_callback is not None:
                iteration_callback(int(boot_iters[active_masters[0]]), self.row_gen)

            for k in newly_converged:
                active[k] = 0
                if rank == k:
                    theta_results[k] = master['theta'].X.copy()
                if k != 0:
                    if rank == k:
                        comm.comm.Send(np.ascontiguousarray(theta_results[k]), dest=0, tag=k)
                    elif rank == 0:
                        theta_results[k] = np.empty(n_features, dtype=np.float64)
                        comm.comm.Recv(theta_results[k], source=k, tag=k)

            if self.verbose and comm.is_root():
                logger.info("Round %4d | active %4d | pricing %.1fs | comm %.1fs | master %.1fs | converged %d",
                            rg_round, int(active.sum()), t_price, t_comm, t_master,
                            len(newly_converged))

            rg_round += 1

        # Collect remaining active masters (hit max_iters)
        for k in np.where(active)[0]:
            if rank == k:
                theta_results[k] = master['theta'].X.copy()
            if k != 0:
                if rank == k:
                    comm.comm.Send(np.ascontiguousarray(theta_results[k]), dest=0, tag=K + k)
                elif rank == 0:
                    theta_results[k] = np.empty(n_features, dtype=np.float64)
                    comm.comm.Recv(theta_results[k], source=k, tag=K + k)

        return theta_results if rank == 0 else None

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
        idx = self.config.row_generation.parameters_to_log or list(range(min(5, self.dim.n_features)))
        logger.info(" ")
        logger.info("-" * 55)
        logger.info(" DISTRIBUTED BOOTSTRAP: %d samples in %.1fs", n_bootstrap, total_time)
        logger.info("-" * 55)
        logger.info(f"{'Param':>6} | {'Mean':>12} | {'SE':>12} | {'t-stat':>10}")
        logger.info("-" * 55)
        for i in idx:
            logger.info(f"θ[{i:>3}] | {result.mean[i]:>12.5f} | {result.se[i]:>12.5f} | {result.t_stats[i]:>10.2f}")
        logger.info("-" * 55)