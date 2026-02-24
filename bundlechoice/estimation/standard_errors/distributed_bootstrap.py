import os
import time
import json
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


# =========================================================================
# BootstrapMaster: Gurobi model lifecycle for one bootstrap sample
# =========================================================================

class BootstrapMaster:
    def __init__(self, model, theta, u):
        self.model, self.theta, self.u = model, theta, u

    @classmethod
    def build(cls, base_data, theta_obj, u_weights, grb_params=None):
        A, rhs, sense = base_data['A'], base_data['rhs'], base_data['sense']
        with suppress_output():
            model = gp.Model()
            params = {"Method": 0, "LPWarmStart": 2, "OutputFlag": 0}
            params.update(grb_params or {})
            for p, v in params.items():
                if v is not None:
                    model.setParam(p, v)
            nf = len(theta_obj)
            theta = model.addMVar(nf, obj=theta_obj,
                                  lb=base_data['theta_lb'], ub=base_data['theta_ub'],
                                  name='parameter')
            u = model.addMVar(len(u_weights), lb=0, obj=u_weights, name='utility')
            model.update()
            all_mvar = gp.MVar.fromlist(model.getVars())
            for s in np.unique(sense):
                mask = sense == s
                model.addMConstr(A[mask], all_mvar, s, rhs[mask])
            model.update()
            model.optimize()
        return cls(model, theta, u)

    @classmethod
    def load(cls, path, n_features, n_agents):
        model = _load_gurobi_model(os.path.join(path, "master.lp"),
                                   os.path.join(path, "master.sol"))
        theta, u = _extract_master_vars(model, n_features, n_agents)
        meta_path = os.path.join(path, "meta.json")
        converged = False
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                converged = json.load(f).get('converged', False)
        return cls(model, theta, u), converged

    def save(self, path, converged):
        os.makedirs(path, exist_ok=True)
        self.model.write(os.path.join(path, "master.lp"))
        self.model.write(os.path.join(path, "master.sol"))
        with open(os.path.join(path, "meta.json"), 'w') as f:
            json.dump({'converged': converged}, f)

    def add_cuts(self, rows):
        """Add violation rows packed as (agent_id, features..., error) per row."""
        if len(rows) == 0:
            return
        ids = rows[:, 0].astype(np.int64)
        self.model.addConstr(self.u[ids] >= rows[:, 1:-1] @ self.theta + rows[:, -1])


    def strip_slack_constraints(self, percentile=100, hard_threshold = float('inf')):
        constrs = self.model.getConstrs()
        if not constrs:
            return 0
        slacks = np.array([c.Slack for c in constrs], dtype=float)
        threshold = np.percentile(slacks, 100.0 - percentile)
        below_threshold = slacks < threshold
        n_below = below_threshold.sum()
        if n_below < len(slacks) - hard_threshold:
            return self.strip_constraints_hard_threshold(hard_threshold)
        to_remove_id = np.where(below_threshold)[0]
        to_remove = [constrs[i] for i in to_remove_id]
        self.model.remove(to_remove)
        return len(to_remove_id)

    def strip_constraints_hard_threshold(self, n_constraints = float('inf')):
        if self.model.NumConstrs < n_constraints:
            return 0
        constrs = self.model.getConstrs()
        if not constrs:
            return 0
        slacks = np.array([c.Slack for c in constrs], dtype=float)
        sorted_indices = np.argsort(slacks)
        to_remove_id = sorted_indices[:-int(n_constraints)]
        to_remove = [constrs[i] for i in to_remove_id]
        self.model.remove(to_remove)
        return len(to_remove_id)


# =========================================================================
# BootstrapState: distributed K-sample state
# =========================================================================

class BootstrapState:
    def __init__(self, num_bootstrap, dim, data_manager, comm_manager):
        assignment = comm_manager.spread_tasks_across_nodes(num_bootstrap)
        self.boot_id = int(assignment.my_tasks[0]) if assignment.has_tasks else None
        self.task_to_rank = assignment.task_to_rank
        self.num_bootstrap = num_bootstrap
        self.dim = dim
        self.data_manager = data_manager

        nf, na = dim.n_features, dim.n_agents
        self.active = np.ones(num_bootstrap, dtype=np.int32)
        self.theta_vals = np.zeros((num_bootstrap, nf), dtype=np.float64)
        self.u_vals = np.zeros((num_bootstrap, na), dtype=np.float64)
        self.stats = np.zeros((num_bootstrap, 2), dtype=np.float64)
        self.max_rc = np.zeros(num_bootstrap, dtype=np.float64)
        self.converged = np.zeros(num_bootstrap, dtype=bool)
        self._vars_buf = np.zeros((num_bootstrap, nf + na), dtype=np.float64)
        self._stats_buf = np.zeros((num_bootstrap, 2), dtype=np.float64)
        self._comm = comm_manager.comm

    @property
    def active_indices(self):
        return np.where(self.active)[0]

    @property
    def num_active(self):
        return int(self.active.sum())

    @property
    def has_active_boot(self):
        return self.boot_id is not None and self.active[self.boot_id]

    def tile_local_ids(self):
        return np.tile(self.data_manager.local_agents_arange, self.num_active)

    def local_u_master(self):
        return self.u_vals[self.active_indices, self.data_manager.local_agent_slice]

    def allreduce_by_boot(self, buf, data=None, op=MPI.SUM):
        buf[:] = 0.0
        if data is not None and self.has_active_boot:
            buf[self.boot_id] = data
        out = np.zeros_like(buf)
        self._comm.Allreduce(buf, out, op=op)
        return out

    def sync_vars(self, master):
        nf = self.theta_vals.shape[1]
        data = np.concatenate([master.theta.X, master.u.X]) if self.has_active_boot else None
        result = self.allreduce_by_boot(self._vars_buf, data)
        active = self.active.astype(bool)
        self.theta_vals[active] = result[active, :nf]
        self.u_vals[active] = result[active, nf:]

    def sync_stats(self, master):
        data = [master.model.NumConstrs, master.model.ObjVal] if self.has_active_boot else None
        result = self.allreduce_by_boot(self._stats_buf, data)
        active = self.active.astype(bool)
        self.stats[active] = result[active]

    def retire(self, is_converged, skip_if):
        if skip_if:
            return np.array([], dtype=np.int64)
        to_retire = self.active_indices[is_converged[self.active_indices]]
        self.active[to_retire] = 0
        return to_retire

    def initialize(self, master, is_converged):
        conv_buf = np.zeros(self.num_bootstrap, dtype=np.int32)
        if self.boot_id is not None and is_converged:
            conv_buf[self.boot_id] = 1
        conv_all = np.zeros_like(conv_buf)
        self._comm.Allreduce(conv_buf, conv_all, op=MPI.SUM)
        self.active[:] = 1 - conv_all
        self._vars_buf[:] = 0.0
        if self.boot_id is not None:
            nf = self.theta_vals.shape[1]
            self._vars_buf[self.boot_id, :nf] = master.theta.X
            self._vars_buf[self.boot_id, nf:] = master.u.X
        result = np.zeros_like(self._vars_buf)
        self._comm.Allreduce(self._vars_buf, result, op=MPI.SUM)
        nf = self.theta_vals.shape[1]
        self.theta_vals[:] = result[:, :nf]
        self.u_vals[:] = result[:, nf:]


# =========================================================================
# Mixin
# =========================================================================

class DistributedBootstrapMixin:

    def compute_distributed_bootstrap(self, num_bootstrap=100, seed=None, verbose=False,
                                      method='bayesian',
                                      pt_estimate_callbacks=(None, None),
                                      bootstrap_callback=None,
                                      save_model_dir=None,
                                      load_model_dir=None):
        if num_bootstrap > self.comm_manager.comm_size:
            raise ValueError(
                f"num_bootstrap ({num_bootstrap}) > comm_size ({self.comm_manager.comm_size}). "
                f"Distributed bootstrap requires at most one sample per rank.")

        self.verbose = verbose
        t0 = time.perf_counter()

        state = BootstrapState(num_bootstrap, self.dim,
                               self.data_manager, self.comm_manager)

        load_dir = self.comm_manager.bcast(
            os.path.join(load_model_dir, "checkpoints") if load_model_dir else None)

        # === Phase 1: Point estimation ===
        self._phase_point_estimate(load_dir, pt_estimate_callbacks)
        save_dir = self._phase_save_point_estimate(save_model_dir)

        # === Phase 2: Broadcast base model ===
        base_data = self._extract_base_model()

        # === Phase 3: Bootstrap weights ===
        self._local_weights, boot_agent_weights = self._phase_bootstrap_weights(
            num_bootstrap, seed, method, state)

        # === Phase 4: Build or load master models ===
        master, is_converged = self._phase_build_masters(
            base_data, boot_agent_weights, load_dir, state)

        # === Phase 5: Row generation ===
        self._save_dir = os.path.join(save_dir, "bootstrap") if save_dir else None
        self._distributed_rg_loop(state, master, is_converged, bootstrap_callback)

        # === Phase 6: Statistics ===
        return self._compute_and_log_stats(state, time.perf_counter() - t0)

    # -------------------------------------------------------------------------
    # Phase 1: Point estimation
    # -------------------------------------------------------------------------

    def _phase_point_estimate(self, load_dir, pt_estimate_callbacks):
        initialization_callback, iteration_callback = pt_estimate_callbacks
        pt_dir = os.path.join(load_dir, "point_estimate") if load_dir else None
        load_ok = self.comm_manager.bcast(
            pt_dir is not None and os.path.exists(os.path.join(pt_dir, "master.lp"))
            if self.comm_manager.is_root() else None)

        init_master = not load_ok
        if load_ok:
            converged = self._load_point_estimate(pt_dir)
            self.subproblem_manager.initialize_subproblems()
            if converged:
                return

        self.point_result = self.row_generation_manager.solve(
            local_obs_weights=np.ones(self.data_manager.num_local_agent),
            initialize_master=init_master, initialize_subproblems=init_master,
            iteration_callback=iteration_callback,
            initialization_callback=initialization_callback,
            verbose=self.verbose)

    def _load_point_estimate(self, pt_dir):
        converged = False
        if self.comm_manager.is_root():
            model = _load_gurobi_model(
                os.path.join(pt_dir, "master.lp"),
                os.path.join(pt_dir, "master.sol"))
            theta, u = _extract_master_vars(model, self.dim.n_features, self.dim.n_agents)
            self.row_generation_manager.install_master_model(model, (theta, u))
            theta_hat = theta.X.copy()
            meta_path = os.path.join(pt_dir, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    converged = json.load(f).get('converged', False)
        else:
            theta_hat = np.zeros(self.dim.n_features, dtype=np.float64)

        converged = self.comm_manager.bcast(converged)
        self.comm_manager.Bcast(theta_hat)

        if converged:
            self.point_result = RowGenerationEstimationResult(
                theta_hat=theta_hat, converged=True, num_iterations=0, final_objective=None)

        if self.verbose and self.comm_manager.is_root():
            idx = self.config.standard_errors.parameters_to_log \
                  or list(range(min(5, self.dim.n_features)))
            logger.info(" LOADED point estimate from %s (%s)",
                        pt_dir, "converged" if converged else "not converged")
            logger.info(" θ = [%s]", ', '.join(f'{theta_hat[i]:.5f}' for i in idx))

        return converged

    def _phase_save_point_estimate(self, save_model_dir):
        save_dir = None
        if save_model_dir is not None:
            save_dir = os.path.join(save_model_dir, "checkpoints")
            if self.comm_manager.is_root():
                pt_dir = os.path.join(save_dir, "point_estimate")
                os.makedirs(pt_dir, exist_ok=True)
                self.row_generation_manager.master_model.write(os.path.join(pt_dir, "master.lp"))
                self.row_generation_manager.master_model.write(os.path.join(pt_dir, "master.sol"))
                with open(os.path.join(pt_dir, "meta.json"), 'w') as f:
                    json.dump({'converged': bool(self.point_result.converged)}, f)
        return self.comm_manager.bcast(save_dir)

    # -------------------------------------------------------------------------
    # Phase 2: Extract base model
    # -------------------------------------------------------------------------

    def _extract_base_model(self):
        if self.comm_manager.is_root():
            model = self.row_generation_manager.master_model
            theta, _ = self.row_generation_manager.master_variables
            data = {
                'A': model.getA().toarray(),
                'rhs': np.array(model.getAttr('RHS', model.getConstrs())),
                'sense': np.array([c.Sense for c in model.getConstrs()]),
                'theta_lb': np.array([theta[i].LB for i in range(self.dim.n_features)]),
                'theta_ub': np.array([theta[i].UB for i in range(self.dim.n_features)]),
            }
        else:
            data = {}
        data = self.comm_manager.bcast_dict(data)

        # Override bounds if SE config specifies its own
        se_bounds = self.config.standard_errors.theta_bounds
        if se_bounds:
            data['theta_lb'], data['theta_ub'] = self.config.standard_errors.theta_bounds_arrays(
                self.dim.n_features)

        return data

    # -------------------------------------------------------------------------
    # Phase 3: Bootstrap weights
    # -------------------------------------------------------------------------

    def _phase_bootstrap_weights(self, num_bootstrap, seed, method, state):
        gen = self.generate_weights_bayesian_bootstrap if method == 'bayesian' \
              else self.generate_weights_standard_bootstrap
        weights_full = gen(seed, num_bootstrap)

        local_weights = self.comm_manager.Scatterv_by_row(
            weights_full, row_counts=self.data_manager.agent_counts,
            dtype=np.float64, shape=(self.dim.n_agents, num_bootstrap))

        boot_agent_weights = np.zeros(self.dim.n_agents, dtype=np.float64)
        if self.comm_manager.is_root():
            for k in range(num_bootstrap):
                dest = int(state.task_to_rank[k])
                if dest == 0:
                    boot_agent_weights[:] = weights_full[:, k]
                else:
                    self.comm_manager.comm.Send(
                        np.ascontiguousarray(weights_full[:, k]), dest=dest, tag=k)
        elif state.boot_id is not None:
            self.comm_manager.comm.Recv(boot_agent_weights, source=0, tag=state.boot_id)

        return local_weights, boot_agent_weights

    # -------------------------------------------------------------------------
    # Phase 4: Build or load master models
    # -------------------------------------------------------------------------

    def _phase_build_masters(self, base_data, boot_agent_weights, load_dir, state):
        local_features = self.oracles_manager.features_oracle(self.data_manager.local_obs_bundles)
        local_theta_obj = -self._local_weights.T @ local_features
        theta_obj_all = np.zeros_like(local_theta_obj)
        self.comm_manager.comm.Allreduce(local_theta_obj, theta_obj_all, op=MPI.SUM)

        master = None
        is_converged = False

        if state.boot_id is not None:
            boot_dir = (os.path.join(load_dir, "bootstrap", f"boot_{state.boot_id:04d}")
                        if load_dir else None)
            if boot_dir and os.path.exists(os.path.join(boot_dir, "master.lp")):
                master, is_converged = BootstrapMaster.load(
                    boot_dir, self.dim.n_features, self.dim.n_agents)
                if self.verbose:
                    logger.info(" Rank %d: loaded boot %d from %s (%s)",
                                self.comm_manager.rank, state.boot_id, boot_dir,
                                "converged" if is_converged else "not converged")
            else:
                master = BootstrapMaster.build(
                    base_data, theta_obj_all[state.boot_id], boot_agent_weights,
                    self.config.standard_errors.master_GRB_Params)

        return master, is_converged

    # -------------------------------------------------------------------------
    # Phase 5: Row generation loop
    # -------------------------------------------------------------------------

    def _pricing(self, state):
        active = state.active_indices
        n_active = len(active)
        n_local = state.data_manager.num_local_agent

        bundles = np.empty((n_active, n_local, state.dim.n_items), dtype=bool)
        for b, boot in enumerate(active):
            bundles[b] = self.subproblem_manager.solve_subproblems(state.theta_vals[boot])

        features, errors = self.oracles_manager.features_and_errors_oracle(
            bundles.reshape(-1, state.dim.n_items), state.tile_local_ids())
        features = features.reshape(n_active, n_local, state.dim.n_features)
        errors = errors.reshape(n_active, n_local)

        u_cuts = np.einsum('bif,bf->bi', features, state.theta_vals[active]) + errors
        reduced_costs = self._local_weights[:, active].T * (u_cuts - state.local_u_master())

        cut_rows = np.concatenate([
            np.tile(state.data_manager.agent_ids, (n_active, 1))[:, :, None],
            features, errors[:, :, None]], axis=-1)

        return cut_rows, reduced_costs

    def _distributed_rg_loop(self, state, master, is_converged, bootstrap_callback):
        state.initialize(master, is_converged)

        if self.verbose and self.comm_manager.is_root():
            n_pre = state.num_bootstrap - state.num_active
            logger.info(" ")
            logger.info(" DISTRIBUTED BOOTSTRAP (%d samples, %d ranks%s)",
                        state.num_bootstrap, self.comm_manager.comm_size,
                        f", {n_pre} pre-converged" if n_pre else "")

        se_cfg = self.config.standard_errors
        tol = se_cfg.rowgen_tol

        for rg_round in range(int(se_cfg.rowgen_max_iters)):
            t_round = time.perf_counter()

            if state.num_active == 0:
                break

            if bootstrap_callback is not None:
                bootstrap_callback(rg_round, self, master)

            # --- Step 1: Broadcast master vars ---
            state.sync_vars(master)

            # --- Step 2: Pricing ---
            t_price = time.perf_counter()
            cut_rows, reduced_costs = self._pricing(state)
            self.comm_manager.comm.Barrier()
            t_price = time.perf_counter() - t_price

            # --- Step 3: Convergence check ---
            local_max_rc = reduced_costs.max(axis=1)
            global_max_rc = np.zeros(state.num_active)
            self.comm_manager.comm.Allreduce(local_max_rc, global_max_rc, op=MPI.MAX)
            state.max_rc[state.active_indices] = global_max_rc

            converged_mask = np.zeros(state.num_bootstrap, dtype=bool)
            converged_mask[state.active_indices] = global_max_rc <= tol

            # --- Step 4: Pack violations and exchange ---
            viol_mask = reduced_costs > tol
            total_viols_arr = np.array([viol_mask.sum()])
            self.comm_manager.comm.Allreduce(MPI.IN_PLACE, total_viols_arr, op=MPI.SUM)

            rows_by_dest = {}
            for b, boot in enumerate(state.active_indices):
                if converged_mask[boot]:
                    continue
                v = np.where(viol_mask[b])[0]
                if len(v) > 0:
                    rows_by_dest[int(state.task_to_rank[boot])] = cut_rows[b, v].ravel()

            recvbuf = self.comm_manager.alltoallv_rows(rows_by_dest)

            # --- Step 5: Add cuts, re-solve, retire ---
            t_master = time.perf_counter()
            if state.has_active_boot and not converged_mask[state.boot_id]:
                master.add_cuts(recvbuf.reshape(-1, state.dim.n_features + 2))
                master.model.optimize()

            self.comm_manager.comm.Barrier()
            t_master = time.perf_counter() - t_master

            state.sync_stats(master)
            retired_boot_id = state.retire(converged_mask, skip_if=rg_round < se_cfg.rowgen_min_iters)

            if state.boot_id in retired_boot_id and self._save_dir is not None:
                master.save(os.path.join(self._save_dir, f"boot_{state.boot_id:04d}"), converged=True)

            # --- Logging ---
            t_total = time.perf_counter() - t_round
            if self.verbose and self.comm_manager.is_root():
                self._log_rg_round(rg_round, state, {
                    'n_active': state.num_active + len(retired_boot_id),
                    'max_rc': global_max_rc.max(),
                    't_price': t_price, 't_comm': max(0.0, t_total - t_price - t_master),
                    't_master': t_master, 'n_viols': int(total_viols_arr[0]),
                    'retired_boot_id': retired_boot_id})

        # --- Collect remaining non-converged ---
        remaining = state.active_indices
        state.converged = (state.active == 0)
        if len(remaining) > 0:
            state.sync_stats(master)
            state.sync_vars(master)
            state.active[remaining] = 0

            if (state.boot_id is not None and state.boot_id in remaining and self._save_dir is not None):
                master.save(os.path.join(self._save_dir, f"boot_{state.boot_id:04d}"), converged=False)

            if self.verbose and self.comm_manager.is_root():
                logger.info(" ")
                logger.info(" WARNING: %d boots did not converge (max_iters=%d)",
                            len(remaining), int(se_cfg.rowgen_max_iters))
                self._log_boot_details(remaining, state)

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_rg_round(self, rg_round, state, s):
        if rg_round % 40 == 0:
            logger.info("-" * 75)
            logger.info(" %5s  %4s  %9s  %9s  %9s  %14s  %9s",
                        "Round", "Act.", "Pricing", "Comm", "Master",
                        "Max Reduced", "Total")
            logger.info(" %5s  %4s  %9s  %9s  %9s  %14s  %9s",
                        "", "", "(s)", "(s)", "(s)", "Cost", "#Viol")
            logger.info("-" * 75)
        logger.info(" %5d  %4d  %8.1fs  %8.1fs  %8.1fs  %14s  %9d",
                     rg_round, s['n_active'],
                     s['t_price'], s['t_comm'], s['t_master'],
                     self._fmt_rc(s['max_rc']),
                     s['n_viols'])
        if len(s['retired_boot_id']) > 0:
            self._log_boot_details(s['retired_boot_id'], state)

    def _log_boot_details(self, boot_ids, state):
        param_idx = self.config.standard_errors.parameters_to_log \
                    or list(range(min(5, self.dim.n_features)))
        range_idx = [i for i in range(self.dim.n_features) if i not in param_idx]
        hdr_params = ''.join(f"{'θ['+str(i)+']':>12}" for i in param_idx)
        logger.info("        %4s  %7s  %14s  %14s  %15s  %s",
                    "Boot", "#Constr", "Reduced Cost", "Objective", "Range θ", hdr_params)
        for k in boot_ids:
            t = state.theta_vals[k]
            t_range = f"[{t[range_idx].min():.1f}, {t[range_idx].max():.1f}]" if range_idx else f"[{t.min():.1f}, {t.max():.1f}]"
            vals = ''.join(format_number(t[i], width=12, precision=5) for i in param_idx)
            logger.info("        ↳ %4d  %7d  %14s  %14s  %15s  %s",
                        k, int(state.stats[k, 0]),
                        self._fmt_rc(state.max_rc[k]),
                        format_number(state.stats[k, 1], width=14, precision=4),
                        t_range, vals)

    @staticmethod
    def _fmt_rc(val, width=14):
        if abs(val) < 1e-6 and val != 0:
            return f"{val:.5e}".rjust(width)
        return format_number(val, width=width, precision=6)

    # -------------------------------------------------------------------------
    # Phase 6: Statistics
    # -------------------------------------------------------------------------

    def _compute_and_log_stats(self, state, total_time):
        if not self.comm_manager.is_root():
            return None
        theta_vals = state.theta_vals
        converged_mask = state.converged
        n_converged = int(converged_mask.sum())
        n_non_converged =  state.num_bootstrap - n_converged
        theta_for_stats = theta_vals if n_converged == 0 else theta_vals[converged_mask]
        
        stats = self.compute_bootstrap_stats(theta_for_stats,
                                             theta_hat=self.point_result.theta_hat)
        if self.verbose:
            theta_hat = self.point_result.theta_hat
            idx = self.config.standard_errors.parameters_to_log \
                  or list(range(min(5, self.dim.n_features)))
            logger.info(" ")
            if n_non_converged > 0:
                logger.info(" WARNING: %d of %d bootstrap samples did not converge",
                            n_non_converged, state.num_bootstrap)
            logger.info("-" * 70)
            logger.info(" DISTRIBUTED BOOTSTRAP: %d samples in %.1fs",
                        state.num_bootstrap, total_time)
            logger.info("-" * 70)
            logger.info(f"{'Param':>8} | {'Point Est':>12} | {'Boot Mean':>12} | {'SE':>12} | {'t-stat':>10}")
            logger.info("-" * 70)
            for i in idx:
                logger.info(f"  θ[{i:>3}] | {theta_hat[i]:>12.5f} | {stats.mean[i]:>12.5f} | "
                            f"{stats.se[i]:>12.5f} | {stats.t_stats[i]:>10.2f}")
            logger.info("-" * 70)
        return stats
