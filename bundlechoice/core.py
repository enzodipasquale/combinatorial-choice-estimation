import os
import inspect
from datetime import datetime
from dataclasses import dataclass, field
from typing import Callable, Tuple, Optional

import numpy as np
import gurobipy as gp
from mpi4py import MPI

from .utils import price_term, suppress_output, log_iteration, log_solution, log_init_master

@dataclass(frozen=True)
class BundleConfig:
    num_agents: int
    num_items: int
    num_features: int
    num_simuls: int

    item_fixed_effects: bool = False

    tol_certificate: float = 0.01
    max_slack_counter: int = np.inf
    tol_row_generation: float = 0
    row_generation_decay: float = 0
    max_iters: int = np.inf
    min_iters: Optional[float] = None

    subproblem: Optional[str] = None 
    subproblem_settings: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(cfg: dict) -> 'BundleConfig':
        cfg = dict(cfg)  
        cfg.setdefault("item_fixed_effects", False)

        cfg.setdefault("tol_certificate", 0.01)
        cfg.setdefault("max_slack_counter", np.inf)
        cfg.setdefault("tol_row_generation", 0)
        cfg.setdefault("row_generation_decay", 0)
        cfg.setdefault("max_iters", np.inf)

        cfg.setdefault("subproblem", None)
        cfg.setdefault("subproblem_settings", {})
        
        cfg["tol_certificate"] = float(cfg["tol_certificate"])
        cfg["tol_row_generation"] = float(cfg["tol_row_generation"])
        cfg["row_generation_decay"] = float(cfg["row_generation_decay"])

        # Compute min_iters 
        if cfg.get("min_iters") is None:
            tol_cert = cfg["tol_certificate"]
            tol_row = cfg["tol_row_generation"]
            decay = cfg["row_generation_decay"]
            if tol_row == 0:
                cfg["min_iters"] = 0
            else:
                cfg["min_iters"] = np.log(tol_cert / (tol_row - 1)) / np.log(decay)

        return BundleConfig(**cfg)

class BundleChoice:
    def __init__(
                    self,
                    data: dict,
                    config: dict,
                    get_x_k: Callable,
                    init_pricing: Callable,
                    solve_pricing: Callable,
                ):
        # Initialize MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.comm_size = self.comm.Get_size()

        # Load data
        self.data = data

        # Parse config and unpack parameters
        self.config = BundleConfig.from_dict(config)

        self.num_agents = self.config.num_agents
        self.num_items = self.config.num_items
        self.num_features = self.config.num_features
        self.num_simuls = self.config.num_simuls

        self.item_fixed_effects = self.config.item_fixed_effects
        self.tol_certificate = self.config.tol_certificate
        self.max_slack_counter = self.config.max_slack_counter
        self.tol_row_generation = self.config.tol_row_generation
        self.row_generation_decay = self.config.row_generation_decay

        self.max_iters = self.config.max_iters
        self.min_iters = self.config.min_iters

        self.subproblem = self.config.subproblem
        self.subproblem_settings = self.config.subproblem_settings

        # Initialize user-defined methods
        self._get_x_k = get_x_k
        self._init_pricing = init_pricing
        self._solve_pricing = solve_pricing


    # Features methods
    def get_x_k(self, i_id, B_j, local = False):
        return self._get_x_k(self, i_id, B_j, local)

    def get_x_i_k(self, B_i_j):
        return np.stack([self.get_x_k(i, B_i_j[i]) for i in range(self.num_agents)])

    def get_x_si_k(self, B_si_j):
        return np.stack([self.get_x_k(si % self.num_agents, B_si_j[si]) 
                        for si in range(self.num_simuls * self.num_agents)])


    # Pricing problem methods
    def init_pricing(self, local_id):
        return self._init_pricing(self, local_id)

    def init_local_pricing(self):
        if self._init_pricing is not None:
            with suppress_output():
                local_pricing_pbs = [self.init_pricing(local_id) for local_id in range(self.num_local_agents)]
        else:
            local_pricing_pbs = self.local_indeces
        return local_pricing_pbs

    def solve_pricing(self, pricing_pb, local_id, lambda_k_iter, p_j_iter):
        return self._solve_pricing(self, pricing_pb, local_id, lambda_k_iter, p_j_iter)

    def solve_local_pricing(self, local_pricing_pbs, lambda_k_iter, p_j_iter):
        if self.subproblem_settings.get("parallel_local", False):
            return np.array(self.solve_pricing(local_pricing_pbs, None, lambda_k_iter, p_j_iter))
        else:
            return np.array([self.solve_pricing(pricing_pb, local_id, lambda_k_iter, p_j_iter) 
                            for local_id, pricing_pb in enumerate(local_pricing_pbs)])

    def solve_pricing_offline(self, lambda_k, p_j = None):
        local_pricing_pbs = self.init_local_pricing()
        local_pricing_results = self.solve_local_pricing(local_pricing_pbs, lambda_k, p_j)
        pricing_results = self.comm.gather(local_pricing_results, root= 0)

        return np.concatenate(pricing_results) if self.rank == 0 else None

    # Data methods
    def scatter_data(self):
        if self.rank == 0:
            self.item_data = self.data.get("item_data", None)
            self.agent_data = self.data.get("agent_data", None)
            self.error_s_i_j = self.data.get("errors", None)
            self.error_si_j = self.error_s_i_j.reshape(-1, self.num_items) if self.error_s_i_j is not None else None
            del self.error_s_i_j
            self.obs_bundle = self.data.get("obs_bundle", None)
            # if self.obs_bundle is not None:
            #     self.obs_bundle = self.obs_bundle.astype(bool)
        else:
            self.item_data = None
        self.item_data = self.comm.bcast(self.item_data, root=0)

        if self.rank == 0:
            i_chunks = np.array_split(np.tile(np.arange(self.num_agents), self.num_simuls), self.comm_size)
            si_chunks = np.array_split(np.arange(self.num_simuls * self.num_agents), self.comm_size)

            data_chunks = [
                            {"agent_indeces": i_chunks[r],
                            "agent_data": {key : value[i_chunks[r]] for key, value in self.agent_data.items()} 
                                            if self.agent_data is not None else None,
                            "errors": self.error_si_j[si_chunks[r],:] if self.error_si_j is not None else None
                            }
                            for r in range(self.comm_size)]
        else:
            data_chunks = None
        local_data = self.comm.scatter(data_chunks, root=0)

        self.local_indeces = local_data["agent_indeces"]
        self.local_errors = local_data["errors"]
        self.local_agent_data = local_data["agent_data"]
        self.num_local_agents = len(self.local_indeces)


    def local_data_to_torch(self, precision="float32"):
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float64 if precision == "float64" else torch.float32

        self.using_torch = True
        self.torch_device = device
        self.torch_dtype = dtype

        def to_torch(x):
            return torch.tensor(x, device=device, dtype=dtype) if x is not None else None

        self.torch_item_data = {
                                k: to_torch(v) for k, v in self.item_data.items()
                                } if self.item_data is not None else None

        self.torch_local_agent_data = {
                                        k: to_torch(v) for k, v in self.local_agent_data.items()
                                        } if self.local_agent_data is not None else None

        self.torch_local_errors = to_torch(self.local_errors)

    # Master problem methods
    def _init_master(self):
        if self.rank == 0:
            master_pb = gp.Model('GMM_pb')
            master_pb.setParam('Method', 0)
            master_pb.setParam('LPWarmStart', 2)

            x_hat_i_k = self.get_x_i_k(self.obs_bundle)
            x_hat_k = x_hat_i_k.sum(0)

            master_pb.setAttr('ModelSense', gp.GRB.MAXIMIZE)
            lambda_k = master_pb.addMVar(self.num_features, obj =  self.num_simuls * x_hat_k, ub = 1e9 , name='parameter')
            u_si = master_pb.addMVar(self.num_simuls * self.num_agents, obj = - 1, name='utility')

            if self.item_fixed_effects:
                p_j = master_pb.addMVar(self.num_items, obj = - self.num_simuls, name='price')
            else:
                p_j = None

            x_i_k_all = self.get_x_i_k(np.ones_like(self.obs_bundle))
            master_pb.addConstrs((
                    u_si[si] + price_term(p_j, np.ones(self.num_items)) >= 
                    self.error_si_j[si].sum() + x_i_k_all[si % self.num_agents, :] @ lambda_k
                    for si in range(self.num_simuls * self.num_agents)
                                ))

            master_pb.addConstrs((
                                    u_si[si] + price_term(p_j, self.obs_bundle[si % self.num_agents]) >= 
                                    self.error_si_j[si] @ self.obs_bundle[si % self.num_agents] 
                                    + x_hat_i_k[si % self.num_agents, :] @ lambda_k
                                    for si in range(self.num_simuls * self.num_agents)
                                ))


            log_init_master(self, x_hat_k)
            master_pb.optimize()
            print('-'*80)
            print("Parameter:", lambda_k.x)
            return master_pb, (lambda_k, u_si, p_j), lambda_k.x, p_j.x if p_j is not None else None
        else:
            return None, None, None, None


    def _update_slack_counter(self, master_pb, slack_counter):
        to_remove = []
        for constr in master_pb.getConstrs():
            constr_name = constr.ConstrName
            
            if constr_name not in slack_counter:
                slack_counter[constr_name] = 0

            # if constr.Slack < 0:
            if constr.CBasis == 0:
                slack_counter[constr_name] += 1
            else:
                slack_counter[constr_name] = 0
  
            if slack_counter[constr_name] >= self.max_slack_counter:
                to_remove.append((constr_name, constr))

        for constr_name, constr in to_remove:
            master_pb.remove(constr)
            slack_counter.pop(constr_name, None)

        return slack_counter, len(to_remove)

    def _master_iteration(self, master_pb, vars_tuple, pricing_results, slack_counter = None):
        if self.rank == 0:
            pricing_results = np.concatenate(pricing_results)
            lambda_k, u_si, p_j = vars_tuple
            u_si_star = pricing_results[:,0]
            B_star_si_j = pricing_results[:,1:].astype(bool)

            eps_si_star = (self.error_si_j * B_star_si_j).sum(1)
            x_star_si_k = self.get_x_si_k(B_star_si_j)
            
            u_si_master = u_si.x
            max_reduced_cost = np.max(u_si_star - u_si_master)
            print('-'*80)
            print("Reduced cost:", max_reduced_cost)
            if max_reduced_cost < self.tol_certificate:
                return True, lambda_k.x, p_j.x if p_j is not None else None
            
            new_constrs_id = np.where(u_si_star > u_si_master * (1+ self.tol_row_generation))[0]
            print("New constraints:", len(new_constrs_id))
            master_pb.addConstrs((  
                                u_si[si] + price_term(p_j, B_star_si_j[si,:]) >= eps_si_star[si] + x_star_si_k[si] @ lambda_k 
                                for si in new_constrs_id
                                ))

            slack_counter, num_constrs_removed = self._update_slack_counter(master_pb, slack_counter)
            print("Removed constraints:", num_constrs_removed)
            print('-'*80)
            master_pb.optimize()
            print('-'*80)
            print("Parameter:", lambda_k.x)

            self.tol_row_generation *= self.row_generation_decay

            return False, lambda_k.x, p_j.x if p_j is not None else None
        else:
            return None, None, None
    
    # Row generation 
    def compute_estimator_row_gen(self):
        #========== Initialization =========#
        local_pricing_pbs = self.init_local_pricing()
        master_pb, vars_tuple, lambda_k_iter, p_j_iter = self._init_master()     
        slack_counter = {}
        lambda_k_iter, p_j_iter = self.comm.bcast((lambda_k_iter, p_j_iter), root=0)

        #=========== Main loop ===========#
        for iteration in range(self.max_iters):
            local_pricing_results = self.solve_local_pricing(local_pricing_pbs, lambda_k_iter, p_j_iter)
            pricing_results = self.comm.gather(local_pricing_results, root= 0)

            log_iteration(iteration, lambda_k_iter, self.rank )                
            stop, lambda_k_iter, p_j_iter = self._master_iteration(master_pb, vars_tuple, pricing_results, slack_counter)
            stop, lambda_k_iter, p_j_iter = self.comm.bcast((stop, lambda_k_iter, p_j_iter) , root=0)

            if stop and iteration >= self.min_iters:
                log_solution(master_pb, lambda_k_iter, self.rank)
                break
        return lambda_k_iter, p_j_iter











    # def compute_estimator_ellipsoid(self, tol, A_init = None, c_init = None, max_iters = np.inf, lb_c = None):

    #     if self.item_fixed_effects:
    #         raise NotImplementedError("Ellipsoid method not implemented for item fixed effects.") 

    #     if self.rank == 0:

    #         num_dims = self.num_features 
    #         A_k = np.eye(num_dim) if A_init is None else A_init
    #         c_k = np.zeros(num_dim) if c_init is None else c_init
    #         lb_c = - np.inf if lb_c is None else lb_c

    #         # check this
    #         max_iters = self.num_features * (self.num_features - 1) * np.log(1/ tol)  if max_iters is None else max_iters
    #         centers, matrices = [c_k], [A_k]

    #     # Initialize pricing 
    #     if self._init_pricing is not None:
    #         with suppress_output():
    #             local_pricing_pbs = [self.init_pricing(local_id) for local_id in range(self.num_local_agents)]
    #     else:
    #         local_pricing_pbs = self.local_indeces
    #     iter = 0
    #     while iter < max_iters:
    #         # if self.rank == 0:
    #         if np.any(c_k < 0):
    #             a_k = - ((c_k < 0) * 1)
    #             value = np.inf
    #             # print('Non productive')
            
    #         else:
    #             # Solve pricing problems
    #             local_pricing_results = np.array([self.solve_pricing(pricing_pb, local_id, lambda_k_iter, p_j_iter) 
    #                                         for local_id, pricing_pb in enumerate(local_pricing_pbs)])

    #             # Gather pricing results at rank 0
    #             pricing_results = self.comm.gather(local_pricing_results, root= 0)

    #             # Compute subgradient
                
                


            
    #         ### Update ellipsoid
    #         b_k = (A_k @ a_k) / ((a_k @ A_k @ a_k) ** (1 / 2))
    #         c_k = c_k - (1 / (num_dim + 1)) * b_k
    #         A_k = gamma_1 * A_k - gamma_2 * np.outer(b_k, b_k)


    #         hplanes.append(a_k.copy())
    #         values.append(value)

    #         centers.append(c_k.copy())
    #         ellipsoid_matrices.append(A_k.copy())

    #         iter += 1

    #     best_val_id = np.argmin(values)
    #     solution = centers[best_val_id]

    #     return solution, hplanes, values, centers, ellipsoid_matrices


