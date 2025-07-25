import os
import inspect
from datetime import datetime
from dataclasses import dataclass, field, fields, MISSING
from typing import Callable, Tuple, Optional, Union, get_origin, get_args
from types import MethodType
import sys

import numpy as np
import gurobipy as gp
from mpi4py import MPI
from joblib import Parallel, delayed

from .utils import price_term, suppress_output, log_iteration, log_solution, log_init_master

import logging

logger = logging.getLogger(__name__)
if not logger.hasHandlers():  # Prevent duplicate handlers in notebooks or reloads
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
# Suppress Gurobi's license/banner logs routed through logging
logging.getLogger('gurobipy').setLevel(logging.WARNING)

@dataclass
class BundleConfig:
    # Required fields
    num_agents: int
    num_items: int
    num_features: int
    num_simuls: int

    # Optional configuration with defaults
    item_fixed_effects: bool = False
    tol_certificate: float = 0.01
    max_slack_counter: int = None  
    tol_row_generation: float = 0.0
    row_generation_decay: float = 0.0
    max_iters: int = None
    min_iters: int = 0
    subproblem_name: Optional[str] = None
    subproblem_settings: dict = field(default_factory=dict)
    master_settings: dict = field(default_factory=dict)
    master_ubs: Optional[dict] = None
    master_lbs: Optional[list] = None

    @staticmethod
    def from_dict(cfg: dict) -> 'BundleConfig':
        config_fields = {}

        for f in fields(BundleConfig):
            name = f.name

            # Determine the value to use
            if name in cfg:
                val = cfg[name]
            elif f.default is not MISSING:
                val = f.default
            elif f.default_factory is not MISSING:
                val = f.default_factory()
            else:
                continue 

            # Extract base type if Optional
            typ = f.type
            if get_origin(typ) is Union:
                args = get_args(typ)
                not_none = [t for t in args if t is not type(None)]
                if not_none:
                    typ = not_none[0]

            # Coerce basic scalar types
            if typ in {int, float, bool, str} and val is not None:
                if typ is int and isinstance(val, float) and not np.isfinite(val):
                    pass  # Avoid int(float('inf')) -> OverflowError
                else:
                    val = typ(val)

            config_fields[name] = val

        return BundleConfig(**config_fields)

class BundleChoice:
    def __init__(
                    self,
                    data: dict,
                    config: dict,
                    get_x_k: Callable,
                    init_pricing: Callable,
                    solve_pricing: Callable,
                ):
  
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.comm_size = self.comm.Get_size()
        self.local_thread_count = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() // self.comm_size))
        self.master_threads = int(os.environ.get("SLURM_CPUS_PER_NODE", os.cpu_count() ))

        self.data = data

        self.config = BundleConfig.from_dict(config)
        if self.config.tol_row_generation > 0:
            self.config.min_iters = np.ceil(np.log(self.config.tol_certificate 
                                        /(self.config.tol_row_generation + 1)) 
                                        /np.log(self.config.row_generation_decay))
        if self.config.max_slack_counter is None:
            self.config.max_slack_counter = float('inf')
        if self.config.max_iters is None:
            self.config.max_iters = float('inf')

        self.num_agents = self.config.num_agents
        self.num_items = self.config.num_items
        self.num_features = self.config.num_features
        self.num_simuls = self.config.num_simuls

        self.subproblem_name = self.config.subproblem_name
        self.subproblem_settings = self.config.subproblem_settings

        # Initialize user-defined methods
        self.get_x_k = MethodType(get_x_k, self)
        self.solve_pricing = MethodType(solve_pricing, self)
        self.init_pricing = MethodType(init_pricing, self) if init_pricing is not None else None


    # Features methods
    def get_x_i_k(self, B_i_j):
        return np.stack([self.get_x_k(i, B_i_j[i]) for i in range(self.num_agents)])

    def get_x_si_k(self, B_si_j):
        return np.stack([self.get_x_k(si % self.num_agents, B_si_j[si]) 
                        for si in range(self.num_simuls * self.num_agents)])

    def get_x_si_k_MPI(self, B_local):
        x_i_k_local = []
        for local_id in range(self.num_local_agents):
            x_local = self.get_x_k(local_id, B_local[local_id], local = True) 
            x_i_k_local.append(x_local)
        x_i_k_local = np.array(x_i_k_local) 
        x_si_k = self.comm.gather(x_i_k_local, root=0)
        if self.rank == 0:
            return np.concatenate(x_si_k)
        else:
            return None


    # Subproblem methods
    def init_local_pricing(self):
        if self.init_pricing is not None:
            with suppress_output():
                local_pricing_pbs = [self.init_pricing(local_id) for local_id in range(self.num_local_agents)]
        else:
            local_pricing_pbs = self.local_indeces
        return local_pricing_pbs


    def solve_local_pricing(self, local_pricing_pbs, lambda_k, p_j):
        tic = datetime.now()
        # If solve_subproblem method is parallelizable across local agents
        if self.subproblem_settings.get("parallel_local", False):
            if self.subproblem_settings.get("multithreading", False):
                # Parallelize across CPUs using joblib
                indices_chunks = np.array_split(np.arange(self.num_local_agents), self.local_thread_count)

                results = Parallel(n_jobs=self.local_thread_count, backend="threading")(
                                        delayed(self.solve_pricing)(None, chunk, lambda_k, p_j) 
                                        for chunk in indices_chunks
                                        )

                out = np.concatenate(results)
            else:
                # Solve all local pricing problems in parallel
                out =  np.array(self.solve_pricing(local_pricing_pbs, None, lambda_k, p_j))

        # If solve_subproblem method is serial across local agents
        elif self.subproblem_settings.get("multithreading", False):
            # Parallelize across CPUs using joblib
            out =  np.array(
                            Parallel(n_jobs=self.local_thread_count, backend="threading")(
                                delayed(self.solve_pricing)(pb, local_id, lambda_k, p_j)
                                for local_id, pb in enumerate(local_pricing_pbs)
                                )
                            )
        else:
            # Fallback: solve sequentially
            out = np.array([self.solve_pricing(pb, local_id, lambda_k, p_j)
                                for local_id, pb in enumerate(local_pricing_pbs)
                            ])
        self.comm.Barrier()  
        if self.rank == 0:
            logger.info("Pricing solved. Time elapsed: %s", datetime.now() - tic)                 
        return out


    def solve_pricing_offline(self, lambda_k, p_j = None):
        local_pricing_pbs = self.init_local_pricing()
        local_pricing_results = self.solve_local_pricing(local_pricing_pbs, lambda_k, p_j)
        pricing_results = self.comm.gather(local_pricing_results, root= 0)

        return np.concatenate(pricing_results) if self.rank == 0 else None

    # Data methods
    def scatter_data(self):
        if self.rank == 0:
            self.item_data = self.data.get("item_data")
            self.agent_data = self.data.get("agent_data")
            error_s_i_j = self.data.get("errors")
            self.error_si_j = error_s_i_j.reshape(-1, self.num_items) if error_s_i_j is not None else None        
            self.obs_bundle = self.data.get("obs_bundle")
    
            all_indices_chunks = np.array_split(np.arange(self.num_simuls * self.num_agents), self.comm_size)

            data_chunks =   [
                    {
                    "agent_indeces":    indices,
                    "agent_data":       {key : value[indices % self.num_agents] for key, value in self.agent_data.items()} 
                                        if self.agent_data is not None else None,
                    "errors":           self.error_si_j[indices,:] if self.error_si_j is not None else None
                    }
                    for indices in all_indices_chunks
                    ]
        else:
            self.item_data = None
            data_chunks = None

        self.item_data = self.comm.bcast(self.item_data, root=0)
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

        self.torch_item_data =  {
                                k: to_torch(v) for k, v in self.item_data.items()
                                } if self.item_data is not None else None

        self.torch_local_agent_data =   {
                                        k: to_torch(v) for k, v in self.local_agent_data.items()
                                        } if self.local_agent_data is not None else None

        self.torch_local_errors = to_torch(self.local_errors)

    # Row generation 
    def _init_master(self):
        if self.rank == 0:
            with suppress_output():
                master_pb = gp.Model()
                master_pb.setParam('Method', 0)
                master_pb.setParam('Threads', self.master_threads)
                master_pb.setParam('LPWarmStart', 2)
                master_pb.setAttr('ModelSense', gp.GRB.MAXIMIZE)
                OutputFlag = self.config.master_settings.get("OutputFlag")
                if OutputFlag is not None:
                    master_pb.setParam('OutputFlag', OutputFlag) 
            x_hat_i_k = self.get_x_i_k(self.obs_bundle)
            x_hat_k = x_hat_i_k.sum(0)

            lambda_k = master_pb.addMVar(self.num_features, obj =  self.num_simuls * x_hat_k, ub = 1e4 , name='parameter')
            u_si = master_pb.addMVar(self.num_simuls * self.num_agents, obj = - 1, name='utility')
            
            ubs = self.config.master_settings.get("ubs")
            if ubs is None:
                pass
            else:
                for k in range(self.num_features):
                    if ubs[k] is not None:
                        lambda_k[k].ub = ubs[k]

            lbs = self.config.master_settings.get("lbs")
            if lbs is None:
                pass
            else:
                for k in range(self.num_features):
                    if lbs[k] is not None:
                        lambda_k[k].lb = lbs[k]
            master_pb.update()

            if self.config.item_fixed_effects:
                p_j = master_pb.addMVar(self.num_items, obj = - self.num_simuls, name='price')
            else:
                p_j = None

            # x_i_k_all = self.get_x_i_k(np.ones_like(self.obs_bundle))
            # master_pb.addConstrs((
            #         u_si[si] + price_term(p_j, np.ones(self.num_items)) >= 
            #         self.error_si_j[si].sum() + x_i_k_all[si % self.num_agents, :] @ lambda_k
            #         for si in range(self.num_simuls * self.num_agents)
            #                     ))

            master_pb.addConstrs((
                                    u_si[si] + price_term(p_j, self.obs_bundle[si % self.num_agents]) >= 
                                    self.error_si_j[si] @ self.obs_bundle[si % self.num_agents] 
                                    + x_hat_i_k[si % self.num_agents, :] @ lambda_k
                                    for si in range(self.num_simuls * self.num_agents)  
                                    if np.all(self.error_si_j[si] > float('-inf'))
                                ))


            log_init_master(self, x_hat_k)
            master_pb.optimize()
            logger.info("Master Initialized. Parameter: %s", lambda_k.x)

            return master_pb, (lambda_k, u_si, p_j), lambda_k.x, p_j.x if p_j is not None else None
        else:
            return None, None, None, None


    def _update_slack_counter(self, master_pb, slack_counter):
        to_remove = []
        for constr in master_pb.getConstrs():
            constr_name = constr.ConstrName
            
            if constr_name not in slack_counter:
                slack_counter[constr_name] = 0

            if constr.Slack < 0:
            # if constr.CBasis == 0:
                slack_counter[constr_name] += 1
            else:
                slack_counter[constr_name] = 0
  
            if slack_counter[constr_name] >= self.config.max_slack_counter:
                to_remove.append((constr_name, constr))

        for constr_name, constr in to_remove:
            master_pb.remove(constr)
            slack_counter.pop(constr_name, None)

        return slack_counter, len(to_remove)

    def _master_iteration(self, master_pb, vars_tuple, pricing_results, slack_counter = None):
        if self.rank == 0:
            tic = datetime.now()

            B_star_si_j = np.concatenate(pricing_results).astype(bool)
            lambda_k, u_si, p_j = vars_tuple

            # eps_si_star = (self.error_si_j * B_star_si_j).sum(1)
            eps_si_star = np.where(B_star_si_j, self.error_si_j , 0).sum(1)
            # assert that the values in eps_si_star are finite
            assert np.all(np.isfinite(eps_si_star)), "WARNING: eps_si_star contains non-finite values." 
            x_star_si_k = self.get_x_si_k(B_star_si_j)
            u_si_star = x_star_si_k @ lambda_k.x + eps_si_star
            assert np.all(np.isfinite(u_si_star)), "WARNING: u_si_star contains non-finite values."

            if p_j is not None:
                u_si_star -= B_star_si_j @ p_j.x
            
            u_si_master = u_si.x
            max_reduced_cost = np.max(u_si_star - u_si_master)
            logger.info("Reduced cost: %s", max_reduced_cost)
            if max_reduced_cost < self.config.tol_certificate:
                return True, lambda_k.x, p_j.x if p_j is not None else None
            
            new_constrs_id = np.where(u_si_star > u_si_master * (1+ self.config.tol_row_generation))[0]
            
            logger.info("New constraints: %d", len(new_constrs_id))
            master_pb.addConstrs((  
                                u_si[si] + price_term(p_j, B_star_si_j[si,:]) >= eps_si_star[si] + x_star_si_k[si] @ lambda_k 
                                for si in new_constrs_id
                                ))

            slack_counter, num_constrs_removed = self._update_slack_counter(master_pb, slack_counter)
            logger.info("Removed constraints: %d", num_constrs_removed)
            master_pb.optimize()
            logger.info(f"Master solved.    Time elapsed: {datetime.now() - tic}")
            logger.info(f"Objective value: {master_pb.objVal:.2f}")

            logger.info("Parameter: %s", lambda_k.x)
            self.config.tol_row_generation *= self.config.row_generation_decay


            return False, lambda_k.x, p_j.x if p_j is not None else None
        else:
            return None, None, None
    
    def compute_estimator_row_gen(self):
        #========== Initialization =========#
        tic = datetime.now()
        local_pricing_pbs = self.init_local_pricing()
        master_pb, vars_tuple, lambda_k_iter, p_j_iter = self._init_master()     
        slack_counter = {}
        lambda_k_iter, p_j_iter = self.comm.bcast((lambda_k_iter, p_j_iter), root=0)

        #=========== Main loop ===========#
        for iteration in range(int(self.config.max_iters)):
            log_iteration(iteration, self.rank)  

            local_pricing_results = self.solve_local_pricing(local_pricing_pbs, lambda_k_iter, p_j_iter)
            pricing_results = self.comm.gather(local_pricing_results, root= 0)
            # x_star_si_k = self.get_x_si_k_MPI(local_pricing_results) 
    
            stop, lambda_k_iter, p_j_iter = self._master_iteration(master_pb, vars_tuple, pricing_results, slack_counter)
            stop, lambda_k_iter, p_j_iter = self.comm.bcast((stop, lambda_k_iter, p_j_iter) , root=0)
            

            if stop and iteration >= self.config.min_iters:
                log_solution(master_pb, lambda_k_iter, self.rank, datetime.now() - tic)
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
    #             # logger.info('Non productive')
            
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


