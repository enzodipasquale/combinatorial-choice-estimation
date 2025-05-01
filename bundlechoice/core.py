from mpi4py import MPI
import numpy as np
import gurobipy as gp
from .utils import update_slack_counter

class BundleChoice:
    def __init__(self, data, dims, config, compute_features, init_pricing, solve_pricing):

        # Initialize MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.comm_size = self.comm.Get_size()

        # Load data 
        self.data = data
        self.num_agents, self.num_items, self.num_features = dims


        # Unpack config 
        self.item_fixed_effects = bool(config["item_fixed_effects"])
        self.tol_certificate = float(config["tol_certificate"])
        self.max_slack_counter = int(config["max_slack_counter"])
        self.tol_row_generation = int(config["tol_row_generation"])
        self.row_generation_decay = float(config["row_generation_decay"])
        self.num_simuls = int(config["num_simuls"])
        self.max_iters = int(config["max_iters"])
        if config["min_iters"] is None:
            self.min_iters = np.log(self.tol_certificate / (self.tol_row_generation - 1)) / np.log(self.row_generation_decay)
        else:
            self.min_iters = config["min_iters"]

        # Initialize user-defined variables
        self._compute_features = compute_features
        self._init_pricing = init_pricing
        self._solve_pricing = solve_pricing



    def compute_features(self, bundles):
        return self._compute_features(self, bundles)

    def init_pricing(self, local_id):
        return self._init_pricing(self, local_id)

    def solve_pricing(self, pricing_pb, local_id, lambda_k_iter, p_j_iter):
        return self._solve_pricing(self, pricing_pb, local_id, lambda_k_iter, p_j_iter)

    def scatter_data(self):
        # Load agent-independent data on all ranks
        if self.rank == 0:
            self.item_data = self.data.get("item_data", None)
            self.agent_data = self.data.get("agent_data", None)
            self.error_s_i_j = self.data.get("errors", None)
            self.error_si_j = self.error_s_i_j.reshape(self.num_simuls * self.num_agents, self.num_items) if self.error_s_i_j is not None else None

            self.obs_bundle = self.data["obs_bundle"]
        else:
            self.item_data = None

        # scatter item_data to all ranks
        self.item_data = self.comm.bcast(self.item_data, root=0)

        # scatter agent_data and errors in chunks
        if self.rank == 0:
            i_chunks = np.array_split(np.tile(np.arange(self.num_agents), self.num_simuls), self.comm_size)
            si_chunks = np.array_split(np.arange(self.num_simuls * self.num_agents), self.comm_size)

            data_chunks = [{"agent_indeces": i_chunks[r],
                            "agent_data": {key : value[i_chunks[r]] for key, value in self.agent_data.items()} 
                                            if self.agent_data is not None else None,
                            "errors": self.error_si_j[si_chunks[r],:] if self.error_si_j is not None else None
                            }
                            for r in range(self.comm_size)
                            ]
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

        def to_tensor(x):
            return torch.tensor(x, device=device, dtype=dtype) if x is not None else None

        self.torch_item_data = {
                                    k: to_tensor(v) for k, v in self.item_data.items()
                                } if self.item_data is not None else None

        self.torch_local_agent_data = {
                                            k: to_tensor(v) for k, v in self.local_agent_data.items()
                                        } if self.local_agent_data is not None else None

        self.torch_local_errors = to_tensor(self.local_errors)


    

    @staticmethod
    def price_term(p_j, bundle_j = None):
        if p_j is None:
            return 0
        if bundle_j is None:
            return p_j.sum() 
        else:
            return bundle_j @ p_j 
    
    def init_master(self):

        master_pb = gp.Model('GMM_pb')
        master_pb.setParam('Method', 0)
        # master_pb.setParam('OutputFlag', 0)
        # master_pb.setParam('LPWarmStart', 2)

        # Variables and Objective
        x_hat_k = self.compute_features(self.obs_bundle).sum(0)

        master_pb.setAttr('ModelSense', gp.GRB.MAXIMIZE)
        lambda_k = master_pb.addMVar(self.num_features, obj = x_hat_k, ub = 1e9 , name='parameter')
        u_si = master_pb.addMVar(self.num_simuls * self.num_agents, obj = - (1/ self.num_simuls), name='utility')

        if self.item_fixed_effects:
            p_j = master_pb.addMVar(self.num_items, obj = -1 , name='price')
        else:
            p_j = None

        # Initial Constraint (to make problem bounded)
        x_i_k_all = self.compute_features(np.ones_like(self.obs_bundle))
        master_pb.addConstrs((
                u_si[si] + self.price_term(p_j) >= self.error_si_j[si].sum() + x_i_k_all[si % self.num_agents, :] @ lambda_k
                for si in range(self.num_simuls * self.num_agents)
                            ))

        # Solve master problem
        master_pb.optimize()

        return master_pb, (lambda_k, u_si, p_j), lambda_k.x, p_j.x if p_j is not None else None


    def solve_master(self, master_pb, vars_tuple, pricing_results, slack_counter = None):

        lambda_k, u_si, p_j = vars_tuple

        # Unpack pricing results
        u_si_star = pricing_results[:,0]
        eps_si_star = pricing_results[:,1]
        x_star_si_k = pricing_results[:,2: - self.num_items]
        B_star_si_j = pricing_results[:,- self.num_items:]

        # Check certificate
        u_si_master = u_si.x
        max_reduced_cost = np.max(u_si_star - u_si_master)
        if max_reduced_cost < self.tol_certificate:
            return True, lambda_k.x, p_j.x if p_j is not None else None

        # Add new constraints
        new_constrs_id = np.where(u_si_star > u_si_master * (1+ self.tol_row_generation))[0]
        master_pb.addConstrs((  
                            u_si[si] + self.price_term(p_j, B_star_si_j[si,:]) >= eps_si_star[si] + x_star_si_k[si] @ lambda_k 
                            for si in new_constrs_id
                            ))

        # Update slack_counter
        slack_counter, num_constrs_removed = update_slack_counter(master_pb, slack_counter)

        # Solve master problem
        master_pb.optimize()
        print("Reduced cost:", max_reduced_cost)

        # Save results
        # master_pb.write('output/master_pb.mps')
        # master_pb.write('output/master_pb.bas')
                            
        return False, lambda_k.x, p_j.x if p_j is not None else None
    
    
    


    def compute_estimator_row_gen(self):

        # Initialize pricing 
        if self._init_pricing is not None:
            self.local_pricing_pbs = [self._init_pricing(local_id) for local_id in range(self.num_local_agents)]
            

        if self.rank == 0:
            # Initialize master problem
            master_pb, vars_tuple, lambda_k_iter, p_j_iter = self.init_master()
            # Initialize slack counter                                                               
            slack_counter = {"MAX_SLACK_COUNTER": self.max_slack_counter}
        else:
            lambda_k_iter, p_j_iter = None, None

        # Broadcast master solution to all ranks
        lambda_k_iter, p_j_iter = self.comm.bcast((lambda_k_iter, p_j_iter), root=0)


        # Main loop
        for iteration in range(self.max_iters):

            ### Solve pricing problems
            local_new_rows = np.array([self.solve_pricing(pricing_pb, local_id, lambda_k_iter, p_j_iter) 
                                        for local_id, pricing_pb in enumerate(local_pricing_pbs)])

            # Gather pricing results at rank 0
            pricing_results = self.comm.gather(local_new_rows, root= 0)


            ### Solve master at rank 0 
            if self.rank == 0:        
                
                print("ITERATION:", iteration)
                pricing_results = np.concatenate(pricing_results)
                stop, lambda_k_iter, p_j_iter = self.solve_master(master_pb, vars_tuple, pricing_results, slack_counter)
                print("#" * 80)
                print('Parameter:', lambda_k_iter)
                print("#" * 80)
            else:
                stop, lambda_k_iter, p_j_iter = None, None, None

            # Broadcast master results to all ranks
            stop, lambda_k_iter, p_j_iter = self.comm.bcast((stop, lambda_k_iter, p_j_iter) , root=0)

            # Break loop if stop is True and min iters is reached
            if stop and iteration > self.min_iters:
                if self.rank == 0:
                    print("Solution found:", lambda_k_iter)
                break


