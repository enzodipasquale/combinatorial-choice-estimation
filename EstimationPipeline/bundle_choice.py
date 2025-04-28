from mpi4py import MPI
import numpy as np
import gurobipy as gp

class BundleChoice:
    def __init__(self, data, dims, config, compute_features):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.comm_size = self.comm.Get_size()

        self.data = data
        self.config = config
        self.num_agents, self.num_items, self.num_features = dims


        # Unpack config 
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

        self._compute_features = compute_features



    def compute_features(self, bundles):
        return self._compute_features(self, bundles)




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
                            "errors": self.error_si_j[si_chunks[r],:,None] if self.error_si_j is not None else None
                            }
                            for r in range(self.comm_size)
                            ]
        else:
            data_chunks = None
        
        local_data = self.comm.scatter(data_chunks, root=0)
        self.local_indeces = local_data["agent_indeces"]
        self.local_errors = local_data["errors"]
        self.local_agent_data = local_data["agent_data"]


    def initialize_master(self):

        master_pb = gp.Model('GMM_pb')
        master_pb.setParam('Method', 0)
        # master_pb.setParam('LPWarmStart', 2)

        

        # Variables and Objective
        phi_hat_k = self.compute_features(self.obs_bundle).sum(0)

        master_pb.setAttr('ModelSense', gp.GRB.MAXIMIZE)
        lambda_k = master_pb.addMVar(self.num_features, obj = phi_hat_k, ub = 1e9 , name='parameter')
        u_si = master_pb.addMVar(self.num_simuls * self.num_agents, obj = - (1/ self.num_simuls), name='utility')
        p_j = master_pb.addMVar(self.num_items, obj = -1 , name='price')

        # Initial Constraint (to make problem bounded)
        phi_i_k_all = self.compute_features(np.ones_like(self.obs_bundle))
        master_pb.addConstrs((
                u_si[si] + p_j.sum() >= self.error_si_j[si].sum() + phi_i_k_all[si % self.num_agents, :] @ lambda_k
                for si in range(self.num_simuls * self.num_agents)
                            ))

        # Solve master problem
        master_pb.optimize()




    
    







