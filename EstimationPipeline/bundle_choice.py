from mpi4py import MPI
import numpy as np

class BundleChoice:
    def __init__(self, data, config):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.comm_size = self.comm.Get_size()

        self.data = data
        self.config = config


        # Unpack config
        self.tol_certificate = config["tol_certificate"]
        self.max_slack_counter = config["max_slack_counter"]
        self.tol_row_generation = config["tol_row_generation"]
        self.row_generation_decay = config["row_generation_decay"]
        self.num_simulations = config["num_simulations"]
        self.max_iters = config["max_iters"]
        self.min_iters = config["min_iters"]




    def scatter_data(self):
        # Load agent-independent data on all ranks
        if self.rank == 0:
            self.item_data = self.data.get("item_data", None)
            self.agent_data = self.data.get("agent_data", None)
            self.errors = self.data.get("errors", None)

            dims = self.errors.shape if self.errors is not None else None
        else:
            self.item_data = None
            dims = None

        # scatter item_data to all ranks
        self.item_data = self.comm.bcast(self.item_data, root=0)
        self.num_simul, self.num_agents, self.num_objs = self.comm.bcast(dims, root=0)

        # scatter agent_data and errors in chunks
        if self.rank == 0:
            i_chunks = np.array_split(np.tile(np.arange(self.num_agents), self.num_simul), self.comm_size)
            si_chunks = np.array_split(np.arange(self.num_simul * self.num_agents), self.comm_size)

            errors_flat = self.errors.reshape(self.num_simul * self.num_agents, self.num_objs) if self.errors is not None else None

            data_chunks = [{
                            "agent_data": {key : value[i_chunks[r]] for key, value in self.agent_data.items()} 
                                            if self.agent_data is not None else None,
                            "errors": errors_flat[si_chunks[r],:,None]
                            }
                            for r in range(self.comm_size)
                            ]
        else:
            data_chunks = None
        
        local_data = self.comm.scatter(data_chunks, root=0)
        self.local_errors = local_data["errors"]
        self.local_agent_data = local_data["agent_data"]



    







