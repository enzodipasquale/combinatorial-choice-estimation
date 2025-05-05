import numpy as np
import gurobipy as gp
from bundlechoice.utils import price_term


def init_QKP(self, local_id):

    subproblem = gp.Model() 
    subproblem.setParam('OutputFlag', 0)
    subproblem.setParam('Threads', 1)
    subproblem.setParam('TimeLimit', 60)
    subproblem.setAttr('ModelSense', gp.GRB.MAXIMIZE)

    # Create variables
    B_j = subproblem.addMVar(self.num_items, vtype = gp.GRB.BINARY)

    # Knapsack constraint
    weight_j = self.item_data["weights"]
    capacity = self.local_agent_data["capacity"][local_id]

    subproblem.addConstr(weight_j @ B_j <= capacity)
    subproblem.update()

    return subproblem 

def solve_QKP(self, subproblem, local_id, lambda_k, p_j):

    error_j = self.local_errors[local_id]
    modular_j_k = self.local_agent_data["modular"][local_id]
    quadratic_j_j_k = self.item_data["quadratic"]

    # Define objective from data and master solution 
    num_mod = modular_j_k.shape[-1]

    L_j =  error_j + modular_j_k @ lambda_k[:num_mod] - price_term(p_j)
    Q_j_j = quadratic_j_j_k @ lambda_k[num_mod: ]

    B_j = subproblem.getVars()
    subproblem.setObjective(B_j @ L_j + B_j @ Q_j_j @ B_j)

    # Solve the updated subproblem
    subproblem.optimize()

    optimal_bundle = np.array(subproblem.x, dtype=bool)
    value = subproblem.objVal
    
    if subproblem.MIPGap > .01:
        raise ValueError("MIP gap is larger than 1%")
    
    # Compute value, characteristics and error at optimal bundle
    pricing_result =   np.concatenate(( [value],
                                        [error_j[optimal_bundle].sum(0)],
                                        (modular_j_k[optimal_bundle]).sum(0), 
                                        quadratic_j_j_k[optimal_bundle][:, optimal_bundle].sum((0, 1)),
                                        subproblem.x
                                        ))
    return pricing_result
