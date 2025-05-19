import numpy as np
import gurobipy as gp

def init_KP(self, local_id):

    subproblem = gp.Model() 
    subproblem.setParam('OutputFlag', 0)
    subproblem.setParam('Threads', 1)
    time_limit = self.subproblem_settings.get("TimeLimit")
    if time_limit is not None:  
        subproblem.setParam("TimeLimit", time_limit)

    subproblem.setAttr('ModelSense', gp.GRB.MAXIMIZE)
    B_j = subproblem.addVars(self.num_items, vtype = gp.GRB.BINARY)

    weight_j = self.item_data["weights"]
    capacity = self.local_agent_data["capacity"][local_id]
    subproblem.addConstr(gp.quicksum(weight_j[j] * B_j[j] for j in range(self.num_items)) <= capacity)
    
    subproblem.update()

    return subproblem 


def solve_KP(self, subproblem, local_id, lambda_k, p_j):

    error_j = self.local_errors[local_id]
    modular_j_k = self.local_agent_data["modular"][local_id]
    
    L_j =  error_j + modular_j_k @ lambda_k
    if p_j is not None:
        L_j -= p_j

    subproblem.setObjective(gp.quicksum(L_j[j] * B_j[j] for j in range(self.num_items)))
    subproblem.optimize()

    optimal_bundle = np.array(subproblem.x, dtype=bool)
    
    mip_gap_tol = self.subproblem_settings.get("MIPGap_tol")
    if mip_gap_tol is not None:
        if subproblem.MIPGap > float(mip_gap_tol):
            print(f"WARNING: subproblem {local_id} in rank {self.rank} MIPGap: {subproblem.MIPGap}, value: {subproblem.objVal}")
    
    return optimal_bundle
