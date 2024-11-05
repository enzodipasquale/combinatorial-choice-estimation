import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
options = {
 "WLSACCESSID":"a4353fb7-f95b-4075-b288-ca3f60983b36",
"WLSSECRET":"d894d460-2dac-4210-8c40-c91c68ecfb13",
"LICENSEID":2562382
}



K = 4
lambda_k_true = np.array([1,15,-1,1])

I = 30
J = 100



### generate exogenous data
np.random.seed(1)

φ_i_j_k = np.random.normal(1,1, size=[I,J,K-1])

def φ_k(i,B):
    # return φ_i_j_k[i,B,:].sum(0)
    return np.concatenate((φ_i_j_k[i,B,:].sum(0), [ - np.sum(B)**(2)]))

### greedy
def greedy(i,lambda_k, p_j = np.zeros(J)):
    B_k =  np.zeros(J, dtype=bool)
    val = np.inner(φ_k(i,B_k),lambda_k)
    k = 0 
    while k < J:
        j_k = None
        max_add_j =  - np.inf
        for j in np.where(1-B_k)[0]:
            val_add_j = np.inner(φ_k(i,B_k+ np.eye(1,J,j,dtype=bool)[0]), lambda_k)   - p_j[j] - p_j[B_k].sum() - val
            if val_add_j > max_add_j:
                j_k = j
                max_add_j = val_add_j
            
        if j_k is None or max_add_j < 0:
            break
        val += max_add_j 
        B_k[j_k] = 1
        k += 1
    return B_k , val



max_iters = 1000
tol = 1e-12

constraints_list = []

# Create the environment with license parameters
with gp.Env(params=options) as env:
    # Create a Gurobi model within the environment
    with gp.Model(env=env) as model:
        ### Initialize 
        # Create variables
        u_i = model.addVars(I, name="utilities")
        # p_j = model.addVars(J, name="prices")
        p_j = model.addVars(J, lb= 0 ,ub=GRB.INFINITY, name="prices")

        # Set objective
        model.setObjective(u_i.sum() + p_j.sum(), GRB.MINIMIZE)
        
        # Optimize the model
        model.setParam('OutputFlag', 0)
        model.optimize()
        
        # Extract the solution (theta_solution)
        theta_solution = np.array(model.x)

        
        ### Column Generation
        iter = 0
        while iter < max_iters:
            print(f"ITER: {iter}")
            ### Pricing problem
            start_time = time.time()
            B_star_i = []
            reduced_cost_i = []
            for i in range(I):
                        B_star, val = greedy(i  ,lambda_k = lambda_k_true, p_j = theta_solution[-J:])
                        B_star_i.append(B_star)
                        reduced_cost_i.append(val -  theta_solution[i])
            end_time = time.time()
            print(f"greedy done:{end_time - start_time}")
            # stop if certificate holds
            print(f"reduced cost: {np.max(reduced_cost_i)}")
            start_time = time.time()
            if np.max(reduced_cost_i) <= tol:
                primal_solution = np.array(model.x)
                dual_solution = np.array(model.pi)
                break
            
            ### Master problem
            model.addConstrs((u_i[i] + gp.quicksum(p_j[j] for j in np.where(B_star_i[i])[0])>= 
                            φ_k(i, B_star_i[i])@lambda_k_true 
                            for i in range(I)), name="constraint_batch")
            
            constraints_list.append(B_star_i)
            end_time = time.time()
            # print(f"constraints added:{end_time - start_time}")

            # Optimize the model
            start_time = time.time()
            u_i.start = reduced_cost_i + theta_solution[:I]
            p_j.start = theta_solution[-J:]
            model.optimize()
            theta_solution = np.array(model.x)
            iter += 1
            end_time = time.time()  
            # print(f"model solved:{end_time - start_time}")
            print("##############")




            