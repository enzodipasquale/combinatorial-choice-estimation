import torch
import sys

def solve_QS(self, _pricing_pb, local_id, lambda_k, p_j):
    local_id = torch.arange(self.num_local_agents) if local_id is None else local_id
    n_local_id = len(local_id)

    error_i_j = self.torch_local_errors
    modular_j_k = self.torch_item_data.get("modular", None)
    modular_i_j_k = self.torch_local_agent_data.get("modular", None)
    quadratic_j_j_k = self.torch_item_data.get("quadratic", None)
    quadratic_i_j_j_k = self.torch_local_agent_data.get("quadratic", None)

    device = error_i_j.device
    precision = error_i_j.dtype
    lambda_k = torch.tensor(lambda_k, device=device, dtype=precision)
    if p_j is not None:
        p_j = torch.tensor(p_j, device=device, dtype=precision)

    num_mod = modular_j_k.shape[-1] if modular_j_k is not None else 0
    num_mod_agent = modular_i_j_k.shape[-1] if modular_i_j_k is not None else 0
    num_quad = quadratic_j_j_k.shape[-1] if quadratic_j_j_k is not None else 0
    num_quad_agent = quadratic_i_j_j_k.shape[-1] if quadratic_i_j_j_k is not None else 0

    offset = 0
    lambda_mod = lambda_k[offset:offset + num_mod]; offset += num_mod
    lambda_mod_agent = lambda_k[offset:offset + num_mod_agent]; offset += num_mod_agent
    lambda_quad = lambda_k[offset:offset + num_quad]; offset += num_quad
    lambda_quad_agent = lambda_k[offset:]

    P_i_j_j = torch.zeros((n_local_id, self.num_items, self.num_items), device=device, dtype=precision)

    diag_i_j = P_i_j_j.diagonal(dim1=1, dim2=2)
    diag_i_j.copy_(diag_i_j + error_i_j)
    if modular_j_k is not None:
        diag_i_j.copy_(diag_i_j + modular_j_k @ lambda_mod)
    if modular_i_j_k is not None:
        diag_i_j.copy_(diag_i_j + modular_i_j_k[local_id] @ lambda_mod_agent)
    if p_j is not None:
        diag_i_j.copy_(diag_i_j - p_j)

    if quadratic_j_j_k is not None:
        P_i_j_j += (quadratic_j_j_k @ lambda_quad).unsqueeze(0)
    if quadratic_i_j_j_k is not None:
        P_i_j_j += quadratic_i_j_j_k[local_id] @ lambda_quad_agent

    # Symmetrize
    diag_i_j = P_i_j_j.diagonal(dim1=1, dim2=2).clone() 
    P_i_j_j = P_i_j_j + P_i_j_j.transpose(1, 2)    
    P_i_j_j.diagonal(dim1=1, dim2=2).copy_(diag_i_j)
    
    constraint_i_j = error_i_j == - float('inf')

    mask_i_j_j = torch.tril(torch.ones((self.num_items, self.num_items), device=device, dtype=precision)).bool().unsqueeze(0)

    def _grad_lovatz_extension(z_i_j, P_i_j_j):
        # Sort z_i_j for each i
        sigma_i_j = torch.argsort(z_i_j, dim=1, descending=True)

        P_sigma = torch.gather(P_i_j_j, 1, sigma_i_j.unsqueeze(2).expand(-1, -1, self.num_items))
        P_sigma = torch.gather(P_sigma, 2, sigma_i_j.unsqueeze(1).expand(-1, self.num_items, -1))
        grad_i_sigma = (P_sigma * mask_i_j_j).sum(2) 

        # Compute the gradient
        grad_i_j = torch.zeros_like(z_i_j, device=device, dtype=precision)
        grad_i_j.scatter_(1, sigma_i_j, grad_i_sigma)

        # Compute the value of the Lovatz extension (assuming the value of the empty bundle is 0)
        fun_value_i = (z_i_j * grad_i_j).sum(1)

        return grad_i_j, fun_value_i


    z_t = torch.full((n_local_id, self.num_items), 0.5, device= device)
    z_t[constraint_i_j] = 0  
    z_best_i_j = torch.zeros_like(z_t)
    val_best_i = torch.full((n_local_id,),  -torch.inf)
    
    # sys.exit()

    num_iters_SGM = int(self.subproblem_settings["num_iters_SGM"])
    alpha = float(self.subproblem_settings["alpha"])
    method = self.subproblem_settings["method"]

    for iter in range(num_iters_SGM):
        # Compute gradient
        grad_i_j , val_i = _grad_lovatz_extension(z_t, P_i_j_j)

        grad_i_j = torch.where(constraint_i_j, 0, grad_i_j) 
        val_i = (z_t * grad_i_j).sum(1)


        # Take step: z_t + α g_t
        if method == 'constant_step_lenght':
            z_new = z_t + alpha * grad_i_j / torch.norm(grad_i_j, dim= 1, keepdim= True)

        elif method == 'constant_step_size':
            z_new = z_t + alpha * grad_i_j

        elif method == 'constant_over_sqrt_k':
            z_new = z_t + alpha * grad_i_j / ((iter + 1)**(1/2))

        elif method == 'mirror_descent':
            z_new = z_t * torch.exp(alpha * grad_i_j / torch.norm(grad_i_j, dim= 1, keepdim= True) )

        # project on Hypercube(J)
        z_t[constraint_i_j] = 0 
        z_t = torch.clamp(z_new, 0, 1)

        # Update best value
        new_best_value = val_best_i < val_i
        z_best_i_j[new_best_value] = z_t[new_best_value]
        val_best_i[new_best_value] = val_i[new_best_value]

    # Take the best value
    z_star = z_best_i_j
    optimal_bundle = (z_star.round() > 0).bool()
    # random_tensor = torch.rand_like(z_star)
    # optimal_bundle = (random_tensor < z_star).bool()

    verbose = self.subproblem_settings.get("verbose", True)

    if verbose:
        violations_rounding = ((z_star > .1) & (z_star < .9)).sum(1).cpu().numpy()
        aggregate_demand = optimal_bundle.sum(1).cpu().numpy()
        print(
            f"violations of rounding in SFM at rank {self.rank}: "
            f"mean={violations_rounding.mean()}, "
            f"max={violations_rounding.max()}, "
            f"demand: {aggregate_demand.mean()}, "
            f"{aggregate_demand.min()}, "
            f"{aggregate_demand.max()}"
            )      

    # print(optimal_bundle.sum(1).cpu().numpy())

    return optimal_bundle