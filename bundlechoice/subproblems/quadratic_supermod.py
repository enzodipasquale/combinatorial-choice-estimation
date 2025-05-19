import torch

def solve_QS(self, _pricing_pb, _local_id, lambda_k, p_j):
    
    error_i_j = self.torch_local_errors
    modular_i_j_k = self.torch_local_agent_data.get("modular", None)
    quadratic_j_j_k = self.torch_item_data.get("quadratic", None)
    quadratic_i_j_j_k = self.torch_local_agent_data.get("quadratic", None)

    device = error_i_j.device

    z_t = torch.full((self.num_local_agents, self.num_items), 0.5, device= device)
    z_best_i_j = torch.zeros_like(z_t)
    val_best_i = torch.full((self.num_local_agents,),  -torch.inf)

    num_mod = modular_i_j_k.shape[2] if modular_i_j_k is not None else 0
    num_quad = quadratic_j_j_k.shape[2] if quadratic_j_j_k is not None else 0
    num_quad_agent = quadratic_i_j_j_k.shape[2] if quadratic_i_j_j_k is not None else 0

    lambda_k_mod = lambda_k[:num_mod]
    lambda_k_quad = lambda_k[num_mod : num_mod + num_quad]
    lambda_k_quad_agent = lambda_k[num_mod + num_quad: num_mod + num_quad + num_quad_agent]

    zeros_i_j_j = torch.zeros((self.num_local_agents, self.num_items, self.num_items), device=device)
    upper_triangular = torch.triu(torch.ones((self.num_items, self.num_items), device=device), diagonal=1)

    def _grad_lovatz_extension(z_i_j, lambda_k, p_j, error_i_j, modular_i_j_k, quadratic_i_j_j_k, quadratic_j_j_k):
        # Sort z_i_j for each i
        sorted_z_id_j = torch.argsort(z_i_j, dim=1, descending=True)
        zeros_i_j_j = torch.zeros((self.num_local_agents, self.num_items, self.num_items), device=device)

        zeros_i_j_j[torch.arange(self.num_local_agents).unsqueeze(1).unsqueeze(2) ,
                        sorted_z_id_j.unsqueeze(1) ,
                        sorted_z_id_j.unsqueeze(2)] = upper_triangular

        mask = zeros_i_j_j.unsqueeze(-1)

        # Compute gradient
        grad_i_j = torch.zeros_like(z_i_j, device=device)

        if modular_i_j_k is not None:
            grad_i_j += modular_i_j_k @ lambda_k_mod
         
        if quadratic_j_j_k is not None:
            grad_i_j += torch.matmul((quadratic_j_j_k.unsqueeze(0) * mask).sum(-2) , lambda_k_quad)

        if quadratic_i_j_j_k is not None:
            grad_i_j += torch.matmul((quadratic_i_j_j_k * mask).sum(-2) , lambda_k_quad_agent)

        if error_i_j is not None:
            grad_i_j += error_i_j

        if p_j is not None:
            grad_i_j += p_j[None, :]

        # Compute the value of the Lovatz extension (assuming the value of the empty bundle is 0)
        fun_value_i = (z_i_j * grad_i_j).sum(1)

        return grad_i_j, fun_value_i
    
    num_iters_SGD = int(self.subproblem_settings["num_iters_SGD"])
    alpha = float(self.subproblem_settings["alpha"])
    method = self.subproblem_settings["method"]

    for iter in range(num_iters_SGD):
        # Compute gradient
        grad_i_j , val_i = _grad_lovatz_extension(z_t, lambda_k, p_j, error_i_j, modular_i_j_k, quadratic_i_j_j_k, quadratic_j_j_k)
     
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
        z_t = torch.clamp(z_new, 0, 1)

        # Update best value
        new_best_value = val_best_i < val_i
        z_best_i_j[new_best_value] = z_t[new_best_value]
        val_best_i[new_best_value] = val_i[new_best_value]

    # Take the best value
    z_star = z_best_i_j
    bundle_star = (z_star.round() > 0).bool()
    



    # random_tensor = torch.rand_like(z_star)
    # bundle_star = (random_tensor < z_star).bool()
    violations_rounding = ((z_star > .1) & (z_star < .9)).sum(1).cpu().numpy()
    print(f"violations of rounding in SFM at rank {self.rank}: ", violations_rounding.mean())
    print(bundle_star.sum(1).cpu().numpy())

    return bundle_star