import torch

def solve_QS(self, lambda_k, p_j, num_iters_SGD, alpha, method = 'constant_step_lenght'):

    error_i_j = self.torch_local_errors
    modular_i_j_k = self.torch_local_agent_data["modular"]
    quadratic_i_j_j_k = self.torch_local_agent_data["quadratic"]
    quadratic_j_j_k = self.torch_item_data["quadratic"]

    num_MOD = modular_i_j_k.shape[-1]

    z_t = torch.full((self.num_local_agents, self.num_items), 0.5, device= self.torch_device, dtype= self.torch_dtype)
    z_best_i_j = torch.zeros_like(z_t)
    val_best_i = torch.full_like(z_t, -torch.inf)

    for iter in range(num_iters_SGD):
        ### Compute gradient
        grad_i_j , val_i = _grad_lovatz_extension(z_t, lambda_k, p_j)

        ### Compute proj_H (z_t + α g_t) 
        # Take step z_t + α g_t
        if method == 'constant_step_lenght':
            z_new = z_t + alpha * grad_i_j / torch.norm(grad_i_j, dim= 1, keepdim= True)

        elif method == 'constant_step_size':
            z_new = z_t + alpha * grad_i_j

        elif method == 'constant_over_sqrt_k':
            z_new = z_t + alpha * grad_i_j / torch.sqrt(iter + 1)

        elif method == 'mirror_descent':
            z_new = z_t * torch.exp(alpha * grad_i_j / torch.norm(grad_i_j, dim= 1, keepdim= True) )

        # project on Hypercube(J)
        z_t = torch.clamp(z_new, 0, 1)

        # Update best value
        new_best_value = val_best_i < val_i
        z_best_i_j[new_best_value] = z_new[new_best_value]
        val_best_i[new_best_value] = val_i[new_best_value]

    ### Take the best value
    z_star = z_best_i_j
    bundle_star = (z_star.round() > 0).bool()

    # random_tensor = torch.rand_like(z_star)
    # bundle_star = (random_tensor < z_star).bool()
    # violations_rounding = ((z_star > 0.25) & (z_star < 0.75)).sum(1).cpu().numpy()
    # print('violations of rounding in SFM: ', violations_rounding.sum(),
    #       ', mean: ', violations_rounding.mean(),
    #       ', std : ', violations_rounding.std(),
    #       ', max: ', violations_rounding.max())
    # print('gradiend min:', grad_i_j.min().cpu().numpy(), 'max:', grad_i_j.max().cpu().numpy())
    # print('step lenght gradient: ', torch.norm(grad_i_j, dim=1, keepdim=True).mean().cpu().numpy())


def _grad_lovatz_extension(i_idx, z_i_j, lambda_k, p_j, return_fun_value = True):

    ### Sort z_i_j for each i
    sorted_z_id_j = torch.argsort(z_i_j, dim=1, descending=True)

    self.zeros_i_j_j[i_idx.unsqueeze(1).unsqueeze(2) ,
                      sorted_z_id_j.unsqueeze(1) ,
                      sorted_z_id_j.unsqueeze(2)] = self.upper_triangular

    mask = self.zeros_i_j_j.unsqueeze(-1)

    ### Compute gradient
    # Supermodular
    grad_i_j = torch.matmul((self.P_j_j_k.unsqueeze(0) * mask).sum(-2) , lambda_k[self.K_MOD : -self.K_QS_i])
    grad_i_j += torch.matmul((self.P_i_j_j_k * mask).sum(-2) , lambda_k[-self.K_QS_i:])

    # Modular
    grad_i_j += self.φ_i_j_k[i_idx] @ lambda_k[:self.K_MOD]

    # Add heterogeneity
    if eps_i_j is not None:
        grad_i_j  += eps_i_j

    # Add prices
    if p_j is not None:
        grad_i_j += p_j[None, :]

    if return_fun_value:
        # Compute the value of the Lovatz extension (assuming the value of the empty bundle is 0)
        fun_value_i = (z_i_j * grad_i_j).sum(1)

        return grad_i_j, fun_value_i
    else:
        return grad_i_j