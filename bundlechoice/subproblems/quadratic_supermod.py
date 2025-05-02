# def init_USM(self, local_id):
#     raise NotImplementedError("init_USM is not implemented yet.")

# def solve_USM(self, subproblem, local_id, lambda_k, p_j):
#     raise NotImplementedError("solve_USM is not implemented yet.")



def solve_QS(self,i_idx, lambda_k, p_j = None, num_iters_SGD, alpha,
                          method = 'constant_step_lenght'):

    # z_best_i_j = self.z_0[i_idx,:].clone()
    # z_t = self.z_0[i_idx,:].clone()
    # val_best_i = self.val_best_i

    z_t = torch.ones(len(i_idx), self.J, device='cuda', dtype=torch.float) / 2
    z_best_i_j =  torch.zeros(len(i_idx), self.J, device='cuda', dtype=torch.float)
    val_best_i = torch.ones(len(i_idx), device='cuda', dtype=torch.float) * (- np.inf)

    for iter in range(num_iters_SGD):

        ### Compute gradient
        grad_i_j , val_i = self.grad_lovatz_extension(i_idx, z_t, lambda_k, eps_i_j, p_j, return_fun_value = True)

        ### Compute proj_H (z_t + α g_t) to z_list

        # Take step z_t + α g_t
        if method == 'constant_step_lenght':
            z_new = z_t + alpha * grad_i_j / torch.norm(grad_i_j, dim=1, keepdim=True)

        elif method == 'constant_step_size':
            z_new = z_t + alpha * grad_i_j

        elif method == 'constant_over_sqrt_k':
            z_new = z_t + alpha * grad_i_j / torch.sqrt(iter + 1)

        elif method == 'mirror_descent':
            z_new = z_t * torch.exp(alpha * grad_i_j /  torch.norm(grad_i_j, dim=1, keepdim=True) )

        if not knapsack:
            # project on Hypercube(J)
            z_new = torch.clamp(z_new, 0, 1)
        else:
            # project on Hypercube(J) intersected with knapsack constraint
            z_new = self.knapsack_projection(i_idx, z_new, bisection_iters)

        z_t = z_new

        # Update best value
        new_best_value = val_best_i < val_i
        z_best_i_j[new_best_value] = z_new[new_best_value]
        val_best_i[new_best_value] = val_i[new_best_value]

    ### Take the best value
    z_star = z_best_i_j
    bundle_star = (z_star.round() > 0).bool()

    # random_tensor = torch.rand_like(z_star)
    # bundle_star = (random_tensor < z_star).bool()
    violations_rounding = ((z_star > 0.25) & (z_star < 0.75)).sum(1).cpu().numpy()
    print('violations of rounding in SFM: ', violations_rounding.sum(),
          ', mean: ', violations_rounding.mean(),
          ', std : ', violations_rounding.std(),
          ', max: ', violations_rounding.max())
    print('gradiend min:', grad_i_j.min().cpu().numpy(), 'max:', grad_i_j.max().cpu().numpy())
    print('step lenght gradient: ', torch.norm(grad_i_j, dim=1, keepdim=True).mean().cpu().numpy())
    if return_value:
        return bundle_star, z_star, val_best_i
    return bundle_star, z_star