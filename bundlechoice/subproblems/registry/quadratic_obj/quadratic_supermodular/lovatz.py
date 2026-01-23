import numpy as np
from ....subproblem_base import BatchSubproblemBase
from .supermodular_quadratic_obj_base import SupermodularQuadraticObjectiveMixin

class QuadraticSupermodularLovasz(SupermodularQuadraticObjectiveMixin, BatchSubproblemBase):

    def solve(self, theta):
        linear, quadratic = self.build_linear_and_quadratic_coef(theta)
        P = quadratic.copy()
        diag = np.arange(self.dimensions_cfg.n_items)
        P[:, diag, diag] += linear
        mask = self._qinfo.constraint_mask
        num_iters = int(self.subproblem_cfg.GRB_Params.get(
            'num_iters_SGM', max(100000, 1000 * self.dimensions_cfg.n_items)))
        alpha = float(self.subproblem_cfg.GRB_Params.get(
            'alpha', 0.1 / np.sqrt(self.dimensions_cfg.n_items)))
        z = np.full((self.data_manager.num_local_agent, self.dimensions_cfg.n_items), 0.5)
        if mask is not None:
            z[~mask] = 0.0
        z_best = z.copy()
        val_best = np.full(self.data_manager.num_local_agent, -np.inf)
        tril = np.tril(np.ones((self.dimensions_cfg.n_items, self.dimensions_cfg.n_items), dtype=bool))
        for _ in range(num_iters):
            grad, val = self.batched_grad_lovasz_extension(z, P, tril)
            if mask is not None:
                grad[~mask] = 0.0
            norm = np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-10)
            z = np.clip(z + alpha / norm * grad, 0.0, 1.0)
            improved = val > val_best
            z_best[improved] = z[improved]
            val_best[improved] = val[improved]
        return z_best > 0.5

    def batched_grad_lovasz_extension(self, z, P, tril):
        n_agents, n_items = z.shape
        sigma = np.argsort(z, axis=1)[:, ::-1]
        P_s = np.take_along_axis(np.take_along_axis(P, sigma[:, :, None], axis=1), sigma[:, None, :], axis=2)
        P_sym = P_s + P_s.transpose(0, 2, 1)
        diag = np.arange(n_items)
        P_sym[:, diag, diag] = np.diagonal(P_s, axis1=1, axis2=2)
        grad_s = (P_sym * tril[None, :, :]).sum(axis=2)
        grad = np.zeros_like(z)
        np.put_along_axis(grad, sigma, grad_s, axis=1)
        return grad, (z * grad).sum(axis=1)
