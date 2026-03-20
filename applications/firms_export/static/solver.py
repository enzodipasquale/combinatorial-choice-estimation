import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from combest import SubproblemSolver
from combest.utils import suppress_output

FEATURE_KEYS = [
    ("modular_3d", "id_data"), ("modular_1d", "item_data"),
    ("modular_2d", "item_data"), ("entry_1d", "item_data"),
    ("quadratic_2d", "item_data"), ("quadratic_3d", "item_data"),
    ("consec_1d", "item_data"),
]

COVARIATE_NAMES = {
    "modular_3d": ["revenue"],
    "modular_1d": ["oper×dist"],
    "entry_1d":   ["entry"],
    "quadratic_2d": ["proximity", "prox×dist"],
    "consec_1d":  ["consec×dist"],
}

COVARIATE_LBS = {
    "quadratic_2d": -1.0,
    "consec_1d": 0.0,
}


class DiscountedJointQuadKnapsackSolver(SubproblemSolver):

    def initialize(self):
        ld = self.data_manager.local_data
        self._nd = ld["item_data"]["n_dest"]
        self._ny = ld["item_data"]["n_years"]
        self._dw = ld["id_data"]["discount_weights"]
        self._edw = ld["id_data"]["entry_discount_weights"]
        off, self._sl = 0, {}
        for key, src in FEATURE_KEYS:
            arr = ld[src].get(key)
            if arr is not None:
                self._sl[key] = slice(off, off + arr.shape[-1])
                off += arr.shape[-1]

        mask = ld["id_data"].get("constraint_mask")
        caps = ld["id_data"]["capacity"]
        nd, ny, n_items = self._nd, self._ny, self.dimensions_cfg.n_items
        self._models = []
        for i in range(self.comm_manager.num_local_agent):
            with suppress_output():
                m = gp.Model()
                m.setParam('OutputFlag', 0)
                m.setParam('Threads', 1)
                for k, v in self.subproblem_cfg.gurobi_params.items():
                    if v is not None:
                        m.setParam(k, v)
                m.setAttr('ModelSense', gp.GRB.MAXIMIZE)
                B = m.addMVar(n_items, vtype=gp.GRB.BINARY, name='b')
                for t in range(ny):
                    m.addConstr(B[t*nd:(t+1)*nd].sum() <= caps[i])
                if mask is not None:
                    fixed = np.where(~mask[i])[0]
                    if len(fixed):
                        m.addConstr(B[fixed] == 0)
                m.update()
            self._models.append(m)

    def _build_linear(self, theta):
        L = self.features_manager.local_modular_errors.copy()
        ld = self.data_manager.local_data
        dw, n = self._dw, L.shape[0]
        if 'modular_3d' in self._sl:
            v = ld["id_data"]['modular_3d'] @ theta[self._sl['modular_3d']]
            L += (dw[:, :, None] * v).reshape(n, -1)
        if 'modular_1d' in self._sl:
            v = ld["item_data"]['modular_1d'] @ theta[self._sl['modular_1d']]
            L += (dw[:, :, None] * v[None, None, :]).reshape(n, -1)
        if 'modular_2d' in self._sl:
            v = ld["item_data"]['modular_2d'] @ theta[self._sl['modular_2d']]
            L += (dw[:, :, None] * v[None, :, :]).reshape(n, -1)
        if 'entry_1d' in self._sl:
            v = ld["item_data"]['entry_1d'] @ theta[self._sl['entry_1d']]
            L += (self._edw[:, :, None] * v[None, None, :]).reshape(n, -1)
        return L

    def _build_quadratic(self, theta, i):
        ld = self.data_manager.local_data
        nd, ny, dw = self._nd, self._ny, self._dw
        n_items = nd * ny
        rows, cols, vals = [], [], []

        if 'quadratic_2d' in self._sl:
            Qd = ld["item_data"]['quadratic_2d'] @ theta[self._sl['quadratic_2d']]
            rr, cc = np.nonzero(Qd)
            vv = Qd[rr, cc]
            for t in range(ny):
                w = dw[i, t]
                if w == 0:
                    continue
                off = t * nd
                rows.append(rr + off); cols.append(cc + off); vals.append(vv * w)

        if 'quadratic_3d' in self._sl:
            Qd = ld["item_data"]['quadratic_3d'] @ theta[self._sl['quadratic_3d']]
            for t in range(ny):
                w = dw[i, t]
                if w == 0:
                    continue
                rr, cc = np.nonzero(Qd[t])
                off = t * nd
                rows.append(rr + off); cols.append(cc + off)
                vals.append(Qd[t][rr, cc] * w)

        if 'consec_1d' in self._sl:
            v = ld["item_data"]['consec_1d'] @ theta[self._sl['consec_1d']]
            jj = np.arange(nd)
            for t in range(1, ny):
                w = dw[i, t]
                if w == 0:
                    continue
                half = v * w / 2
                rows.append(t*nd + jj);     cols.append((t-1)*nd + jj); vals.append(half)
                rows.append((t-1)*nd + jj); cols.append(t*nd + jj);     vals.append(half)

        if rows:
            return sp.csr_matrix(
                (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                shape=(n_items, n_items))
        return sp.csr_matrix((n_items, n_items))

    def solve(self, theta):
        L = self._build_linear(theta)
        n_items = self.dimensions_cfg.n_items
        results = np.zeros((len(self._models), n_items), dtype=bool)
        for i, m in enumerate(self._models):
            Q = self._build_quadratic(theta, i)
            m.setMObjective(Q, L[i], 0.0, sense=gp.GRB.MAXIMIZE)
            m.optimize()
            results[i] = np.array(m.x) > 0.5
        return results

    def update_solver_settings(self, settings_dict):
        for m in self._models:
            for k, v in settings_dict.items():
                m.setParam(k, v)


def discounted_covariates_oracle(bundles, ids, data):
    id_d, it_d = data.id_data, data.item_data
    nd, ny = it_d["n_dest"], it_d["n_years"]
    B = bundles.reshape(-1, ny, nd).astype(float)
    dw = id_d["discount_weights"][ids]
    dB = dw[:, :, None] * B

    feats = []
    if 'modular_3d' in id_d:
        feats.append(np.einsum('itj,itjk->ik', dB, id_d['modular_3d'][ids]))
    if 'modular_1d' in it_d:
        feats.append(np.einsum('itj,jk->ik', dB, it_d['modular_1d']))
    if 'modular_2d' in it_d:
        feats.append(np.einsum('itj,tjk->ik', dB, it_d['modular_2d']))
    if 'entry_1d' in it_d:
        edw = id_d["entry_discount_weights"][ids]
        feats.append(np.einsum('itj,jk,it->ik', B, it_d['entry_1d'], edw))
    if 'quadratic_2d' in it_d:
        feats.append(np.einsum('itj,jlk,itl,it->ik', B, it_d['quadratic_2d'], B, dw))
    if 'quadratic_3d' in it_d:
        feats.append(np.einsum('itj,tjlk,itl,it->ik', B, it_d['quadratic_3d'], B, dw))
    if 'consec_1d' in it_d:
        B_lag = np.zeros_like(B)
        B_lag[:, 1:, :] = B[:, :-1, :]
        feats.append(np.einsum('itj,jk,it->ik', B * B_lag, it_d['consec_1d'], dw))
    return np.concatenate(feats, axis=-1)


def discount_errors(model, n_dest):
    dw = model.data.local_data.id_data["discount_weights"]
    model.features.local_modular_errors *= np.repeat(dw, n_dest, axis=1)


def count_covariates(input_data):
    return sum(input_data[src][key].shape[-1]
               for key, src in FEATURE_KEYS if key in input_data[src])
