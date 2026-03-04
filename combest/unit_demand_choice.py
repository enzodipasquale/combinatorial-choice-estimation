import numpy as np
import pandas as pd
from combest.core import Model
from combest.loaders import load_market_data

from combest.utils import get_logger

logger = get_logger(__name__)


class UnitDemandChoice(Model):

    def load_product_data(self, product_data, availability_data=None):
        id_data, n_obs, n_items = load_market_data(
            product_data, availability_data=availability_data, covariates=[])
        id_data["constraint_mask"] = id_data["available_items"]
        self._input_data = {"id_data": id_data, "item_data": {}}

        # store product-level mappings for set_specification
        n_products = len(product_data)
        self._product_data = product_data
        self._product_market_ids = id_data["market_id"][:n_products]
        self._product_item_ids = np.argmax(id_data["obs_bundles"][:n_products], axis=1)

        n_markets = int(self._product_market_ids.max()) + 1
        self._n_markets = n_markets

        # fixed effect map from availability: (n_markets, n_items) → FE index
        _, first_idx = np.unique(self._product_market_ids, return_index=True)
        market_avail = id_data["available_items"][first_idx]
        self._FE_map = np.full((n_markets, n_items), -1, dtype=np.int64)
        self._FE_map[market_avail] = np.arange(market_avail.sum())
        self._n_FEs = int(market_avail.sum())

        self.load_config({
            "dimensions": {"n_obs": n_obs, "n_items": n_items},
            "subproblem": {"name": "UnitDemand"},
        })
        self.subproblems.load_solver()


    def load_demographics(self, market_demographics, demographics=None):
        if self.comm_manager.is_root():
            nu_cols = sorted(c for c in market_demographics.columns if c.startswith('nodes'))
            demo_cols = list(demographics) if demographics is not None else [
                c for c in market_demographics.columns if c not in {'market_id', 'weights'} | set(nu_cols)]
            _, inv = np.unique(market_demographics['market_id'].values, return_inverse=True)
            counts = np.bincount(inv)
            if not np.all(counts == counts[0]):
                raise ValueError("Unbalanced simulations across markets not yet supported.")
            n_sims, n_markets = int(counts[0]), len(counts)
            self._nu = market_demographics[nu_cols].values.astype(np.float64).reshape(n_markets, n_sims, -1)
            self._demos = market_demographics[demo_cols].values.astype(np.float64).reshape(n_markets, n_sims, -1) if demo_cols else None
            parsed = (self._nu, self._demos, n_sims, demo_cols)
        else:
            parsed = None
        self._nu, self._demos, n_sims, self._demo_names = self.comm_manager.bcast(parsed)

        self.load_config({"dimensions": {"n_simulations": n_sims}})
        self.data.load_and_distribute_input_data(self._input_data)
        del self._input_data

        self._log_dims(n_sims)

    def set_specification(self, specification):
        self._specification = specification

        # build characteristics tensor: (n_markets, n_items, n_covariates)
        n_products = len(self._product_data)
        df_cols = [c for c in specification if c != "constant"]
        raw = self._product_data[df_cols].values.astype(np.float64) if df_cols else np.empty((n_products, 0))
        if "constant" in specification:
            raw = np.insert(raw, specification.index("constant"), 1.0, axis=1)
        self._characteristics = np.zeros((self._n_markets, self.n_items, len(specification)))
        self._characteristics[self._product_market_ids, self._product_item_ids] = raw

        if self.comm_manager.is_root():
            logger.info(" SPECIFICATION: %s", ", ".join(specification))

    def build_rc_error_oracle(self, sigma, pi=None, rc_covariates=None, seed=42):
        sigma = np.atleast_2d(sigma)

        # select K2 subset of characteristics for random coefficients
        if rc_covariates is not None:
            rc_idx = [self._specification.index(c) for c in rc_covariates]
            characteristics = self._characteristics[..., rc_idx]
        else:
            rc_covariates = list(self._specification)
            characteristics = self._characteristics

        market_idx = self.data.local_data["id_data"]["market_id"]
        sim_idx = self.comm_manager.agent_ids // self.config.dimensions.n_obs

        # c_i = Σ ν_i + Π d_i
        c = self._nu[market_idx, sim_idx] @ sigma.T
        if pi is not None and self._demos is not None:
            c += self._demos[market_idx, sim_idx] @ np.atleast_2d(pi).T

        # ε_ij = x_j' c_i + gumbel_ij
        rc_errors = np.einsum('ijk,ik->ij', characteristics[market_idx], c)
        gumbel = np.zeros((self.comm_manager.num_local_agent, self.config.dimensions.n_items))
        for i, agent_id in enumerate(self.comm_manager.agent_ids):
            gumbel[i] = np.random.default_rng((seed, agent_id)).gumbel(0, 1, self.config.dimensions.n_items)

        self.features.local_modular_errors = rc_errors + gumbel
        self.features._error_oracle = lambda bundles, ids: (self.features.local_modular_errors[ids] * bundles).sum(-1)
        self.features._error_oracle_vectorized = True
        self.features._error_oracle_takes_data = False

        if self.comm_manager.is_root():
            dim_nu = sigma.shape[1]
            n_demo = self._demos.shape[-1] if self._demos is not None else 0
            self._log_rc(dim_nu, n_demo, self._demo_names, rc_covariates)

    def setup_FE_estimation(self):
        FE_map, n_FEs = self._FE_map, self._n_FEs

        # configure for FE estimation
        self.config.dimensions.n_covariates = n_FEs
        self.config.dimensions._build_labels()
        self.config.row_generation.theta_lbs = -1000
        self.config.row_generation.theta_ubs = 1000
        self.config.row_generation.master_gurobi_params["Threads"] = 0
        self.config.row_generation.master_gurobi_params["Method"] = -1


        # store FE index in local data for solver
        market_idx = self.data.local_data["id_data"]["market_id"]
        self.data.local_data["id_data"]["fe_index"] = FE_map[market_idx]

        fe_idx = self.data.local_data["id_data"]["fe_index"]
        def FE_oracle(bundles, ids):
            idx = (bundles * fe_idx[ids]).sum(1)
            out = np.zeros((len(ids), n_FEs))
            m = bundles.any(1)
            out[m, idx[m]] = 1.0
            return out

        self.features._covariates_oracle = FE_oracle
        self.features._covariates_oracle_vectorized = True
        self.features._covariates_oracle_takes_data = False

    def estimate_fixed_effects(self, **kwargs):
        result = self.row_generation.solve(**kwargs)

        # extract delta: one per product observation
        if self.comm_manager.is_root():
            FE_map = self._FE_map
            fe_indices = FE_map[self._product_market_ids, self._product_item_ids]
            return result.theta_hat[fe_indices]

    def _log_dims(self, n_sims):
        if not self.comm_manager.is_root():
            return
        n_obs = self.config.dimensions.n_obs
        n_items = self.config.dimensions.n_items
        n_markets = len(np.unique(self.data.local_data["id_data"]["market_id"]))
        header = f"{'n_markets':>10} | {'n_obs':>6} | {'n_items':>8} | {'n_simulations':>14}"
        values = f"{n_markets:>10} | {n_obs:>6} | {n_items:>8} | {n_sims:>14}"
        logger.info(" PRODUCT DATA")
        logger.info("-" * 50)
        logger.info(header)
        logger.info(values)
        logger.info("-" * 50)

    def _log_rc(self, dim_nu, n_demographics, demo_names, rc_covariates):
        header = f"{'dim_nu':>8} | {'n_demographics':>15}"
        values = f"{dim_nu:>8} | {n_demographics:>15}"
        logger.info(" RANDOM COEFFICIENTS")
        logger.info("-" * 28)
        logger.info(header)
        logger.info(values)
        logger.info("-" * 28)
        if rc_covariates:
            logger.info(" rc covariates: %s", ", ".join(rc_covariates))
        if demo_names:
            logger.info(" demographics: %s", ", ".join(demo_names))
